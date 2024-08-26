from models import Siren,PEMLP
from util.logger import log
from util.tensorboard import writer
from util.misc import gen_cur_time
from util import io
from trainer.base import BaseTrainer
from components import frequency as freq
from components.dct import DCTImageProcessor
from components.ssim import compute_ssim_loss
from components.laplacian import compute_laplacian_loss

import numpy as np
import imageio.v2 as imageio
import torch
import os

from tqdm import trange

import matplotlib.pyplot as plt


## 先写出来 后面再做抽象环和模块化
## todo: 重构: image trainer → 继承 freq_image_trainner → 
class ImageTrainer(BaseTrainer):
    def __init__(self, args, mode="train"):
        super().__init__(args,mode)
        self.proc = DCTImageProcessor()
        self._parse_input_data()

    def _encode_img_data(self, img:np.ndarray):
        assert img.ndim == 3 # h,w,c
        img = np.clip(img,0,255) 
        img = img / 255.
        if self.args.zero_mean: 
            img = self.encode_zero_mean(img)
        return img

    def _decode_img_data(self, data:np.ndarray):
        assert data.ndim == 3 # h,w,c
        if self.args.zero_mean: 
            data = self.decode_zero_mean(data)
        data = np.clip(data,0,1)
        data  = data * 255.
        return data
    
    def _normalize_data(self, data:np.ndarray):
        self._max = np.max(data) 
        self._min = np.min(data)
        return (data - self._min) / (self._max - self._min)
    
    def _denormalize_data(self, normalized_data:np.ndarray):
        return normalized_data * (self._max - self._min) + self._min
    
    def _parse_input_data(self):
        path = self.args.input_path
        img = np.array(imageio.imread(path), dtype=np.float32)
        self.raw_img = img
        self.H, self.W, self.C =  img.shape

    def _get_data(self):
        '''获取原始图像数据'''
        img = self.raw_img
        
        input_img  = self._encode_img_data(img)
        gt = torch.tensor(input_img).view(-1, self.C)
        coords = torch.stack(torch.meshgrid([torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)], indexing='ij'), dim=-1).view(-1, 2)
        # coords | -1 → 1 | 512*512, 2
        return coords, gt

    def _get_coff(self):
        '''获取全部dct系数数据'''
        self.proc.set_img_data(self.raw_img)
        dct_img = self.proc.apply_dct_to_image()
        input_dct = self._normalize_data(dct_img)
        input_dct = self.encode_zero_mean(input_dct)
        gt = torch.tensor(dct_img).view(-1, self.C)
        coords = torch.stack(torch.meshgrid([torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)], indexing='ij'), dim=-1).view(-1, 2)
        return coords, gt

    def _get_low_img(self):
        '''获取低频dct的空域数据'''
        img = self.raw_img
        self.proc.set_img_data(img)
        # raw dct img
        dct_img = self.proc.apply_dct_to_image()
        self.dct_img = dct_img
        mask = self.proc.gen_filter_mask(0,4)
        low_img = self.proc.apply_idct_to_image(dct_img, mask)
        io.save_cv2(low_img, self._get_cur_path('low_img.png'))

        ## 可视化dct数据
        self.visualize_dct(dct_img)

        input_img  = self._encode_img_data(low_img)
        gt = torch.tensor(input_img).view(-1, self.C)
        coords = torch.stack(torch.meshgrid([torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)], indexing='ij'), dim=-1).view(-1, 2)
        return coords, gt
    
    def _get_high_coff(self):
        '''不经过permuetd获取dct高频系数 reconstruct这块还没写 不好写 重构的时候再说吧'''
        dct_img = self.dct_img
        h, w, c = dct_img.shape
        mask = self.proc.gen_filter_mask(0,4)

        input_dct = self._normalize_data(dct_img)
        input_dct = self.encode_zero_mean(input_dct)

        gt = torch.zeros(int(h*w*(64-16)/64),c) #系数数量计算 此处先流氓数
        coords = torch.zeros(int(h*w*(64-16)/64),2) # xy
        x = torch.linspace(-1,1,h)
        y = torch.linspace(-1,1,w)

        global_idx = 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = dct_img[i:i+8, j:j+8]
                for k in range(0, 8):
                    for l in range(0, 8):
                        if not mask[k][l]: 
                            x_index = i + k
                            y_index = j + l
                            coords[global_idx] = torch.tensor([x[x_index], y[y_index]])
                            gt[global_idx] = torch.tensor(input_dct[x_index][y_index])
                            global_idx += 1
        
        return coords, gt


    def _get_permuted_coff(self):
        '''获取高频dct的系数数据(经过permuted)'''
        dct_img = self.dct_img
        _mask = self.proc.gen_filter_mask(0,4)
        permuted_dct = self.permute_dct(dct_img, _mask)

        # 可视化block dct
        self.visualize_block_dct(permuted_dct)

        ######### 测试permuted是否能恢复
        # 生成低频部分
        # low_dct = self.permute_dct(dct_img,~_mask)
        # 重构回来
        # r_dct = self.de_permute_dct(permuted_dct, low_dct)
        # r_img = self.proc.apply_idct_to_image(r_dct)
        # _pnsr = self.compute_raw_psnr(r_img, self.raw_img) #目前已经恢复回来
        #############

        # normalization
        input_dct = self._normalize_data(permuted_dct)
        input_dct = self.encode_zero_mean(input_dct)

        # 最简单按照块走 # 
        # todo: 也可以纯按照xy映射 看看哪个好
        block_a = 8
        intra_a = int(np.sqrt(input_dct.shape[1])) # sqrt(4096) = 64

        X = torch.linspace(-1,1,block_a) # 8x8 dct
        Y = torch.linspace(-1,1,block_a)
        # x,y 先咋那是按照block_size内部的序走 (可能会太稀疏了 后续看看实验) 
        x = torch.linspace(-1,1,intra_a)
        y = torch.linspace(-1,1,intra_a)
        gt = torch.tensor(input_dct).view(-1, self.C) # 直接排列就行
        coords = torch.zeros((input_dct.shape[0], input_dct.shape[1], 4))

        self.set_block_coff_idx_matrix()
        
        for idx_b in range(coords.shape[0]):
            for idx_intra in range(coords.shape[1]):
                ## block_idx现在先简单处理下 后续可以更精准
                # X_idx = idx_b // block_a
                # Y_idx = idx_b % block_a
                X_idx, Y_idx = self.get_row_col_from_masking_block_by_idx(idx_b)
                x_idx = idx_intra // intra_a
                y_idx = idx_intra % intra_a
                coords[idx_b][idx_intra] = torch.tensor([X[X_idx], Y[Y_idx], x[x_idx], y[y_idx]])
        

        ##### 纯按照xy走
        # coords = torch.zeros((input_dct.shape[0], input_dct.shape[1], 2))
        # x = torch.linspace(-1,1,self.H)
        # y = torch.linspace(-1,1,self.W)
        # for idx_b in range(coords.shape[0]):
        #     for idx_intra in range(coords.shape[1]):
        #         ## todo: block_idx现在先简单处理下 后续可以更精准 (感觉提升)
        #         x_idx = idx_b // block_a + (idx_intra // intra_a)
        #         y_idx = idx_b % block_a + (idx_intra % intra_a)
        #         coords[idx_b][idx_intra] = torch.tensor([x[x_idx], y[y_idx]])

        coords = coords.reshape(-1, coords.shape[-1])
        
        # '''ablation 打乱顺序'''
        # shuffled_indexs = torch.randperm(coords.shape[0])
        # coords = coords.index_select(0, shuffled_indexs)


        return coords, gt


        
    def drp_get_data(self, use_dct=False):
        '''drp舍弃的函数 | 暂时不要删'''
        path = self.args.input_path

        img = np.array(imageio.imread(path), dtype=np.float32)
        H, W, C = img.shape
        self.raw_img = img
        
        #注意dct有一个溢值的问题

        self.proc.set_img_data(img)
        dct_img = self.proc.apply_dct_to_image()
        self.dct_img = dct_img
    
        

        mask = self.proc.gen_filter_mask(0,4)
        low_img = self.proc.apply_idct_to_image(dct_img, mask)
        
        io.save_cv2(low_img, self._get_cur_path('low_img.png'))

        ### test permuted
        _mask = self.proc.gen_filter_mask(0,4)
        permuted_dct = self.permute_dct(dct_img, _mask)
        self.visualize_block_dct(permuted_dct)

        ######### 测试permuted是否能恢复
        # 生成低频部分
        # low_dct = self.permute_dct(dct_img,~_mask)
        # 重构回来
        # r_dct = self.de_permute_dct(permuted_dct, low_dct)
        # r_img = self.proc.apply_idct_to_image(r_dct)
        # _pnsr = self.compute_raw_psnr(r_img, self.raw_img) #目前没有恢复回来
        #############
        

        self.visualize_dct(dct_img)

        if not use_dct:
            # 正常图像做GT
            input_img  = self._encode_img_data(low_img)
            gt = torch.tensor(input_img).view(-1, C)
        else:
            # dct系数做 gt 预处理 (预实验) 后面丢弃掉
            input_dct = self._normalize_data(dct_img)
            input_dct = self.encode_zero_mean(input_dct)
            gt = torch.tensor(dct_img).view(-1, C)
            

        coords = torch.stack(torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)], indexing='ij'), dim=-1).view(-1, 2)
        
        return coords, gt,(H,W,C)
    

    # overwrite
    def train(self):
        
        num_epochs = self.args.num_epochs
        use_coff_net = self.args.use_coff_net

        
        # base_model transfer2cuda
        coords, gt = self._get_low_img() if use_coff_net else self._get_data()
        model = self._get_model(in_features=2, out_features=3).to(self.device)
        coords = coords.to(self.device)
        gt = gt.to(self.device)
        # self.raw_img = torch.tensor(self.raw_img).to(self.device)
        optimizer = torch.optim.Adam(lr=self.args.lr, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1))

        # coff_model
        if use_coff_net:
            coff_coords, coff_gt = self._get_permuted_coff()
            # coff_coords, coff_gt = self._get_high_coff()
            coff_model = self._get_model(in_features=4, out_features=3).to(self.device)
            coff_coords = coff_coords.to(self.device)
            coff_gt = coff_gt.to(self.device)
            optimizer_c = torch.optim.Adam(lr=self.args.lr, params=coff_model.parameters())
            scheduler_c = torch.optim.lr_scheduler.LambdaLR(optimizer_c, lambda iter: 0.1 ** min(iter / num_epochs, 1))

            # 兜底逻辑 以免pred_c没变量 重建的时候有问题
            pred_c = torch.zeros_like(coff_gt)

        for epoch in trange(1, num_epochs + 1): 

            log.start_timer("train")
            _train_coff = self.train_coff_net(epoch)
            
            # inference 
            if not _train_coff:
                pred = model(coords)
                freq_loss = self._compute_freq_loss(pred, gt)
                # freq_loss = 0
                loss = 1e-5 * freq_loss + self._compute_loss(pred, gt)
                psnr = self._eval_performance(pred, gt, "psnr")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            else:
                pred_c = coff_model(coff_coords)
                loss_c = self._compute_loss(pred_c, coff_gt)

                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()
                scheduler_c.step()


            torch.cuda.synchronize() # 确保测量时间
            log.pause_timer("train")

            # record | writer
            if epoch % self.args.log_epoch == 0:
                # log.inst.info(f"epoch: {epoch} → PSNR {psnr:.4f}")
                # log.inst.info(f"epoch: {epoch} → loss {loss:.8f}")
                if _train_coff: log.inst.info(f"epoch: {epoch} → loss_c {loss_c:.8f}")
                else: 
                    log.inst.info(f"epoch: {epoch} → loss {loss:.8f}")
                    log.inst.info(f"epoch: {epoch} → PSNR {psnr:.4f}")

                ######## 低频信号训练inr+直接保存无损高频系数
                # r_low_img = self._decode_img_data(pred.reshape(H,W,C).cpu().detach().numpy())
                # io.save_cv2(r_low_img, self._get_cur_path('r_low_img.png'))
                # self.proc.set_img_data(r_low_img)
                # dct_low_img = self.proc.apply_dct_to_image()
                # mask = self.proc.gen_filter_mask(0,4)
                # r_dct = self.proc.reconstruct_dct_coff(dct_low_img,self.dct_img,mask)
                # r_img = self.proc.apply_idct_to_image(r_dct)
                # io.save_cv2(r_img, self._get_cur_path('r_img.png'))
                
                # raw_psnr = self.compute_raw_psnr(r_img, self.raw_img)
                # log.inst.info(f"epoch: {epoch} → Real PSNR {raw_psnr:.4f}")
                ######################################################## 

                ###### 判断pred 和 gt同样恢复后psnr是否可比: 结论: 一般恢复后的psnr会高~0.01 整体差不多
                # r_gt = self._decode_img_data(gt.reshape(H,W,C).cpu().detach().numpy())
                # raw_comp_psnr = self.compute_raw_psnr(r_low_img, r_gt)
                # log.inst.info(f"epoch: {epoch} → COMPARE: Real PSNR {raw_comp_psnr:.4f}")
                ######################################################## 

                ####### 直接fit dct做图片信息恢复
                # decoded_dct = self._denormalize_data(self.decode_zero_mean(pred.reshape(self.H, self.W, self.C).cpu().detach().numpy()))
                # r_img = self.proc.apply_idct_to_image(decoded_dct)
                # raw_comp_psnr = self.compute_raw_psnr(r_img, self.raw_img)
                # io.save_cv2(r_img, self._get_cur_path('pure_fit_dct_img.png'))
                # log.inst.info(f"epoch: {epoch} → COMPARE: Real PSNR {raw_comp_psnr:.4f}")
                ########################################################

                ######## 根据两个空域像素和dct系数恢复图像
                if use_coff_net:
                    r_permuted_dct = pred_c.reshape(-1, 64*64 , pred_c.shape[-1]).cpu().detach().numpy()
                    r_permuted_dct = self._denormalize_data(self.decode_zero_mean(r_permuted_dct))
                    # idct 低频
                    r_low_img = self._decode_img_data(pred.reshape(self.H, self.W, self.C).cpu().detach().numpy())
                    io.save_cv2(r_low_img, self._get_cur_path('r_low_img.png'))
                    self.proc.set_img_data(r_low_img)
                    dct_low_img = self.proc.apply_dct_to_image()
                    _mask = self.proc.gen_filter_mask(0,4)
                    low_dct = self.permute_dct(dct_low_img, ~_mask)
                    r_dct = self.de_permute_dct(r_permuted_dct, low_dct)
                    r_img = self.proc.apply_idct_to_image(r_dct)
                    real_pnsr = self.compute_raw_psnr(r_img, self.raw_img) 
                    io.save_cv2(r_img, self._get_cur_path('real_fit_img.png'))
                    log.inst.info(f"epoch: {epoch} → COMPARE: Real PSNR {real_pnsr:.4f}")


            # writer.inst.add_scalar("train/loss", loss.detach().item(), global_step=epoch)
            # writer.inst.add_scalar("train/psnr", psnr.detach().item(), global_step=epoch)

        with torch.no_grad():
            final_pred = model(coords).reshape(self.H, self.W, self.C).cpu()

            final_pred = io.cvt_tensor_np(final_pred)
            io.save_cv2(self._decode_img_data(final_pred), self._get_cur_path("final_pred.png"))

        # save ckpt
        self.model = model # 暂时先存base model
        self._save_ckpt(epoch,optimizer,scheduler)
    
    def _compute_freq_loss(self, pred, gt):
        # 很烦  后续全改成 torch数据就没问题了
        # reconstruct pred 2 img 然后直接用self.raw_img即可:
        r_img = pred.reshape(self.H, self.W, self.C)
        # 参考_decode_img_data 只不过是torch 重构的时候能用torch用torch吧
        # decode
        r_img = r_img / 2 + 0.5
        r_img = torch.clamp(r_img, min=0.0, max=1.0)
        r_img  = r_img * 255.

        gt_img = gt.reshape(self.H, self.W, self.C)
        gt_img = gt_img / 2 + 0.5
        gt_img = torch.clamp(gt_img, min=0.0, max=1.0)
        gt_img  = gt_img * 255.

        # loss = compute_ssim_loss(r_img, gt_img)
        loss = compute_laplacian_loss(r_img, gt_img)
        # loss = self.compute_mse(r_img, gt_img)
        # print('pixel level 255 loss: ', loss)
        return loss
    

    # deprecated
    def subband_partition(self, img):
        # 先写针对于img结构的 后续扩展到任意维度
         # self.raw_gt_img = img
    
        # if self.args.subband: 
        #     band_data, band_coff = self.subband_partition(self._decode_img_data(img))
        #     # 取低频段 先训
        #     img = band_data[0]
        #     psnr = self._eval_performance(img, self.raw_gt_img ,"psnr")
        #     print('psnr: ', psnr)
        #     io.save_cv2(self._decode_img_data(img),self._get_cur_path('low.png'))
        
        transformed = freq.dct_img(img)
        band_coff = freq.partition_coff(transformed)
        nums = band_coff.shape[0]
        band_data = np.zeros((nums, *img.shape))
        for i in range(band_coff.shape[0]):
            band_data[i] = freq.idct(band_coff[i])
        
        # subband_data: img(low), img, dct
        # subband_coff: dct(low), dct, dct
            
        return band_data, band_coff
    
    def testcase(self,img):
        
        coff = freq.dct_img(img)
        r_img = freq.idct_img(coff)

        img = io.cvt_np_tensor(img)
        ##### @test 用dct 测试了几乎无损 
        # r_img = cvt_np_tensor(r_img)
        # psnr = self._eval_performance(img,r_img,"psnr")
        # print('psnr: ', psnr)
        # error = self._eval_performance(img,r_img,"mse")
        # print('error: ', error)

        ### band
        band_coff = freq.partition_coff(coff)
        r_coff = freq.reconstruct_coff(band_coff)
        r_img = freq.idct_img(r_coff)
        
        r_img = io.cvt_np_tensor(r_img)

        psnr = self._eval_performance(img,r_img,"psnr")
        print('psnr: ', psnr)
        error = self._eval_performance(img,r_img,"mse")
        print('error: ', error)
        exit()


    def visualize_dct(self, dct_img, mask_freq=4, line_block=8):
        
        # mask掉左上角能量极大的部分
        mask = self.proc.gen_filter_mask(0, mask_freq)
        dct_img = self.proc.reconstruct_dct_coff(np.zeros_like(dct_img), dct_img, mask)
        h,w,c = dct_img.shape

        # 先取单通道
        dct_img = dct_img[:,:,0]
        plt.figure(figsize=(20, 20)) 
        plt.imshow(dct_img, cmap='hot', interpolation='nearest')
        plt.colorbar()

        # 网格
        for i in range(0, w+1, line_block):  
            plt.axhline(i-0.5, color='blue', linewidth=0.5)  
        for i in range(0, h+1, line_block):
            plt.axvline(i-0.5, color='blue', linewidth=0.5)  


        plt.title('Visualization of DCT Coefficients')
        plt.savefig(self._get_cur_path("dct_visual.png"), dpi=300) 
        plt.close()

    def visualize_block_dct(self, dct_img):
        block_h = int(np.sqrt(dct_img.shape[1])) # 48， 4096，3
        dct_img = dct_img.reshape(dct_img.shape[0], block_h, block_h, -1)
        
        # 先取一个通道做可视化
        dct_img = dct_img[:,:,:,0]
        
        fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
        idx = 0
        for i, ax in enumerate(axes.flat):
            # 暂时先这么处理
            ax.axis('off')
            row = i // 8
            col = i % 8
            if row < 4 and col < 4: continue # mask掉左上角16
            im = ax.imshow(dct_img[idx], cmap='hot', interpolation='nearest')
            idx += 1
            

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
        
        plt.title('Visualization of Permuted DCT Coefficients')
        plt.savefig(self._get_cur_path("dct_visual_permuted.png"), dpi=300) 
        plt.close()

    def permute_dct(self, dct_img, mask):
        # mask = self.proc.gen_filter_mask(low_mask_freq, high_mask_freq)
        mask_size = np.sum(mask)
        
        block_size = 8 * 8 
        h,w,c = self.dct_img.shape
        p_block_size = block_size - mask_size # 64 - 16 = 48
        block_num = int((h * w) / block_size) # 64*64
        permute_dct = np.zeros((p_block_size, block_num, c))    
        
        ### todo: 改成一个循环即可 现在太慢
        ### 注: 以下都是行优先
        block_idx = 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                cur_block = dct_img[i:i+8, j:j+8]
                ele_idx = 0
                for k in range(0, 8):
                    for l in range(0, 8):
                        # skip mask
                        if not mask[k][l]: 
                            permute_dct[ele_idx][block_idx] = cur_block[k][l]
                            ele_idx += 1
                block_idx += 1
        return permute_dct

    def de_permute_dct(self, permuted_dct, low_dct, mask_freq=4):
        # permuted_dct (48, 64*64,3)
        # low_dct (4*4, 64*64, 3)
        assert low_dct.shape[0] == mask_freq * mask_freq
        total_ = (permuted_dct.shape[0]+low_dct.shape[0]) * permuted_dct.shape[1] # 64 * 64 * 64
        h = w = int(np.sqrt(total_)) # 512
        c = permuted_dct.shape[2]
        r_dct = np.zeros((h, w, c))

        mask = self.proc.gen_filter_mask(0, mask_freq)
        
        block_idx = 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                cur_block = r_dct[i:i+8, j:j+8]
                l_ele_idx = 0 # low_elements
                p_ele_idx = 0 # permuted_elements
                for k in range(0, 8):
                    for l in range(0, 8):
                        # 低频部分
                        if mask[k][l]:
                            cur_block[k][l] = low_dct[l_ele_idx][block_idx]
                            l_ele_idx += 1
                        # permuted部分
                        else:
                            cur_block[k][l] = permuted_dct[p_ele_idx][block_idx]
                            p_ele_idx += 1
                block_idx += 1
                
        return r_dct

    def set_block_coff_idx_matrix(self):
        matrix = torch.zeros(8, 8, dtype=torch.int8)
        matrix[:4, :4] = -1
        assign_index = 0
        for i in range(8):
            for j in range(8):
                if matrix[i, j] != -1:
                    matrix[i, j] = assign_index
                    assign_index += 1
        self._block_coff_idx_matrix = matrix
        return matrix
        
    def get_row_col_from_masking_block_by_idx(self, idx):
        # 最好能直接写成函数 要不然还是慢
        matrix = self._block_coff_idx_matrix
        x, y =  (matrix == idx).nonzero()[0]
        return x, y 


    def train_coff_net(self, epoch, base_ratio=0.6):
        '''改成完全交替训练'''
        if not self.args.use_coff_net: 
            return False
        
        base_num = 100
        return bool((epoch % base_num) >= int(base_ratio * base_num))


    def drp_train_coff_net(self, epochs, cross_ratio = 0.001, base_ratio = 0.6):
        '''选择器,在指定epoch下是否train_coff_net 否则就是base net'''
        '''后续改成纯交叉的'''
        if not self.args.use_coff_net: 
            return False
        total_epochs = self.args.num_epochs
        cross_epochs = int(cross_ratio * total_epochs)
        base_epochs = (total_epochs - cross_epochs) * base_ratio + cross_epochs

        # 一开始交替训  → bugfree | 后面改逻辑(这个就是兜底逻辑)
        if epochs < cross_epochs:
            return bool(epochs % 2)
        # 后面直接按比例 先训低频 → 后高频系数
        else:
            return bool(epochs > base_epochs)
        
        ##### 
        