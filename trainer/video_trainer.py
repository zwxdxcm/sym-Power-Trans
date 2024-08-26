import torch
import os

from trainer.img_trainer import ImageTrainer
import numpy as np
from util.logger import log
from util.tensorboard import writer
from util import io
from scipy.optimize import minimize
import imageio.v2 as imageio

from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

#### (x,y,t) → (r,g,b) | 输入帧的目录 
class VideoTrainer(ImageTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def _set_status(self):
        self.recover_norm_map = {}
        self.norm_method = self.args.normalization  
        # 之后用
        self.traget_bound_min = -1
        self.traget_bound_max = 1
    
    def _parse_input_data(self):
        # parse input frames
        dir_path = self.args.input_path # 目录
        entries = os.listdir(dir_path)
        files = [entry for entry in entries if os.path.isfile(os.path.join(dir_path, entry))]
        files = sorted(files, key= lambda x: int(x.split(".")[0].split("_")[-1][-2:]))  # 更好的写法  
        input_frame_list = []
        for file in files:
            cur_path = os.path.join(dir_path, file)
            img = torch.from_numpy(imageio.imread(cur_path)).permute(2, 0, 1).to(torch.float32)  # c,h,w
            input_frame_list.append(img)
        
        self.input_frames = torch.stack(input_frame_list) # t,c,h,w
        self.T, self.C, self.H, self.W = self.input_frames.shape
        


    def _get_data(self, mark="default", use_normalization=True):
        # get coords
        data = self.input_frames
        frame_coords = torch.stack(
            torch.meshgrid(
                [torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)],
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        x_y_coords = frame_coords.repeat(self.T, 1,1)

        t_coords = torch.linspace(-1, 1,self.T).unsqueeze(1).unsqueeze(2).repeat(1, self.H*self.W, 1)
        coords = torch.cat((x_y_coords, t_coords), dim=-1)
        
        if use_normalization:
            data = self._normalize_data(data, mark)
        
        gt = data.permute(0, 2, 3, 1).reshape(self.T, -1, self.C)  # t, hw, c

        # @test recons
        # r_frames = self.reconstruct_frames(gt)
        # psnr_ = self.compute_frames_psnr(r_frames, self.input_frames)
        # print('psnr_: ', psnr_)
        
        self.raw_coords = coords 
        self.raw_gt = gt

        return coords, gt
    
    def _get_data_loader(self, coords, gt):
        self.bsz = int(self.H * self.W / 1) # 1080 * 540 / 3 → 减小缓存

        # x = coords
        # y = gt

        ## shuffle all pixels
        shuffled_indices = torch.randperm(self.T * self.H*self.W)
        x = coords.reshape(-1, coords.shape[-1]).index_select(0,shuffled_indices).reshape(-1, self.bsz, coords.shape[-1])
        y = gt.reshape(-1, gt.shape[-1]).index_select(0,shuffled_indices).reshape(-1, self.bsz,  gt.shape[-1])


        # shuffle all pixels
        # x = coords.reshape(-1, coords.shape[-1])
        # y = gt.reshape(-1, gt.shape[-1])

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=16, shuffle=True, pin_memory=True)
        
        # for batch_idx, (input_batch, output_batch) in enumerate(dataloader):
        #     print(f"Batch {batch_idx + 1}")
        #     print("Input batch shape:", input_batch.shape)
        #     print("Output batch shape:", output_batch.shape)

        return dataloader
        

    
    def _normalize_data(self, data: torch.tensor, mark="default"):
        if self.norm_method == "min_max":
            # min-max
            _min = torch.min(data)
            _max = torch.max(data)
            self.recover_norm_map[mark] = (_min, _max)
            normed = (data - _min) / (_max - _min)
            normed_data = self.encode_zero_mean(normed)

        elif self.norm_method == "instance":
            # data = data / 255
            _mean = data.mean()
            _std = data.std()
            data = (data - _mean) / (_std + 1e-5)
            log.inst.info(f"max value after instance normalization: {data.max()}")
            log.inst.info(f"min value after instance normalization: {data.min()}")
            self.recover_norm_map[mark] = (_mean, _std)
            normed_data = data
        
        elif self.norm_method == "power":
            _meta, normed_data = self._power_normalization(data)
            self.recover_norm_map[mark] = _meta
        
        normed_data = normed_data * self.args.amp
        return normed_data
    
    def _denormalize_data(self, normalized_data: torch.tensor, mark="default"):

        # 测试振幅
        normalized_data = normalized_data / self.args.amp

        if self.norm_method == "min_max":
            normalized_data = self.decode_zero_mean(normalized_data)
            _min, _max = self.recover_norm_map[mark]
            r_data = normalized_data * (_max - _min) + _min

        elif self.norm_method == "instance":
            _mean, _std = self.recover_norm_map[mark]
            r_data = normalized_data * (_std + 1e-5) + _mean
            # r_data = r_data * 255

        elif self.norm_method == "channel":
            norm_mark_list = self.recover_norm_map[mark]
            c = normalized_data.shape[0]
            r_data = torch.zeros_like(normalized_data)
            for i in range(c):
                _mean, _std = norm_mark_list[i]
                r_data[i] = normalized_data[i] * (_std + 1e-5) + _mean

        elif self.norm_method == "power":
            _meta = self.recover_norm_map[mark]
            r_data = self._inverse_power_normalization(
                _meta, normalized_data
            )
        
        elif self.norm_method == "z_power":
            _meta = self.recover_norm_map[mark]
            # → [0, 255] → [-1,1] → inverse_power
            _min, _max, gamma, _left_shift_len, _shift_len, _mean, _std = _meta 
            r_data = normalized_data * (_std + 1e-5) + _mean
            r_data = (r_data - (_min - _left_shift_len)) / (_max - _min + _shift_len)
            r_data = self.encode_zero_mean(r_data)
            r_data = self._inverse_power_normalization(
                _meta[:-2], r_data
            )            

        elif self.norm_method == "box_cox":
            _min, _max, _lambda = self.recover_norm_map[mark]
            r_data = self._inverse_box_cox_normalization(
                _min, _max, _lambda, normalized_data
            )

        elif self.norm_method == "power_channel":
            norm_mark_list = self.recover_norm_map[mark]
            c = normalized_data.shape[0]
            r_data = torch.zeros_like(normalized_data)
            for i in range(c):
                _meta = norm_mark_list[i]
                cur_r_data = self._inverse_power_normalization(
                    _meta, normalized_data[i]
                )
                r_data[i] = cur_r_data

        else:
            raise NotImplementedError

        return r_data
    
    
    def _get_gamma_by_half(self, data):
        hist = torch.histc(data.flatten(), bins=256, min=data.min(), max=data.max())
        pdf = hist / hist.sum()
        cdf = torch.cumsum(pdf, dim=0)
        half_index = torch.searchsorted(cdf, self.args.pn_cum)  # 0.5
        half_perc = half_index / 256
        gamma = np.log(0.5) / np.log(half_perc)
        return gamma
    
    def _get_gamma_by_edge_claibration(self, data):
        half_gamma = self._get_gamma_by_half(data)
        # 反向校准
        hist = torch.histc(data.flatten(), bins=256, min=data.min(), max=data.max())
        pdf = hist / hist.sum()

        min_max_normed = (data - data.min()) / (data.max() - data.min())
        half_gamma_normed = torch.pow(min_max_normed, half_gamma)
        half_gamma_hist = torch.histc(half_gamma_normed.flatten(), bins=256, min=half_gamma_normed.min(), max=half_gamma_normed.max())
        half_gamma_pdf = half_gamma_hist / half_gamma_hist.sum()

        _beta_len = int(self.args.pn_beta * 256)
        _minmax_bin = pdf[:_beta_len] if half_gamma> 1 else pdf[-_beta_len:]
        _half_gamma_bin = half_gamma_pdf[:_beta_len] if half_gamma> 1 else half_gamma_pdf[-_beta_len:]
        
        delta_sum = _half_gamma_bin.sum() - _minmax_bin.sum() # assert > 0
        delta_sum /= self.args.pn_beta
        if half_gamma < 1: 
            delta_gamma = 1 / half_gamma - 1
        else: 
            delta_gamma = half_gamma - 1
        
        new_delta_gamma = delta_gamma * min(delta_sum,1)
        
        if half_gamma < 1:
            new_gamma = 1/ ( 1 / half_gamma - new_delta_gamma)
        else:
            new_gamma = half_gamma - new_delta_gamma
        
        return new_gamma


    def _power_normalization(self, data: torch.tensor):
        ## min-max → gamma → zero-mean
        _min = torch.min(data)
        _max = torch.max(data)

        hist = torch.histc(data.flatten(), bins=256, min=data.min(), max=data.max())
        pdf = hist / hist.sum()

        if self.args.pn_beta <= 0:
            gamma = self._get_gamma_by_half(data)
        else:
            gamma = self._get_gamma_by_edge_claibration(data)
        
        #### set min and max
        boundary = self.args.gamma_boundary
        if gamma > 1: gamma = min(boundary, gamma)
        else: gamma = max(1/boundary, gamma)
        
        log.inst.info(f"gammma _{gamma}")

        # buffer pn_buffer=0 相当于不生效 | 工程设计
        if self.args.pn_buffer < 0:
            _alpha_len = int(self.args.pn_alpha * 256)
            left_alpha_sum = pdf[:_alpha_len].sum()
            right_alpha_sum = pdf[-_alpha_len:].sum()
            _left_shift_len = self.args.pn_k * left_alpha_sum
            _right_shift_len = self.args.pn_k * right_alpha_sum
            log.inst.info(f"_left_shift_len : {_left_shift_len}")
            log.inst.info(f"_right_shift_len : {_right_shift_len}")

            ## 对称 & 切割 实验
            # _max_shift =  max(_left_shift_len, _right_shift_len)
            # if _max_shift < 0.1: _max_shift = 0
            # _left_shift_len = _right_shift_len = _max_shift

        else: 
            _left_shift_len = (_max - _min) * self.args.pn_buffer
            _right_shift_len = (_max - _min) * self.args.pn_buffer
        
        _shift_len = _left_shift_len + _right_shift_len
        
        data = (data - (_min - _left_shift_len)) / (_max - _min + _shift_len)  # [0,1]
        data = torch.pow(data, gamma)
        normed_data = self.encode_zero_mean(data)
        return [_min, _max, gamma, _left_shift_len, _shift_len], normed_data

    def _inverse_power_normalization(
        self, _meta, normalized_data: torch.tensor
    ):  
        _min, _max, gamma, _left_shift_len, _shift_len = _meta
        ## zero-mean → gamma →  min-max
        normalized_data = self.decode_zero_mean(normalized_data)
        # clip to (0,1)
        normalized_data = torch.clamp(normalized_data, min=0.0, max=1.0)

        normalized_data = torch.pow(normalized_data, 1.0 / gamma)
        r_data = normalized_data * (_max - _min + _shift_len) + (_min - _left_shift_len)
        return r_data
        

    def reconstruct_frames(self, pred, cuda=False) -> torch.tensor:  # c,h,w
           r_frames = pred.reshape(self.T, self.H, self.W, self.C).permute(0,3,1,2)
           r_frames = self._denormalize_data(r_frames)
           r_frames = torch.clamp(r_frames, min=0, max=255)
           return r_frames

    def compute_frames_psnr(self, pred, gt):
        psnr_list = []
        for i in range(self.T):
            cur_psnr = self.compute_psnr(pred[i], gt[i])
            psnr_list.append(cur_psnr)
        log.inst.info(f'psnr_list: {psnr_list}')
        return sum(psnr_list) / self.T

    def compute_frames_ssim(self, pred, gt):
        ssim_list = []
        for i in range(self.T):
            cur_psnr = self.compute_ssim(pred[i], gt[i])
            ssim_list.append(cur_psnr)
        log.inst.info(f'ssim_list: {ssim_list}')
        return sum(ssim_list) / self.T

    def train(self):
        # log.start_timer("train")
        num_epochs = self.args.num_epochs
        coords, gt = self._get_data()
        dataloader = self._get_data_loader(coords, gt)
        
        self.model = self._get_model(in_features=3, out_features=3).to(self.device)
        optimizer = torch.optim.Adam(lr=self.args.lr, params=self.model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1)
        )

        for epoch in trange(1, num_epochs + 1):
            
            for batch_idx, (input_batch, output_batch) in enumerate(dataloader):
                
                coords = input_batch.squeeze().to(self.device) 
                gt = output_batch.squeeze().to(self.device)
                pred = self.model(coords)
                loss = self.compute_mse(pred, gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
                # log.pause_timer("train")
                writer.inst.add_scalar("train/loss", loss.detach().item(), global_step=epoch)
    
            ### inference
            if epoch % self.args.snap_epoch == 0: 
                self.inference(epoch)
        
        self.inference(epoch)
        self._save_ckpt(epoch, self.model, optimizer, scheduler)
        log.inst.info(f"####### end training #######")
    
    def inference(self, epoch=0):
        with torch.no_grad():
            pred_frames = []
            for i in range(self.T):
                cur_coords = self.raw_coords[i].to(self.device)
                # cur_gt = self.raw_gt[i].to(self.device)
                cur_pred = self.model(cur_coords)
                pred_frames.append(cur_pred)
            pred_frames = torch.stack(pred_frames)
            r_frames = self.reconstruct_frames(pred_frames.cpu()) 
            av_psnr = self.compute_frames_psnr(r_frames, self.input_frames)
            log.inst.info(f"av_psnr_{epoch} : {av_psnr}")
            av_ssim = self.compute_frames_ssim(r_frames, self.input_frames)     
            log.inst.info(f"av_ssim_{epoch} : {av_ssim}")
            
            frame_folder = self._get_cur_path(f"{epoch}_pred_frames")
            os.makedirs(frame_folder, exist_ok=True)
            for i in range(self.T):
                cur_frame = r_frames[i].permute(1,2,0) #h,w,c

                io.save_cv2(
                    cur_frame.numpy(),
                    os.path.join(frame_folder, f"final_pred_frame_{i}.png")
                )




