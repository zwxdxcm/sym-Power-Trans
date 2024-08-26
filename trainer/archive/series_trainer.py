from models import Siren,PEMLP
from util.logger import log
from util.tensorboard import writer
from util.misc import gen_cur_time
from trainer.base import BaseTrainer

import numpy as np
import imageio.v2 as imageio
import torch
import os
from tqdm import trange


## 先写出来 后面再做抽象环和模块化
class SeriesTrainer(BaseTrainer):
    def __init__(self, args, mode="train"):
        super().__init__(args,mode)

    # overwrite
    def _get_data(self):

        gt = self.gen_random_signal().unsqueeze(1)
        size = gt.shape[0]
        coords = torch.arange(size, dtype=torch.int).unsqueeze(1)
        coords = self.encode_zero_mean(self.normalize(coords))

        if self.args.zero_mean:
            gt = self.encode_zero_mean(gt)

        # todo: map → [-1,1]
        return coords, gt, size
    
    # overwrite
    def train(self):
        coords, gt, size = self._get_data()
    
        # transfer2cuda
        self.model = self._get_model(in_features=1, out_features=1).to(self.device)
        coords = coords.to(self.device)
        gt = gt.to(self.device)

        num_epochs = self.args.num_epochs
        

        optimizer = torch.optim.Adam(lr=self.args.lr, params=self.model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1))
        
        for epoch in trange(1, num_epochs + 1): 

            log.start_timer("train")
            
            # inference 
            pred = self.model(coords)
            loss = self._compute_loss(pred, gt)
            psnr = self._eval_performance(pred, gt, "psnr")

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize() # 确保测量时间
            log.pause_timer("train")

            # record | writer
            if epoch % self.args.log_epoch == 0:
                log.inst.info(f"epoch: {epoch} → PSNR {psnr:.4f}")
            
            writer.inst.add_scalar("train/loss", loss.detach().item(), global_step=epoch)
            writer.inst.add_scalar("train/psnr", psnr.detach().item(), global_step=epoch) # psnr 衡量时序数据这个值仅仅维持在10 (因为我信号全都归一化了)
        # save ckpt
        self._save_ckpt(epoch,optimizer,scheduler)
    
    @staticmethod
    def gen_random_signal(num=300000):
        signal = torch.rand(num, dtype=torch.float32)
        return signal
        
    
