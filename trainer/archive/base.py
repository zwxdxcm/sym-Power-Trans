from models import Siren, PEMLP, Finer, Wire, Gauss
from util.logger import log
from util.tensorboard import writer
from util.misc import gen_cur_time
from util.io import cvt_all_tensor

import numpy as np
import imageio.v2 as imageio
import torch
import os
from tqdm import trange

## 先写出来 后面再做抽象环和模块化
class BaseTrainer(object):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.args = args
        self.device = torch.device('cuda:0') # 此处外部用cuda_visible控制


    def _get_model(self, in_features, out_features):

        model_type = self.args.model_type

        if model_type == "siren":
            model = Siren(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        first_omega_0=self.args.first_omega, hidden_omega_0=self.args.hidden_omega)
        elif model_type == 'pemlp':
            model = PEMLP(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        N_freqs=self.args.N_freqs)
        elif model_type == 'finer':
            model = Finer(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        first_omega_0=self.args.first_omega, hidden_omega_0=self.args.hidden_omega, 
                        first_bias_scale=self.args.first_bias_scale, scale_req_grad=self.args.scale_req_grad) # specific for FINER
        elif model_type == 'wire':
            model = Wire(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        first_omega_0=self.args.first_omega, hidden_omega_0=self.args.hidden_omega, scale=self.args.scale)
        elif model_type == 'gauss':
            model = Gauss(in_features=in_features, out_features=out_features, hidden_layers=self.args.hidden_layers, hidden_features=self.args.hidden_features,
                        scale=self.args.scale)
        else:
            raise NotImplementedError
        return model
        
    def _get_data(self):
        return NotImplementedError
        
    def _compute_loss(self, pred, gt):
        return self.compute_mse(pred, gt)
    
    def _eval_performance(self, pred, gt, mode="mse"):
        
        pred = cvt_all_tensor(pred)
        gt = cvt_all_tensor(gt)

        if mode == "mse":
            return self.compute_mse(pred, gt)
        elif mode == "psnr":
            if self.args.zero_mean:
                pred = self.decode_zero_mean(pred)
                gt = self.decode_zero_mean(gt)
            return self.compute_psnr(pred, gt)
        else:
            raise NotImplementedError

    def train(self):
        raise NotImplementedError
        
        
    def _save_ckpt(self,epoch,optimizer,scheduler):
        state_dict = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            # 'best_score': 0,
            # 'final_score': 0,
            'args': self.args
        }
        cur_time = gen_cur_time()
        saved_name =  f"ckpt_{epoch}_{cur_time}.pth"
        torch.save(state_dict, os.path.join(self.args.save_folder, saved_name))
        log.inst.success(f"Save ckpt in epoch {epoch} named {saved_name}")

    def _get_cur_path(self, filename):
        return os.path.join(self.args.save_folder, filename)

    @staticmethod
    def encode_zero_mean(data):
        return (data - 0.5) / 0.5

    @staticmethod
    def decode_zero_mean(data):
        return data/2 + 0.5
    
    @staticmethod
    def compute_mae(pred, gt):
        return torch.abs((pred - gt)).mean()

    @staticmethod
    def compute_mse(pred, gt):
        return torch.mean((pred - gt) ** 2)
    
    @staticmethod
    def compute_psnr(pred, gt):
        '''仅针对normalize 0-1'''
        mse = BaseTrainer.compute_mse(pred, gt)
        return -10. * torch.log10(mse)
    
    @staticmethod
    def compute_raw_psnr(pred: np.ndarray, gt:np.ndarray):
        '''原始定义'''
        mse = np.mean((pred - gt)**2)
        return 20 * np.log10(gt.max() / np.sqrt(mse))
    
    @staticmethod
    def normalize(data): 
        return (data - data.min()) / (data.max() - data.min())