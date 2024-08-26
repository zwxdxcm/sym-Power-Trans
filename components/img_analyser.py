import imageio.v2 as imageio
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize
import cv2
from scipy import stats

from loguru import logger



class ImgAnalyser(object):
    def __init__(self, input_path, log_path):
        self.input_path = input_path
        self.log_path = log_path
        self.result = {} # for print
        self._read_img()
        self._normlized_img()
        self._log()

        self._beta = 0.05 # 用于统计惩罚项

    def _read_img(self):
        self.input_img = torch.from_numpy(imageio.imread(self.input_path)).to(
            torch.float32
        ).permute(2,0,1)  # c,h,w
        self.C, self.H, self.W = self.input_img.shape
    
    def _log(self):
        logger.info(f"min-max for {self.input_path}")
        self.log_data(self.min_max_img)
        
        logger.info(f"power for {self.input_path}")
        self.log_data(self.power_img)


    def _normlized_img(self):
        # -1 → 1
        self._min = torch.min(self.input_img)
        self._max = torch.max(self.input_img)
        self._mean = torch.mean(self.input_img)
        self._std = torch.std(self.input_img)

        self.min_max_img = (self.input_img - self._min) / (self._max - self._min)
        self.min_max_img = (self.min_max_img - 0.5) / 0.5

        self.bn_img = (self.input_img - self._mean) / (self._std + 1e-5)

        self.power_img = self.power_normalization_enhance(self.input_img)
        self.power_mle_img = self.power_mle_normalization(self.input_img)
        self.box_img = self.box_cox_normalization(self.input_img)

        # print('self.gamma: ', self.gamma)
    
    def profile_penalty(self,data, key):
        hist = torch.histc(data.flatten(), bins=256, min=data.min(), max=data.max())
        pdf = hist / hist.sum()
        # 主要：要有self.gamma
        _len = int(256 * self._beta)
        bin = pdf[:_len] if self.gamma > 1 else pdf[-_len:]
        # 统计
        self.result[f"{key}_sum"] = bin.sum()
        self.result[f"{key}_mean"] = bin.mean()
        # self.result[f"{key}_var"] = bin.var()
        self.result[f"{key}_var"] = (bin/0.05).var()
    
    def get_penalty_gamma(self):
        delta_sum = (self.result["power_sum"] - self.result["minmax_sum"]) / self._beta
        gamma = self.gamma
        # delta_gamma = gamma - 1
        if gamma < 1: 
            delta_gamma = 1 / gamma - 1
        else: 
            delta_gamma = gamma - 1

        new_delta_gamma = delta_gamma * min(delta_sum,1)
    
        if gamma < 1:
            new_gamma = 1/ (1 /gamma - new_delta_gamma)
        else:
            new_gamma = gamma - new_delta_gamma
            
        self.result["new_gamma"] = new_gamma
        return new_gamma


    def _get_path(self, file_name):
        return os.path.join(self.log_path, file_name)
    
    def opencv_hist(self, data):
        ## 调用API无法reverse回来, 最后不行再用这种办吧 | 就算自己手写也需要一个255的表做映射
        assert data.dim() == 2  # 单通道
        hist = torch.histc(data, bins=256, min=0, max=255)
        cdf = hist.cumsum(0)
        cdf = cdf / cdf.max() # → [0,1]
        
        equ = cv2.equalizeHist(data.numpy().astype(np.uint8))
        # todo: reverse 看精度
        return torch.tensor(equ).to(torch.float32)
    
    def power_normalization(self, data):
        hist = torch.histc(data.flatten(), bins=256, min=data.min(), max=data.max())
        pdf = hist / hist.sum()
        cdf = torch.cumsum(pdf, dim=0)
        half_index = torch.searchsorted(cdf, 0.5)
        half_perc = half_index / 256 # 0.128 for 11.png
        gamma = np.log(0.5) / np.log(half_perc) # log_halfperc^0.5
        self.gamma = gamma

        # gamma
        ## 在这里做preprocess
        img = data / 255.
        img = torch.pow(img,  self.gamma)
        img = img * 255.

        # min-max → (-1,1)
        img = (img - img.min()) / (img.max() - img.min()) # [0,1]
        img =  (img - 0.5) / 0.5
        return img
    
    def power_mle_normalization(self, data):
        # → (-1,1)
        data = data.flatten()
        _std = data.std()
        data = (data - data.min()) / (data.max()-data.min()) # (0,1)

        data = data.numpy()
        
        # hist = torch.histc(data.flatten(), bins=256, min=data.min(), max=data.max())
        # pdf = hist / hist.sum()
        
        # cdf = torch.cumsum(pdf, dim=0)

        def likelihood(gamma, data):
            mean = 0.5
            std = 0.25 # _std
            data = np.power(data,  gamma)
            log_likelihood = np.sum(norm.logpdf(data, loc=mean, scale=std))
            return -log_likelihood

        # mle with norm dist
        result = minimize(likelihood, 1.0 , args=(data,), bounds=[(0, 10)])
        gamma = result.x[0]
        self.result["mle_gamma"] = gamma
        print('gamma: ', gamma)

        data = np.power(data,  gamma)
        data = data - 0.5 / 0.5
        return torch.tensor(data)

    def power_normalization_enhance(self, data):
        hist = torch.histc(data.flatten(), bins=256, min=data.min(), max=data.max())
        pdf = hist / hist.sum()
        cdf = torch.cumsum(pdf, dim=0)
        half_index = torch.searchsorted(cdf, 0.5)
        half_perc = half_index / 256 # 0.128 for 11.png
        gamma = np.log(0.5) / np.log(half_perc) # log_halfperc^0.5
        
        self.gamma = gamma
        self.result["half_gamma"] = gamma
        
        # gamma
        img = data
        _len = img.max() - img.min() # 通常为255
        _left_shift = _right_shift =  0.01 * 255
        
        _left_shift = _right_shift = 0
        
        # 自适应计算shift_left & shift_right
        _alpha = 0.01
        _alpha_len = int(_alpha * 256)
        #_alpha_len = 1
        left_alpha_sum = pdf[:_alpha_len].sum()
        right_alpha_sum = pdf[-_alpha_len:].sum()
        _k = 1.0 # to 0.01
        _left_shift = _k * left_alpha_sum
        print('_left_shift: ', _left_shift)
        _right_shift = _k * right_alpha_sum
        print('_right_shift: ', _right_shift)

        _min = img.min() - _left_shift * _len
        _max = img.max() + _right_shift * _len

        img = (img - _min) / (_max - _min) # [0,1]
        img = torch.pow(img,  self.gamma)
        # min-max → (-1,1)
        #  decode
        img =  (img - 0.5) / 0.5
        print('img.mean(): ', img.mean())
        return img
    

    def box_cox_normalization(self, data):
        ## box-cox → min-max → zero mean
        data = data.flatten()
        data = data + 0.1 * 256
        transformed_data, lambda_ = stats.boxcox(data.flatten())
        self._lambda_ = lambda_
        # print('self._lambda_ : ', self._lambda_ )
        img = transformed_data
        img = (img - img.min()) / (img.max() - img.min())
        img =  (img - 0.5) / 0.5
        return torch.tensor(img)
    
    # @test
    def get_pdf(self, data):
        hist = torch.histc(data.flatten(), bins=256, min=data.min(), max=data.max())
        pdf = hist / hist.sum()
        cdf = torch.cumsum(pdf, dim=0)
        half_index = torch.searchsorted(cdf, 0.5)
        half_perc = half_index / 256 # 0.128 for 11.png
        gamma = np.log(0.5) / np.log(half_perc) # log_halfperc^0.5
        print('gamma: ', gamma)


        # self.visual_histc(pdf, "pdf")
    
    def log_data(self, data):
        data = data.flatten()
        _mean = torch.mean(data)
        _std = torch.std(data)
        _skewness = skew(data)
        _kurt = kurtosis(data)
        logger.info(f"_mean : {_mean}")
        logger.info(f"_std : {_std}")
        logger.info(f"_skewness : {_skewness}")
        logger.info(f"_kurt : {_kurt}")


    def visual_histc(self, data, save_name="hist"):
        # assert data.dim() == 2  # 单通道
        flatten = data.flatten()
        histogram = torch.histc(flatten, bins=100)

        x = torch.linspace(flatten.min(), flatten.max(), steps=histogram.shape[0])
        plt.figure()
        plt.title("Pixel Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        # plt.xlim([-1,1])
        plt.plot(x, histogram)

        plt.savefig(self._get_path(f"{save_name}.png"), dpi=300)
        plt.close()

