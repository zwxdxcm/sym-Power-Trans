from models import Siren, PEMLP
from util.logger import log
from util.tensorboard import writer
from util.misc import gen_cur_time
from util import io
from trainer.img_trainer import ImageTrainer
from components.dct_tool import DCTTool
from components.ssim import compute_ssim_loss
from components.laplacian import compute_laplacian_loss
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import imageio.v2 as imageio
import torch
import os
import cv2

from tqdm import trange

import matplotlib.pyplot as plt
from enum import Enum
from collections import defaultdict

from models import BranchSiren
from scipy import stats
from scipy.stats import norm, skew, kurtosis,skewnorm

from scipy.optimize import minimize


class Mode(Enum):
    pure_coff = 0
    permute_coff = 1  # for dev
    dual_coff = 2
    polar_fft_coff = 3  # fft + polar
    wavelet_subband = 4  # wavelet subband 4
    wavelet_subband_multinet = 5  # 4 net
    spatial_split = 6
    pure_spatial = 7


class DevImageTrainer(ImageTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.tool = DCTTool(block_size=self.block_size)
        
        self._pre_study_hist() # 预实验 测试sknewness & variance的
        self._transform_data()

    def _transform_data(self):
        ## 提前transform data
        if self.args.inverse:
            self.input_img = 255 - self.input_img 
        if self.args.rpp:
            pixels = self.input_img.reshape(self.C, -1)
            indices = torch.randperm(pixels.size(1))
            permuted_pixels = pixels[:, indices]
            permuted_img = permuted_pixels.reshape(self.C, self.H, self.W)
            self.input_img = permuted_img

    def _pre_study_hist(self):
        if not self.args.pre_study: return 
        self.input_img = self.input_img.to(torch.float32)

        self._plot_hist(self.input_img, saved_tag="raw_hist")
        # self._plot_hist(eq_img, saved_tag="eq_hist")
        
        
        # 真实数据 → hist 均衡图 sel
        # img_ = io.cvt_torch_cv2(self.input_img)
        # img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        # eq_img = cv2.equalizeHist(img_)
        

        ### 模拟数据 → 3*h*w 采样训练 | 结论：不能用模拟的去证明 因为会有频率做干扰
        # mock_img = torch.zeros_like(self.input_img)
        mean_ = 255
        std_ = 127.5 / 3  # 3倍标准差→99.7%
        a_ = self.args.skew_a  # 偏度系数
        # -3,-2,-1,0,1,2,3
        size = (3, 256, 256)
        mean1 = 255 * (1/4)
        mean2 = 255  - mean1
        weight1 = self.args.skew_w1
        mock_img = norm.rvs(loc = mean_ * 0.5, scale=std_, size=size)
        # mock_img = norm.rvs(loc = mean_ * 0.5, scale=127.5 / weight1, size=size)
        # mock_img = skewnorm.rvs(a_, loc=mean_, scale=std_,size=size)
        # mock_img = self._gen_bimodal_dist(mean1, mean2, std_, std_, weight1,size)
        mock_img = torch.tensor(mock_img).to(torch.float32)
        mock_img = torch.clamp(mock_img, min=0.0, max=255.0)

        self._plot_hist(mock_img, saved_tag="mock_hist")
        io.save_cv2(
                mock_img.permute(1,2,0).numpy(),
                self._get_cur_path(f"mock_img.png"),
            )

        self.input_img = mock_img
        # exit()
    
    def _gen_bimodal_dist(self, mean1, mean2, std1, std2, weight1, shape):
        total_size = np.prod(shape)
        size1 = int(total_size * weight1) + 1
        weight2 = 1 - weight1
        size2 = int(total_size * weight2) + 1
        samples1 = norm.rvs(loc=mean1, scale=std1, size=size1)
        samples2 = norm.rvs(loc=mean2, scale=std2, size=size2)
        samples = np.concatenate([samples1, samples2])
        np.random.shuffle(samples)
        samples = samples[:total_size]
        samples = samples.reshape(shape)
        return samples



    def _plot_hist(self, data, saved_tag='hist'):
        # assert data.dim() == 2  # 单通道
        if type(data) == np.ndarray:
            data = io.cvt_cv2_torch(data)
            data = data.to(torch.float32)

        flatten = data.flatten()
        histogram = torch.histc(flatten, bins=100)

        _mean = torch.mean(data)
        _std = torch.std(data)
        _skewness = skew(flatten)
        log.inst.info(f"_mean : {_mean}")
        log.inst.info(f"_std : {_std}")
        log.inst.info(f"_skewness : {_skewness}")
        

        x = torch.linspace(flatten.min(), flatten.max(), steps=histogram.shape[0])
        plt.figure()
        plt.title("Pixel Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.xlim(0,255)
        plt.plot(x, histogram)

        plt.savefig(self._get_cur_path(f"{saved_tag}_{self.data_name}.png"), dpi=300)
        plt.close()

    def _set_status(self):
        super()._set_status()
        self.mode = self.args.coff_mode
        self.block_size = self.args.dct_block_size
        self.recover_norm_map = {}  # dict: tuple(_min, _max)
        self.wavelet = "haar"
        self.use_branch_train = False
        self.norm_method = self.args.normalization  # min_max, instance, channel
        self.data_name = os.path.basename(self.args.input_path)
        # self.gamma = self.args.gamma
        self.result = {}
    
    def _profile_data(self, data):
        data = data.flatten()
        _scale = data.max() - data.min()
        _mean = torch.mean(data)
        _std = torch.std(data)
        _skewness = skew(data)
        _kurt = kurtosis(data)
        
        self.result["stats_scale"] = _scale
        self.result["stats_mean"] = _mean
        self.result["stats_std"] = _std
        self.result["stats_skewness"] = _skewness
        self.result["stats_kurt"] = _kurt


    def _get_model(self, in_features, out_features):
        model = super()._get_model(in_features, out_features)
        if model is not None:
            return model
        model_type = self.args.model_type
        if model_type == "branch_siren":
            model = BranchSiren(
                branch=4,
                in_features=in_features,
                out_features=out_features,
                hidden_layers=self.args.hidden_layers,
                hidden_features=self.args.hidden_features,
                first_omega_0=self.args.first_omega,
                hidden_omega_0=self.args.hidden_omega,
            )
        else:
            raise NotImplementedError

        return model

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

        elif self.norm_method == "channel":
            norm_mark_list = []
            c, h, w = data.shape
            normed_data = torch.zeros_like(data)
            # todo: 后续优化
            for i in range(c):
                _mean = data[i].mean()
                _std = data[i].std()
                norm_mark_list.append((_mean, _std))
                normed_data[i] = (data[i] - _mean) / (_std + 1e-5)

            self.recover_norm_map[mark] = norm_mark_list
        
        elif self.norm_method == "z_power":
            _meta, normed_data = self._power_normalization(data)
            ## → [0,1] → [0,255] → z-std.
            normed_data = self.decode_zero_mean(normed_data)
            _min, _max, gamma, _left_shift_len, _shift_len = _meta
            normed_data = normed_data * (_max - _min + _shift_len) + (_min - _left_shift_len)
            _mean = normed_data.mean()
            _std = normed_data.std()
            normed_data = (normed_data - _mean) / (_std + 1e-5)
            _meta.extend([_mean,_std])
            self.recover_norm_map[mark] = _meta

        elif self.norm_method == "power":
            _meta, normed_data = self._power_normalization(data)
            self.recover_norm_map[mark] = _meta

        elif self.norm_method == "box_cox":
            _min, _max, _lambda, normed_data = self._box_cox_normalization(data)
            self.recover_norm_map[mark] = (_min, _max, _lambda)

        elif self.norm_method == "power_channel":
            ## 先简单写下 后续再去抽象
            norm_mark_list = []
            c, h, w = data.shape
            normed_data = torch.zeros_like(data)
            for i in range(c):
                _meta, cur_normed_data = self._power_normalization(data[i])
                norm_mark_list.append(
                    _meta
                )
                normed_data[i] = cur_normed_data

            self.recover_norm_map[mark] = norm_mark_list
        else:
            raise NotImplementedError

        # 测试振幅
        normed_data = normed_data * self.args.amp
        return normed_data
    
    def _get_gamma_by_mle(self, data):
        data = data.flatten()
        _std = data.std()
        data = (data - data.min()) / (data.max()-data.min())
        data = data.numpy()
        
        def likelihood(gamma, data):
            data = np.power(data,  gamma)
            log_likelihood = np.sum(norm.logpdf(data, loc=0.5, scale=_std))
            return -log_likelihood

        result = minimize(likelihood, 1.0 , args=(data,), bounds=[(0, 5)])
        gamma = result.x[0]
        return gamma
    
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
        
        ### fource setting gamma
        if self.args.force_gamma > 0:
            gamma = self.args.force_gamma
        
        
        
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

    def _box_cox_normalization(self, data: torch.tensor):
        ### + shift → boxcox → min-max → zero-mean
        data = data + self.args.box_shift * 256
        _shape = data.shape
        data, _lambda = stats.boxcox(data.flatten())
        data = torch.tensor(data)
        data = data.reshape(_shape)
        log.inst.info(f"lambda _{_lambda}")

        _min = torch.min(data)
        _max = torch.max(data)
        data = (data - _min) / (_max - _min)  # [0,1]
        normed_data = self.encode_zero_mean(data)
        return _min, _max, _lambda, normed_data

    def _inverse_box_cox_normalization(self, _min, _max, _lambda, normalized_data):
        ### zero-mean → min-max → boxcox → - shift
        normalized_data = self.decode_zero_mean(normalized_data)
        normalized_data = torch.clamp(normalized_data, min=0.0, max=1.0)
        normalized_data = normalized_data * (_max - _min) + _min

        if _lambda == 0:
            normalized_data = torch.exp(normalized_data)
        else:
            normalized_data = (_lambda * normalized_data + 1) ** (1 / _lambda)

        r_data = normalized_data - self.args.box_shift * 256
        return r_data

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

    def _band_normalization(self, band_data, mark="default"):
        # 也可以按照通道数走 (没用)

        _mean = band_data.mean()
        _std = band_data.std()
        band_data = (band_data - _mean) / (_std + 1e-5)

        _min = torch.min(band_data)
        _max = torch.max(band_data)
        band_data = (band_data - _min) / (_max - _min)

        self.recover_norm_map[mark] = (_mean, _std, _min, _max)
        return band_data

    def _band_denormalization(self, band_data, mark="default"):
        _mean, _std, _min, _max = self.recover_norm_map[mark]
        band_data = band_data * (_max - _min) + _min
        band_data = band_data * (_std + 1e-5) + _mean
        return band_data

    def _parse_fft_coff_0(self, data: torch.tensor, mark="default"):
        # 目前差异过大 感觉没法用
        # 0 → max-min
        _min = torch.min(data)
        _max = torch.max(data)
        self.recover_norm_map[mark] = (_min, _max)
        data = data - _min
        # log
        data = torch.log(data + 1)
        # 0 → 1
        data = data / (_max - _min)
        # encode zero_mean
        data = self.encode_zero_mean(data)
        return data

    def _de_parse_fft_coff_0(self, data: torch.tensor, mark="default"):
        # decode zero_mean
        data = self.decode_zero_mean(data)
        # 0 → max-min
        _min, _max = self.recover_norm_map[mark]
        data = data * (_max - _min)
        # exp
        data = torch.exp(data) - 1
        # min → max
        data = data + _min
        return data

    def _block_normalize_coff(self, permuted_coff: torch.tensor):
        # input : c, block_size_h, block_size_w, num_block_h, num_block_w
        permuted_coff = permuted_coff.permute(
            1, 2, 0, 3, 4
        )  # → block_size_h, block_size_w, c, num_block_h, num_block_w
        coff_matrix = permuted_coff.reshape(self.block_size * self.block_size, -1)
        # todo: min & max先离线存储 后续会优化 (做成函数)
        self._min_map = coff_matrix.min(-1).values
        self._max_map = coff_matrix.max(-1).values
        block_normed_coff = torch.stack(
            [
                (block - self._min_map[i]) / (self._max_map[i] - self._min_map[i])
                for i, block in enumerate(coff_matrix)
            ]
        )
        block_normed_coff = block_normed_coff.reshape(
            self.block_size,
            self.block_size,
            self.C_,
            permuted_coff.shape[-1],
            permuted_coff.shape[-1],
        )
        return block_normed_coff.permute(
            2, 0, 1, 3, 4
        )  # return: c, block_size_h, block_size_w, num_block_h, num_block_w

    def _block_denormalize_coff(self, normalized_coff: torch.tensor):
        normalized_coff = normalized_coff.permute(1, 2, 0, 3, 4)
        coff_matrix = normalized_coff.reshape(self.block_size * self.block_size, -1)
        recoverd_coff = torch.stack(
            [
                block * (self._max_map[i] - self._min_map[i]) + self._min_map[i]
                for i, block in enumerate(coff_matrix)
            ]
        )
        recoverd_coff = recoverd_coff.reshape(
            self.block_size,
            self.block_size,
            self.C_,
            normalized_coff.shape[-1],
            normalized_coff.shape[-1],
        )
        return recoverd_coff.permute(
            2, 0, 1, 3, 4
        )  # return: c, block_size_h, block_size_w, num_block_h, num_block_w

    def _get_coords_gt(self, data, mark="default", use_normalization=True):
        c, h, w = data.shape
        self.C_, self.H_, self.W_ = c, h, w
        coords = torch.stack(
            torch.meshgrid(
                [torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)],
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)

        if use_normalization:
            data = self._normalize_data(data, mark)

        self._profile_data(data)
        # normed_data = self._band_normalization(data, mark)
        # normed_data = self.encode_zero_mean(normed_data)

        gt = data.permute(1, 2, 0).reshape(-1, c)  # hw, c
        return coords.to(self.device), gt.to(self.device)

    def _get_coords_gt_by_permuted_block(self, coff, drop_low_pass=0):
        c, h, w = coff.shape
        permuted_coff = self.tool.permute_block_coff(coff)
        if self.args.block_normalization:
            normed_coff = self._block_normalize_coff(permuted_coff)
            normed_coff = self.encode_zero_mean(normed_coff)
        else:
            normed_coff = self._normalize_data(
                permuted_coff, mark="coff"
            )  # 后续管理成系数
        self.num_block_h = permuted_coff.shape[-2]
        self.num_block_w = permuted_coff.shape[-1]
        # X = self._gen_exp_decay_spacing(-1,1, self.block_size, base=1.2)
        # Y = self._gen_exp_decay_spacing(-1,1, self.block_size, base=1.2)
        X = torch.linspace(-1, 1, self.block_size)
        Y = torch.linspace(-1, 1, self.block_size)
        x = torch.linspace(-1, 1, self.num_block_h)
        y = torch.linspace(-1, 1, self.num_block_w)
        grid = torch.meshgrid(
            [X, Y, x, y], indexing="ij"
        )  # block_size, block_size, block_num, block_num
        coords = torch.stack(grid, dim=-1).reshape(-1, 4)
        gt = normed_coff.permute(1, 2, 3, 4, 0).reshape(-1, c)

        # post process drop low pass data (=0)
        if drop_low_pass > 0:
            reserve_indexes = torch.nonzero(
                self.tool.gen_subband_mask(drop_low_pass, self.block_size).flatten()
            ).flatten()
            coords = (
                coords.reshape(self.block_size * self.block_size, -1)
                .index_select(0, reserve_indexes)
                .reshape(-1, 4)
            )
            gt = (
                gt.reshape(self.block_size * self.block_size, -1)
                .index_select(0, reserve_indexes)
                .reshape(-1, c)
            )

        return coords.to(self.device), gt.to(self.device)

    def _get_coords_gt_by_mag_phase(self, coff):
        self.C_, self.H_, self.W_ = coff.shape
        magnitude = torch.abs(coff)
        phase = torch.angle(coff)

        log_magnitude = torch.log(magnitude + 1)
        phase = phase / 3.1416
        norm_mag = self._normalize_data(log_magnitude, "mag")

        gt = torch.cat(
            (norm_mag.permute(1, 2, 0), phase.permute(1, 2, 0)), dim=2
        ).reshape(-1, 2 * self.C_)
        coords = torch.stack(
            torch.meshgrid(
                [torch.linspace(-1, 1, self.H_), torch.linspace(0, 1, self.W_)],
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        return coords.to(self.device), gt.to(self.device)

    def de_parse_fft_coff(self, pred):
        mag = pred[:, :3]
        phase = pred[:, 3:]

        # deparse mag
        mag = mag.permute(1, 0).reshape(self.C_, self.H_, self.W_)
        mag = self._denormalize_data(mag, "mag")
        mag = torch.exp(mag) - 1

        # deparse phase
        phase = phase.permute(1, 0).reshape(self.C_, self.H_, self.W_)
        phase = phase * 3.1416
        r_coff = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        return r_coff

    def gen_soft_channel(self, img, soft_num=4):  # img: c,h,w
        c, h, w = img.shape
        size = int(np.sqrt(torch.tensor(soft_num)))  # 2
        unfolded = img.unfold(1, size, size).unfold(
            2, size, size
        )  # c, num_block_h, num_block_w, block_size_h, block_size_w
        num_block = unfolded.shape[1]
        unfolded = unfolded.reshape(c, num_block, num_block, soft_num).permute(
            3, 0, 1, 2
        )
        return unfolded  # (soft_num, c, h_, w_)

    def _get_data(self):
        img = self.input_img.to(torch.float32)
        # 注意：这个暂时先不要 | 后面要coff再拿回来
        # coff = self.tool.img2dct(img)
        # self.C_, self.H_, self.W_ = coff.shape

        # @test

        # r_img = self.tool.dct2img(coff)
        # psnr_ = self.compute_psnr(r_img, self.input_img)
        # print('psnr_: ', psnr_)

        if self.mode == Mode.pure_coff.value:
            coords, gt = self._get_coords_gt(coff)
            # @test
            # r_img = self.reconstruct_img(gt)
            # psnr_ = self.compute_psnr(r_img, self.input_img)
            # print('psnr_: ', psnr_)
            return coords, gt

        elif self.mode == Mode.pure_spatial.value:

            # @test 11.png
            # img = img / 255.
            # img  = torch.pow(img, self.gamma)
            # img = img * 255.

            # io.save_cv2(
            #     img.permute(1, 2, 0).numpy(),
            #     self._get_cur_path(f"after_gamma_{self.gamma}.png"),
            # )

            coords, gt = self._get_coords_gt(img)

            # @test 测试normalization方法的无损性
            # r_img = self.reconstruct_img(gt.cpu())
            # psnr_ = self.compute_psnr(r_img, self.input_img)
            # print('psnr_: ', psnr_)
            # exit()

            return coords, gt

        elif self.mode == Mode.spatial_split.value:
            ### sptial split 4 heads
            # img = self._normalize_data(img)
            self.C, self.H, self.W = img.shape

            # 先这么写吧 有效果再说
            # todo: soft channel 实现 按照方块结构

            img_soft_channel = self.gen_soft_channel(img, soft_num=4)
            # @test
            # r_img = self.recover_soft_channel(img_soft_channel)
            # psnr_ = self.compute_psnr(r_img, img)
            # print("psnr_: ", psnr_)

            img_sub_0, img_sub_1, img_sub_2, img_sub_3 = (
                img_soft_channel[0],
                img_soft_channel[1],
                img_soft_channel[2],
                img_soft_channel[3],
            )
            img_sub_0 = img[:, : int(self.H / 2), : int(self.W / 2)]
            img_sub_1 = img[:, : int(self.H / 2), int(self.W / 2) :]
            img_sub_2 = img[:, int(self.H / 2) :, : int(self.W / 2)]
            img_sub_3 = img[:, int(self.H / 2) :, int(self.W / 2) :]

            _norm = True
            coords, gt_0 = self._get_coords_gt(
                img_sub_0, mark="sub0", use_normalization=_norm
            )
            _, gt_1 = self._get_coords_gt(
                img_sub_1, mark="sub1", use_normalization=_norm
            )
            _, gt_2 = self._get_coords_gt(
                img_sub_2, mark="sub2", use_normalization=_norm
            )
            _, gt_3 = self._get_coords_gt(
                img_sub_3, mark="sub3", use_normalization=_norm
            )

            gt = torch.cat((gt_0, gt_1, gt_2, gt_3), dim=-1)
            return coords, gt

        elif self.mode == Mode.permute_coff.value:  # deprecated
            permuted_coff = self.tool.permute_block_coff(coff)

            # @param
            # block_normed_coff = self._normalize_data(permuted_coff)
            block_normed_coff = self._block_normalize_coff(permuted_coff)
            block_normed_coff = self.encode_zero_mean(block_normed_coff)

            # @test
            # recover_coff = self._block_denormalize_coff(block_normed_coff)
            # print(permuted_coff - recover_coff)

            # cordinates
            self.num_block_h = permuted_coff.shape[-2]
            self.num_block_w = permuted_coff.shape[-1]
            # @param
            # X = self._gen_exp_decay_spacing(-1,1, self.block_size, base=1.2)
            # Y = self._gen_exp_decay_spacing(-1,1, self.block_size, base=1.2)
            X = torch.linspace(-1, 1, self.block_size)
            Y = torch.linspace(-1, 1, self.block_size)
            x = torch.linspace(-1, 1, self.num_block_h)
            y = torch.linspace(-1, 1, self.num_block_w)
            grid = torch.meshgrid([X, Y, x, y], indexing="ij")
            coords = torch.stack(grid, dim=-1).reshape(-1, 4)
            gt = block_normed_coff.permute(1, 2, 3, 4, 0).reshape(-1, self.C)  # ..., c

            # @param
            # coords = torch.stack(torch.meshgrid([torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)], indexing='ij'), dim=-1).reshape(-1, 2)
            # gt = self.tool.inverse_permute_block_coff(block_normed_coff).permute(1,2,0).reshape(-1, self.C)
            return coords, gt

        elif self.mode == Mode.dual_coff.value:
            low_pass = self.args.dual_low_pass
            low_pass_mask = self.tool.gen_subband_mask(0, low_pass)
            low_pass_coff = self.tool.exert_block_wise_func(
                coff, lambda block: block * low_pass_mask
            )
            high_pass_mask = self.tool.gen_subband_mask(low_pass, self.block_size)
            high_pass_coff = self.tool.exert_block_wise_func(
                coff, lambda block: block * high_pass_mask
            )

            # low_pass_coff → encode to spatial
            low_pass_img = self.tool.dct2img(low_pass_coff)
            spatial_coords, spatial_gt = self._get_coords_gt(low_pass_img)
            io.save_cv2(
                low_pass_img.permute(1, 2, 0).numpy(),
                self._get_cur_path("low_pass_gt.png"),
            )

            # high_pass_coff → encode to permute coff
            freq_coords, freq_gt = self._get_coords_gt_by_permuted_block(
                high_pass_coff, drop_low_pass=low_pass
            )

            # @test
            # r_img = self.reconstruct_img_from_dual_pred(spatial_gt.cpu(), freq_gt.cpu())
            # psnr_ = self.compute_psnr(r_img, self.input_img)
            # print('psnr_: ', psnr_)

            return [spatial_coords, spatial_gt], [freq_coords, freq_gt]

        elif self.mode == Mode.polar_fft_coff.value:
            # 目前没有实现polar
            coff = torch.fft.rfft2(img)
            coff = torch.fft.fftshift(coff, dim=1)
            self.visual_fft(img, coff)

            coords, gt = self._get_coords_gt_by_mag_phase(coff)
            # @test
            # r_coff = self.de_parse_fft_coff(gt)
            # coff = torch.fft.ifftshift(r_coff, dim=1)
            # r_img = torch.fft.irfft2(coff)
            # psnr_ = self.compute_psnr(torch.real(r_img), self.input_img)
            # print("psnr_: ", psnr_)

            return coords, gt

        elif (
            self.mode == Mode.wavelet_subband.value
            or Mode.wavelet_subband_multinet.value
        ):
            LL, LH, HL, HH = self.tool.dwt2_torch(img, wavelet=self.wavelet)
            self.visual_dwt(LL[0], LH[0], HL[0], HH[0])

            # band_normalization
            coords, LL_gt = self._get_coords_gt(LL, mark="LL")
            _, LH_gt = self._get_coords_gt(LH, mark="LH")
            _, HL_gt = self._get_coords_gt(HL, mark="HL")
            _, HH_gt = self._get_coords_gt(HH, mark="HH")

            self.visual_dwt(
                LL_gt.reshape(self.H_, self.W_, -1)[:, :, 0].cpu(),
                LH_gt.reshape(self.H_, self.W_, -1)[:, :, 0].cpu(),
                HL_gt.reshape(self.H_, self.W_, -1)[:, :, 0].cpu(),
                HH_gt.reshape(self.H_, self.W_, -1)[:, :, 0].cpu(),
                save_name="normlized_dwt",
            )

            gt = torch.cat((LL_gt, LH_gt, HL_gt, HH_gt), dim=-1)

            # vanilla 接的4个头
            # gt_data = torch.cat((LL,LH,HL,HH), dim=0)
            # coords, gt = self._get_coords_gt(gt_data
            return coords, gt

            # @test
            # r_img = self.tool.idwt2_torch(LL, LH, HL, HH, wavelet = self.wavelet)
            # psnr_ = self.compute_psnr(r_img, self.input_img)
            # print("psnr_: ", psnr_)

        else:
            raise NotImplementedError

    def _register_model(self, coords, gt):
        in_features = coords.shape[-1]
        out_features = gt.shape[-1]

        model = self._get_model(in_features, out_features).to(self.device)

        optimizer = torch.optim.Adam(lr=self.args.lr, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / self.args.num_epochs, 1)
        )
        return model, optimizer, scheduler

    def train(self):
        log.inst.info(f"####### strat training for {self.data_name}")
        if self.mode == Mode.dual_coff.value:
            self.train_dual_net()
            return

        elif self.mode == Mode.wavelet_subband_multinet.value:
            self.train_multi_net()
            return

        coords, gt = self._get_data()
        model, optimizer, scheduler = self._register_model(coords, gt)

        if self.mode == Mode.wavelet_subband.value:
            self.tool.register_idwt_func(wavelet=self.wavelet, cuda=True)
            self.input_img_cuda = self.input_img.to(self.device)

        if self.use_branch_train:
            subband_gt = torch.stack(torch.chunk(gt, 4, dim=-1))

        for epoch in trange(1, self.args.num_epochs + 1):
            log.start_timer(f"train")
            # @test 分分支训练
            if self.use_branch_train:
                all_pred = []
                for i in range(model.branch):  # branch size
                    pred = model.forward_branch(coords, i)
                    all_pred.append(pred)
                    gt_ = subband_gt[i]
                    loss = self.compute_mse(pred, gt_)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                pred = torch.cat(all_pred, dim=-1)

            else:
                pred = model(coords)
                # @param
                loss = self.compute_mse(pred, gt)
                # loss = self.compute_band_aware_loss(pred, gt, epoch)
                # r_img = self.reconstruct_img(pred, cuda=True)
                # loss = self.compute_mse(r_img/255, self.input_img_cuda/255)
                # print("mse loss", loss.item())
                # loss = self.compute_block_loss(pred, gt, epoch)

                # @fixme: 需要找到torch版本DCT才能用
                # recons_pred = self.reconstruct_img(pred)
                # psnr = self.compute_psnr(recons_pred, self.input_img)
                # if self._use_laplacian_loss:
                #     laplacian_loss = compute_laplacian_loss(r_img, self.input_img_cuda)
                #     # print("lap loss", loss)
                #     loss += self.args.lambda_l * laplacian_loss
                #     # print("total loss", loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # torch.cuda.synchronize()
            log.pause_timer("train")

            if epoch % self.args.log_epoch == 0:
                # @todo: torch版本dct即可让psnr的计算放在cuda上
                recons_pred = self.reconstruct_img(pred.detach().cpu())
                psnr = self.compute_psnr(recons_pred, self.input_img)
                ssim = self.compute_ssim(recons_pred, self.input_img)
                log.inst.info(f"epoch: {epoch} → loss {loss:.8f}")
                log.inst.info(f"epoch: {epoch} → PSNR {psnr:.4f}")
                log.inst.info(f"epoch: {epoch} → SSIM {ssim:.4f}")
                self.result[f"psnr_{epoch}"] = psnr
                self.result[f"ssim_{epoch}"] = ssim

                if epoch % self.args.snap_epoch == 0:
                    io.save_cv2(
                        recons_pred.permute(1, 2, 0).numpy(),
                        self._get_cur_path(f"snap_{epoch}_{self.data_name}.png"),
                    )


            writer.inst.add_scalar(
                "train/loss", loss.detach().item(), global_step=epoch
            )
            # writer.inst.add_scalar("train/psnr", psnr.detach().item(), global_step=epoch)

        # add result
        recons_pred = self.reconstruct_img(pred.detach().cpu())
        psnr = self.compute_psnr(recons_pred, self.input_img)
        ssim = self.compute_ssim(recons_pred, self.input_img)
        self.result[f"final_psnr"] = psnr
        self.result[f"final_ssim"] = ssim

        with torch.no_grad():
            pred = model(coords)
            final_img = self.reconstruct_img(pred.cpu()).permute(1, 2, 0)  # h,w,c
            io.save_cv2(
                final_img.numpy(),
                self._get_cur_path(f"final_pred_{self.data_name}.png"),
            )

        self._save_ckpt(epoch, model, optimizer, scheduler)
        log.inst.info(f"####### end training for {self.data_name}")

    def train_multi_net(self):

        coords, gt = self._get_data()
        # model_num = 4
        model_num = 2

        net_list = []
        # gt_chunk = torch.chunk(gt, model_num, dim=-1)
        gt_chunk = (gt[:, :3], gt[:, 3:])

        for i in range(model_num):
            model, optimizer, scheduler = self._register_model(coords, gt_chunk[i])
            cur_dic = {
                "coords": coords,
                "gt": gt_chunk[i],
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
            }
            net_list.append(cur_dic)

        self.tool.register_idwt_func(wavelet=self.wavelet, cuda=True)
        self.input_img_cuda = self.input_img.to(self.device)

        for epoch in trange(1, self.args.num_epochs + 1):
            log.start_timer("train")
            all_pred = []
            # todo: 并行化速度优化
            for i in range(model_num):
                coords, gt, model, optimizer, scheduler = (
                    net_list[i]["coords"],
                    net_list[i]["gt"],
                    net_list[i]["model"],
                    net_list[i]["optimizer"],
                    net_list[i]["scheduler"],
                )
                pred = model(coords)

                # @param
                loss = self.compute_mse(pred, gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                all_pred.append(pred)
                # all_pred.append(gt)

            pred = torch.cat(all_pred, dim=-1)
            torch.cuda.synchronize()
            log.pause_timer("train")

            if epoch % self.args.log_epoch == 0:
                # @todo: torch版本dct即可让psnr的计算放在cuda上
                recons_pred = self.reconstruct_img(pred.detach().cpu())
                psnr = self.compute_psnr(recons_pred, self.input_img)
                log.inst.info(f"epoch: {epoch} → loss {loss:.8f}")
                log.inst.info(f"epoch: {epoch} → PSNR {psnr:.4f}")

                if epoch % self.args.snap_epoch == 0:
                    io.save_cv2(
                        recons_pred.permute(1, 2, 0).numpy(),
                        self._get_cur_path(f"snap_{epoch}.png"),
                    )

            writer.inst.add_scalar(
                "train/loss", loss.detach().item(), global_step=epoch
            )
            # writer.inst.add_scalar("train/psnr", psnr.detach().item(), global_step=epoch)

        with torch.no_grad():
            # pred = model(coords)
            # pred = torch.cat(all_pred, dim=-1) # todo:暂时不重新推理
            final_img = self.reconstruct_img(pred.cpu()).permute(1, 2, 0)  # h,w,c
            io.save_cv2(final_img.numpy(), self._get_cur_path("final_pred.png"))

        self._save_ckpt(epoch, model, optimizer, scheduler)

    def train_dual_net(self):
        ### deprecated 以后要是用的话 就用train_multi_net替代
        spatial_comp, freq_comp = self._get_data()
        spatial_comp += self._register_model(*spatial_comp)
        freq_comp += self._register_model(
            *freq_comp
        )  # spatial_coords, spatial_gt, sp_model, sp_optimizer, sp_scheduler

        spatial_pred = None
        freq_pred = None

        for epoch in trange(1, self.args.num_epochs + 1):
            should_train_spatial = self.schedule_dual_net(epoch)
            comp = spatial_comp if should_train_spatial else freq_comp
            coords, gt, model, optimizer, scheduler = comp

            log.start_timer("train")
            pred = model(coords)

            if should_train_spatial:
                spatial_pred = pred
            else:
                freq_pred = pred

            loss = self.compute_mse(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            log.pause_timer("train")

            if epoch % self.args.log_epoch == 0:
                # @todo: torch版本dct即可让psnr的计算放在cuda上
                recons_pred = self.reconstruct_img_from_dual_pred(
                    spatial_pred.detach().cpu(), freq_pred.detach().cpu()
                )
                psnr = self.compute_psnr(recons_pred, self.input_img)
                log.inst.info(f"epoch: {epoch} → loss {loss:.8f}")
                log.inst.info(f"epoch: {epoch} → PSNR {psnr:.4f}")

                if epoch % self.args.snap_epoch == 0:
                    io.save_cv2(
                        recons_pred.permute(1, 2, 0).numpy(),
                        self._get_cur_path(f"snap_{epoch}.png"),
                    )

            if should_train_spatial:
                writer.inst.add_scalar(
                    "train/spatial_loss", loss.detach().item(), global_step=epoch
                )
            else:
                writer.inst.add_scalar(
                    "train/frequency_loss", loss.detach().item(), global_step=epoch
                )

        with torch.no_grad():
            spatial_pred = spatial_comp[2](spatial_comp[0])
            freq_pred = freq_comp[2](freq_comp[0])
            final_img = self.reconstruct_img_from_dual_pred(
                spatial_pred.cpu(), freq_pred.cpu()
            ).permute(
                1, 2, 0
            )  # h,w,c
            io.save_cv2(final_img.numpy(), self._get_cur_path("final_pred.png"))

        state_dict = {
            "epoch": epoch,
            "freq_model": freq_comp[2].state_dict(),
            "freq_optimizer": freq_comp[3].state_dict(),
            "freq_scheduler": freq_comp[4].state_dict(),
            "spatial_model": spatial_comp[2].state_dict(),
            "spatial_optimizer": spatial_comp[3].state_dict(),
            "spatial_scheduler": spatial_comp[4].state_dict(),
            "args": self.args,
        }
        self._save_ckpt_by_dict(state_dict, epoch)

    def schedule_dual_net(self, epoch) -> bool:
        ratio = self.args.cross_ratio  # 0.6
        base_num = 10
        if epoch < self.args.log_epoch:  # 50
            return bool(epoch % 2)
        else:
            return bool((epoch % base_num) < int(ratio * base_num))

    def reconstruct_img_from_dual_pred(self, spatial_pred, freq_pred):  # c,h,w

        # parse spatial pred
        low_pass = spatial_pred.reshape(self.H_, self.W_, self.C_).permute(
            2, 0, 1
        )  # c,h,w
        low_pass = self._denormalize_data(low_pass)
        low_coff = self.tool.img2dct(low_pass)
        permuted_low_coff = self.tool.permute_block_coff(low_coff)
        low_coff = permuted_low_coff.permute(1, 2, 3, 4, 0).reshape(
            self.block_size, self.block_size, -1
        )

        # parse freq pred
        high_pass = self._denormalize_data(freq_pred, mark="coff")
        # dim =  8x8 - 4x4
        dim_0 = (
            self.block_size * self.block_size
            - self.args.dual_low_pass * self.args.dual_low_pass
        )
        high_coff = high_pass.reshape(dim_0, -1)

        # fuse
        fused_coff = torch.zeros_like(low_coff)
        low_pass_mask = self.tool.gen_subband_mask(0, self.args.dual_low_pass)
        high_idx = 0
        for i in range(self.block_size):
            for j in range(self.block_size):
                if low_pass_mask[i][j]:
                    fused_coff[i][j] = low_coff[i][j]
                else:
                    fused_coff[i][j] = high_coff[high_idx]
                    high_idx += 1

        fused_coff = fused_coff.reshape(
            self.block_size, self.block_size, self.num_block_h, self.num_block_w, -1
        ).permute(4, 0, 1, 2, 3)
        r_coff = self.tool.inverse_permute_block_coff(fused_coff)
        r_img = self.tool.dct2img(r_coff)
        # to image
        r_img = torch.clamp(r_img, min=0, max=255)
        return r_img

    def reconstruct_img(self, pred, cuda=False) -> torch.tensor:  # c,h,w
        # First: decode zero_mean

        if self.mode == Mode.pure_coff.value:
            # @fixme: 这里可能有一个bug: 如果dct加了padding维度会变 → 图像转换回来应该cut掉
            coff = pred.reshape(self.H_, self.W_, self.C_).permute(2, 0, 1)  # c,h,w
            coff = self._denormalize_data(coff)
            r_img = self.tool.dct2img(coff)

        elif self.mode == Mode.pure_spatial.value:
            r_img = pred.reshape(self.H_, self.W_, self.C_).permute(2, 0, 1)
            r_img = self._denormalize_data(r_img)

            # @test 11.png
            # r_img = r_img / 255.
            # r_img = torch.pow(r_img, 1.0 / self.gamma)
            # r_img = r_img * 255.

        elif self.mode == Mode.permute_coff.value:
            # ...,c → c, block_size_h, block_size_w, num_block_h, num_block_w
            pred = self.decode_zero_mean(pred)
            coff = pred.reshape(
                self.block_size,
                self.block_size,
                self.num_block_h,
                self.num_block_w,
                self.C,
            ).permute(4, 0, 1, 2, 3)
            coff = self._block_denormalize_coff(coff)
            coff = self.tool.inverse_permute_block_coff(coff)
            r_img = self.tool.dct2img(coff)
        elif self.mode == Mode.polar_fft_coff.value:
            r_coff = self.de_parse_fft_coff(pred)
            coff = torch.fft.ifftshift(r_coff, dim=1)
            r_img = torch.fft.irfft2(coff)

        elif (
            self.mode == Mode.wavelet_subband.value
            or self.mode == Mode.wavelet_subband_multinet.value
        ):
            coff = pred.reshape(self.H_, self.W_, -1).permute(2, 0, 1)  # c,h,w
            # coff = self._denormalize_data(coff)
            LL = coff[:3, :, :]
            LH = coff[3:6, :, :]
            HL = coff[6:9, :, :]
            HH = coff[9:12, :, :]

            # de-band-normalization
            LL = self._denormalize_data(LL, mark="LL")
            LH = self._denormalize_data(LH, mark="LH")
            HL = self._denormalize_data(HL, mark="HL")
            HH = self._denormalize_data(HH, mark="HH")
            # LL = self._band_denormalization(LL,mark="LL")
            # LH = self._band_denormalization(LH,mark="LH")
            # HL = self._band_denormalization(HL,mark="HL")
            # HH = self._band_denormalization(HH,mark="HH")

            if cuda:
                r_img = self.tool.idwt2_torch_by_cache_func(LL, LH, HL, HH)
            else:
                r_img = self.tool.idwt2_torch(LL, LH, HL, HH, wavelet=self.wavelet)

        elif self.mode == Mode.spatial_split.value:
            # pred = self._denormalize_data(pred)

            pred = pred.reshape(int(self.H / 2), int(self.W / 2), -1).permute(
                2, 0, 1
            )  # c,h,w

            pred_0 = pred[:3, :, :]
            pred_1 = pred[3:6, :, :]
            pred_2 = pred[6:9, :, :]
            pred_3 = pred[9:12, :, :]

            pred_0 = self._denormalize_data(pred_0, mark="sub0")
            pred_1 = self._denormalize_data(pred_1, mark="sub1")
            pred_2 = self._denormalize_data(pred_2, mark="sub2")
            pred_3 = self._denormalize_data(pred_3, mark="sub3")

            # todo: recover_soft_channel
            soft_channel_img = torch.stack([pred_0, pred_1, pred_2, pred_3], dim=0)
            r_img = self.recover_soft_channel(soft_channel_img)

            ###### 空间分
            r_img = torch.zeros_like(self.input_img).to(torch.float32)
            r_img[:, : int(self.H / 2), : int(self.W / 2)] = pred_0
            r_img[:, : int(self.H / 2), int(self.W / 2) :] = pred_1
            r_img[:, int(self.H / 2) :, : int(self.W / 2)] = pred_2
            r_img[:, int(self.H / 2) :, int(self.W / 2) :] = pred_3

        else:
            raise NotImplementedError

        r_img = torch.clamp(r_img, min=0, max=255)

        return r_img

    def recover_soft_channel(self, soft_channel_img):  # 4 * c * h * w
        # todo: 先这么写 后续再改
        soft_num, c, h, w = soft_channel_img.shape
        size = int(np.sqrt(soft_num))
        ## 注意这个permute block_num在前 block_size在后
        r_img = soft_channel_img.reshape(size, size, c, h, w).permute(
            2, 3, 0, 4, 1
        )  # # c, size_h, h, size_w, w
        r_img = r_img.reshape(c, size * h, size * w)
        return r_img

    @staticmethod
    def _gen_gaussian_weight(size, sigma, device):
        x = torch.arange(size, device=device)
        y = torch.arange(size, device=device)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        d = torch.sqrt(x_grid**2 + y_grid**2)
        g = torch.exp(-(d**2 / (2.0 * sigma**2)))
        g /= torch.sum(g)  # sum to 1
        return g

    @staticmethod
    def _gen_exp_decay_weight(size, base, device):
        # untest
        #     weights = torch.tensor(
        #         [
        #             [256, 128, 64, 32, 16, 8, 4, 2],
        #             [128, 128, 64, 32, 16, 8, 4, 2],
        #             [64, 64, 64, 32, 16, 8, 4, 2],
        #             [32, 32, 32, 32, 16, 8, 4, 2],
        #             [16, 16, 16, 16, 16, 8, 4, 2],
        #             [8, 8, 8, 8, 8, 8, 4, 2],
        #             [4, 4, 4, 4, 4, 4, 4, 2],
        #             [2, 2, 2, 2, 2, 2, 2, 2],
        #         ]
        #     )
        weights = torch.zeros((size, size), device=device)
        for i in range(size):
            value = base / (2**i)
            weights[i, : size - i] = value
        weights /= torch.sum(weights)  # sum to 1
        return weights

    @staticmethod
    def _gen_exp_decay_spacing(start, end, size, base=2):
        spacing = torch.arange(start=size - 2, end=-1, step=-1)
        spacing = base**spacing
        distances = torch.cumsum(spacing, dim=0)
        values = torch.cat((torch.tensor([0]), distances))
        normalized = (values - values[0]) / (values[-1] - values[0])
        normalized = normalized * (end - start) + start
        return normalized

    # @staticmethod
    # def _gen_half_gaussian_spacing(size, start=-1, end=1, std=1.0):
    #     # untest
    #     half_gauss = torch.abs(torch.randn(size-1) * std)
    #     half_gauss = (half_gauss - half_gauss.min()) / (half_gauss.max() - half_gauss.min())
    #     total_distance = end - start
    #     half_gauss = half_gauss * total_distance / half_gauss.sum()
    #     positions = torch.cat([torch.tensor([start]), torch.cumsum(half_gauss, dim=0)])
    #     return positions

    def compute_block_loss(self, pred, gt, epoch):
        # ...,c → block_size_h* block_size_w, num_block_h * num_block_w,c
        pred = pred.reshape(
            self.block_size * self.block_size,
            self.num_block_h * self.num_block_w,
            self.C,
        )
        gt = gt.reshape(
            self.block_size * self.block_size,
            self.num_block_h * self.num_block_w,
            self.C,
        )
        block_mse = F.mse_loss(pred, gt, reduction="none")
        block_mse = block_mse.reshape(block_mse.shape[0], -1).mean(-1)

        ##### weight set ######
        init_sigma = 3
        final_sigma = 5
        cur_sigma = init_sigma + (final_sigma - init_sigma) * (
            epoch / self.args.num_epochs
        )
        cur_sigma = 2.5
        block_loss_weight = self._gen_gaussian_weight(
            self.block_size, cur_sigma, device=self.device
        )
        block_loss_weight = block_loss_weight.reshape(-1)

        block_mse = block_mse @ block_loss_weight

        return block_mse

    def compute_band_aware_loss(self, pred, gt, epoch):
        subband_pred = torch.stack(torch.chunk(pred, 4, dim=-1))
        subband_gt = torch.stack(torch.chunk(gt, 4, dim=-1))
        subband_mse = (
            F.mse_loss(subband_pred, subband_gt, reduction="none")
            .reshape(4, -1)
            .mean(-1)
        )
        init_lam = 2
        end_lam = 1.95
        lam = init_lam - (init_lam - end_lam) * (epoch / self.args.num_epochs)
        lam = 2
        weight = torch.tensor(
            [lam**2, lam, lam, 1], dtype=torch.float32, device=self.device
        )
        return subband_mse @ weight

    def visual_fft(self, img: torch.tensor, fft_coff: torch.tensor):
        # log for better
        # @todo: visualize complex and real

        # fft_coff = torch.fft.fftshift(fft_coff)
        magnitude = torch.abs(fft_coff)
        phase = torch.angle(fft_coff)
        real = torch.real(fft_coff)
        imag = torch.imag(fft_coff)

        mag_log = torch.log(magnitude + 1)
        real_log = torch.log(torch.abs(real) + 1)
        imag_log = torch.log(torch.abs(imag) + 1)

        plt.subplot(2, 3, 1)
        plt.title("Original Image", fontsize=5)
        plt.imshow(transforms.ToPILImage()(img))
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.title("Magnitude Spectrum (Log Scale) - Red Channel", fontsize=5)
        plt.imshow(mag_log[0])
        plt.colorbar(shrink=0.9)
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.title("Original Image - Red Channel", fontsize=5)
        plt.imshow(transforms.ToPILImage()(img[0]))
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.title("Phase Spectrum - Red Channel", fontsize=5)
        plt.imshow(phase[0])
        plt.colorbar(shrink=0.9)
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.title("Real Part - Red Channel", fontsize=5)
        plt.imshow(real_log[0])
        plt.colorbar(shrink=0.9)
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.title("Imag Part - Red Channel", fontsize=5)
        plt.imshow(imag_log[0])
        plt.colorbar(shrink=0.9)
        plt.axis("off")

        plt.savefig(self._get_cur_path("visual_fft.png"), dpi=300)
        plt.close()

    def visual_dwt(self, LL, LH, HL, HH, save_name="visual_dwt"):

        plt.subplot(2, 2, 1)
        plt.title("LL", fontsize=5)
        plt.imshow(LL)
        plt.colorbar(shrink=0.9)
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("LL", fontsize=5)
        plt.imshow(LH)
        plt.colorbar(shrink=0.9)
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("HL", fontsize=5)
        plt.imshow(HL)
        plt.colorbar(shrink=0.9)
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("HH", fontsize=5)
        plt.imshow(HH)
        plt.colorbar(shrink=0.9)
        plt.axis("off")

        plt.savefig(self._get_cur_path(f"{save_name}.png"), dpi=300)
        plt.close()
