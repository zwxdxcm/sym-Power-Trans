from trainer.base_trainer import BaseTrainer
import torchaudio
import torch
from util.logger import log
import numpy as np
import pesq
from pystoi import stoi
from tqdm import trange
import soundfile as sf
import os
import matplotlib.pyplot as plt


class AudioTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self._set_status()
        self._parse_input_data()
        self.recover_norm_map = {}
       
    
    
    def _set_status(self):
        self.sample_rate = 16000
        self.crop_sec = 5 # 取前10s
        self.result = {}

        self.norm_method = self.args.normalization  
        self.data_name = os.path.basename(self.args.input_path).split(".")[0]
    
    def _parse_input_data(self):
        audio_path = self.args.input_path
        waveform, sample_rate = torchaudio.load(audio_path)
        crop_samples = self.sample_rate * self.crop_sec
        self.input_audio = waveform.squeeze()
        final_len = min(crop_samples,len(self.input_audio))
        self.input_audio = self.input_audio[:final_len]
        self.T = final_len

    def _get_data(self, mark="default", use_normalization=True):
        # parse input audio
        data = self.input_audio
        if use_normalization:
            data = self._normalize_data(data, mark)
        gt = data
        coords = torch.linspace(-1, 1, self.T)

        gt = gt.reshape(-1,1)
        coords = coords.reshape(-1,1)

        # test 
        # r_audio = self.reconstruct_audio(gt)
        # mse = self.compute_mse(r_audio, self.input_audio)
        # snr = self.compute_snr(r_audio, self.input_audio)
        # pesq = self.compute_pesq(r_audio, self.input_audio)
        # stoi = self.compute_stoi(r_audio, self.input_audio)
        # print('stoi: ', stoi)
        # print('pesq: ', pesq)
        # print('mse: ', mse)
        # print('snr: ', snr)
        return coords, gt

    def reconstruct_audio(self, pred):
        pred = pred.squeeze()
        r_audio = self._denormalize_data(pred)
        return r_audio
    
    def compute_snr(self, pred, gt):
        
        noise = gt - pred
        gt_power = torch.mean(gt**2)
        noise_power = torch.mean(noise**2)
        snr = 10 * torch.log10(gt_power / noise_power)
        return snr
    
    def compute_si_snr(self, pred, gt):
        gt = gt - gt.mean()
        pred = pred - pred.mean()
        return self.compute_snr(pred, gt)
    
    def compute_stoi(self, pred, gt):
        return stoi(gt, pred, self.sample_rate, extended=False)

    def compute_pesq(self, pred, gt):
        pesq_score = pesq.pesq(self.sample_rate, gt.numpy(), pred.numpy(), mode='wb')
        return pesq_score

    def train(self):
        num_epochs = self.args.num_epochs
        coords, gt = self._get_data()
        model = self._get_model(in_features=1, out_features=1).to(self.device)

        coords = coords.to(self.device)
        gt = gt.to(self.device)

        optimizer = torch.optim.Adam(lr=self.args.lr, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1)
        )

        for epoch in trange(1, num_epochs + 1):
            log.start_timer("train")
            pred = model(coords)
            loss = self.compute_mse(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            log.pause_timer("train")

            if epoch % self.args.log_epoch == 0:
                r_audio = self.reconstruct_audio(pred).detach().cpu()
                mse = self.compute_mse(r_audio, self.input_audio)
                snr = self.compute_snr(r_audio, self.input_audio)
                si_snr = self.compute_si_snr(r_audio, self.input_audio)
                pesq = self.compute_pesq(r_audio, self.input_audio)
                stoi = self.compute_stoi(r_audio, self.input_audio)

                log.inst.info(f"epoch: {epoch} → mse {mse:.8f}")
                log.inst.info(f"epoch: {epoch} → snr {snr:.4f}")
                log.inst.info(f"epoch: {epoch} → si_snr {si_snr:.4f}")
                log.inst.info(f"epoch: {epoch} → pesq {pesq:.4f}")
                log.inst.info(f"epoch: {epoch} → stoi {stoi:.4f}")

                self.result[f"mse_{epoch}"] = mse
                self.result[f"sisnr_{epoch}"] = si_snr
                self.result[f"stoi_{epoch}"] = stoi
                self.result[f"pesq_{epoch}"] = pesq


        with torch.no_grad():
            pred = model(coords)
            error = gt - pred
            sf.write(self._get_cur_path(f"final_pred_{self.data_name}.wav"), pred.cpu(), self.sample_rate)
            self.plot_error_signal(error.cpu())

        self._save_ckpt(epoch, model, optimizer, scheduler)

    def plot_error_signal(self, error_signal):
        sr = self.sample_rate
        out_path = self._get_cur_path(f"final_error_{self.data_name}.png")
        time_axis = np.linspace(0, len(error_signal) / sr, num=len(error_signal))
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, error_signal, label="Error Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Error Signal Waveform")
        plt.legend()
        plt.grid()
        plt.savefig(out_path)


    
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

