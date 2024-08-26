import numpy as np
from scipy.fftpack import dct, idct
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

from typing import Callable


class DCTTool(object):
    def __init__(self, block_size=8):
        self.block_size = block_size

    @staticmethod
    # todo: update to torch version
    def dct2(block: np.ndarray):
        return dct(dct(block.T, norm="ortho").T, norm="ortho")

    @staticmethod
    def idct2(block: np.ndarray):
        return idct(idct(block.T, norm="ortho").T, norm="ortho")

    @staticmethod
    def dct2_torch(block: torch.tensor):
        return torch.tensor(DCTTool.dct2(block.numpy())).to(torch.float32)

    @staticmethod
    def idct2_torch(block: torch.tensor):
        return torch.tensor(DCTTool.idct2(block.numpy()))

    @staticmethod
    def dwt2_torch(img: torch.tensor, wavelet="haar"):
        img = img.unsqueeze(0).to(torch.float32)
        dwt = DWTForward(J=1, wave=wavelet, mode="zero")
        yl, yh = dwt(img)
        LL = yl.squeeze()
        LH, HL, HH = [subband for subband in yh[0].squeeze().permute(1,0,2,3)] # 3, c, h, w
        return LL, LH, HL, HH
    
    @staticmethod
    def idwt2_torch(LL, LH, HL, HH, wavelet="haar"):
        # TODO: 修改版本
        yl = LL.unsqueeze(0)
        yh = [torch.stack((LH, HL, HH), dim=0).permute(1,0,2,3).unsqueeze(0)] # 1, c, 3, h, w
        idwt = DWTInverse(wave=wavelet, mode="zero")
        r_img = idwt((yl, yh))
        return r_img.squeeze()
    
    def register_idwt_func(self, wavelet="haar", cuda= False):
        self.idwt_func = DWTInverse(wave=wavelet, mode="zero")
        if cuda:
            self.idwt_func = self.idwt_func.cuda()
    
    def idwt2_torch_by_cache_func(self, LL, LH, HL, HH):
        yl = LL.unsqueeze(0)
        yh = [torch.stack((LH, HL, HH), dim=0).unsqueeze(0)]
        r_img = self.idwt_func((yl, yh))
        return r_img.squeeze()


    def pad_img(self, img: torch.tensor):
        c, h, w = img.shape
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
        padded = F.pad(img, padding, mode="constant", value=0)
        return padded

    def img2dct(self, img: torch.tensor):
        return self.exert_block_wise_func(img, self.dct2_torch)

    def dct2img(self, coff: torch.tensor):
        return self.exert_block_wise_func(coff, self.idct2_torch)

    def exert_block_wise_func(
        self, data: torch.tensor, func: Callable[[torch.tensor], torch.tensor]
    ):
        c, h, w = data.shape
        size = self.block_size
        num_block_h = int(h / size)
        num_block_w = int(w / size)
        unfolded = data.unfold(1, size, size).unfold(
            2, size, size
        )  # c, num_block_h, num_block_w, block_size_h, block_size_w
        unfolded = unfolded.permute(1, 2, 0, 3, 4).reshape(-1, size, size)
        exerted = torch.stack([func(block) for block in unfolded])
        exerted = (
            exerted.reshape(num_block_h, num_block_w, c, size, size)
            .permute(2, 0, 3, 1, 4)
            .reshape(c, h, w)
        )
        return exerted

    def gen_subband_mask(self, low_freq, high_freq):
        mask = torch.full((self.block_size, self.block_size), False, dtype=torch.bool)
        mask[:low_freq, :low_freq] = ~mask[:low_freq, :low_freq]
        mask[:high_freq, :high_freq] = ~mask[:high_freq, :high_freq]
        return mask

    def permute_block_coff(self, coff: torch.tensor):
        size = self.block_size
        unfolded = coff.unfold(1, size, size).unfold(
            2, size, size
        )  # → c, num_block_h, num_block_w, block_size_h, block_size_w
        permuted = unfolded.permute(
            0, 3, 4, 1, 2
        )  # → c, block_size_h, block_size_w, num_block_h, num_block_w (3, 8, 8, 64, 64)
        return permuted

    def inverse_permute_block_coff(self, permuted: torch.tensor):
        inverse = permuted.permute(
            0, 3, 1, 4, 2
        )  # → c, num_block_h, block_size_h, num_block_w, block_size_w
        return inverse.reshape(
            inverse.shape[0], inverse.shape[1] * inverse.shape[2], -1
        )  # → c, h, w
