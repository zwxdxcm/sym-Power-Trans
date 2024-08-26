from models import Siren, PEMLP
from util.logger import log
from util.tensorboard import writer
from util.misc import gen_cur_time
from util import io
from trainer.base_trainer import BaseTrainer
from components.dct import DCTImageProcessor
from components.ssim import compute_ssim_loss
from components.laplacian import compute_laplacian_loss

import numpy as np
import imageio.v2 as imageio
import torch
import os

from tqdm import trange


class ImageTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self._set_status()
        self._parse_input_data()

    def _set_status(self):
        self._use_laplacian_loss = bool(self.args.lambda_l > 0)

    def _parse_input_data(self):
        path = self.args.input_path
        img = torch.from_numpy(imageio.imread(path)).permute(2, 0, 1)  # c,h,w
        self.input_img = img
        self.C, self.H, self.W = img.shape

    def _encode_img(self, img, zero_mean=True):
        img = torch.clamp(img, min=0, max=255)
        img = img / 255.0
        if zero_mean:
            img = self.encode_zero_mean(img)
        return img

    def _decode_img(self, data, zero_mean=True):
        if zero_mean:
            data = self.decode_zero_mean(data)
        data = data * 255.0
        data = torch.clamp(data, min=0, max=255)
        return data

    def _get_data(self):
        img = self.input_img
        img = self._encode_img(img)
        gt = img.permute(1, 2, 0).reshape(-1, self.C)  # h*w, C
        coords = torch.stack(
            torch.meshgrid(
                [torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)],
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        return coords, gt

    def train(self):
        num_epochs = self.args.num_epochs
        coords, gt = self._get_data()
        model = self._get_model(in_features=2, out_features=3).to(self.device)

        coords = coords.to(self.device)
        gt = gt.to(self.device)
        self.input_img = self.input_img.to(self.device)

        optimizer = torch.optim.Adam(lr=self.args.lr, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1)
        )

        for epoch in trange(1, num_epochs + 1):
            log.start_timer("train")
            pred = model(coords)
            loss = self.compute_mse(pred, gt)
            recons_pred = self.reconstruct_img(pred)
            psnr = self.compute_psnr(recons_pred, self.input_img)
            if self._use_laplacian_loss:
                loss += self.args.lambda_l * compute_laplacian_loss(
                    recons_pred, self.input_img
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            log.pause_timer("train")

            if epoch % self.args.log_epoch == 0:
                log.inst.info(f"epoch: {epoch} → loss {loss:.8f}")
                log.inst.info(f"epoch: {epoch} → PSNR {psnr:.4f}")

            writer.inst.add_scalar(
                "train/loss", loss.detach().item(), global_step=epoch
            )
            writer.inst.add_scalar(
                "train/psnr", psnr.detach().item(), global_step=epoch
            )

        with torch.no_grad():
            pred = model(coords)
            final_img = (
                self.reconstruct_img(pred).permute(1, 2, 0).cpu().numpy()
            )  # h,w,c
            io.save_cv2(final_img, self._get_cur_path("final_pred.png"))

        self._save_ckpt(epoch, model, optimizer, scheduler)

    def reconstruct_img(self, data) -> torch.tensor:
        img = data.reshape(self.H, self.W, self.C).permute(2, 0, 1)  # c,h,2
        img = self._decode_img(img)
        return img

    @staticmethod
    def compute_psnr(pred, gt):
        """image data"""
        mse = BaseTrainer.compute_mse(pred, gt)
        return 20.0 * torch.log10(gt.max() / torch.sqrt(mse))
    
    @staticmethod
    def compute_ssim(pred, gt):
        return compute_ssim_loss(pred, gt)

    @staticmethod
    def compute_normalized_psnr(pred, gt, zero_mean=True):
        """normalized 0 - 1"""
        if zero_mean:
            pred = BaseTrainer.decode_zero_mean(pred)
            gt = BaseTrainer.decode_zero_mean(gt)
        mse = BaseTrainer.compute_mse(pred, gt)
        return -10.0 * torch.log10(mse)
