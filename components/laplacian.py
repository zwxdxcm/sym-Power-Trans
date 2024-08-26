import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()

    def forward(self, img1, img2):
        if img1.shape[1] == 3:
            img1 = TF.rgb_to_grayscale(img1)
        if img2.shape[1] == 3:
            img2 = TF.rgb_to_grayscale(img2)
        laplacian1 = self.laplacian(img1)
        laplacian2 = self.laplacian(img2)
        laplacian_loss = F.mse_loss(laplacian1, laplacian2, reduction="mean")
        return laplacian_loss

    def laplacian(self, img):
        laplacian_kernel = (
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                dtype=torch.float32,
                device=img.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # img = F.pad(img, (1, 1, 1, 1), mode='reflect')
        laplacian = F.conv2d(img, laplacian_kernel)
        return laplacian


laplacian_loss = LaplacianLoss()


def compute_laplacian_loss(img1, img2):
    img1 = img1.unsqueeze(0).to(torch.float32)  # 1,c,h,w
    img2 = img2.unsqueeze(0).to(torch.float32)
    return laplacian_loss(img1, img2)
