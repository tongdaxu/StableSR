import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from ldm.models.diffusion.resizer import Resizer
import kornia
from torchvision.transforms.functional import to_pil_image
from functools import partial

class SuperResolutionOperator(nn.Module):
    def __init__(self, in_shape, scale_factor):
        super(SuperResolutionOperator, self).__init__()
        self.scale_factor = scale_factor
        self.down_sample = Resizer(in_shape, 1/scale_factor)
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)

    def forward(self, x, keep_shape=False):
        x = (x + 1.0) / 2.0
        y = self.down_sample(x)
        y = (y - 0.5) / 0.5
        if keep_shape:
            y = F.interpolate(y, scale_factor=self.scale_factor, mode='bicubic')
        return y

    def transpose(self, y):
        return self.up_sample(y)

    def y_channel(self):
        return 3
    
    def to_pil(self, y):
        y = (y[0] + 1.0) / 2.0
        y = torch.clip(y, 0, 1)
        y = to_pil_image(y, 'RGB')
        return y
