import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from .image_infer import Deep3DImageInfer as OriInfer


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class Deep3DImageInfer(object):
    def __init__(self, device: str = "cuda:0"):
        self.infer = OriInfer()

    def infer_tensor(self, x: torch.Tensor):
        array_x: np.ndarray = ((x + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_x = Image.fromarray(array_x[0])
        param: dict = self.infer.infer_image(img_pil=pil_x)
        return param  # keys:['id', 'exp', 'tex', 'angle', 'gamma', 'trans']

    def calc_l2(self, v1: torch.Tensor, v2: torch.Tensor):
        exp_dim = v1.shape[-1]
        mse_sum = torch.nn.functional.mse_loss(v1, v2, reduction='mean')
        l2 = torch.sqrt(mse_sum * exp_dim)
        print('deep3d MSE_SUM=%.3f, L2=%.3f, dim=%d' % (mse_sum.data, l2.data, exp_dim))
        return l2
