import os
import numpy as np
import torch
import torch.nn.functional as F

from . import net


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class CosFaceImageInfer(object):
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.size = (112, 112)
        in_size = 256
        self.trans_matrix = torch.tensor(
            [[[1.07695457, -0.03625215, -1.56352194 * (in_size / 256)],
              [0.03625215, 1.07695457, -5.32134629 * (in_size / 256)]]],
            requires_grad=False,).float()

        self.model = net.sphere().to(self.device)
        weights = make_abs_path("../weights/cosface/net_sphere20_data_vggface2_acc_9955.pth")
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

    def infer_tensor(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if (h, w) != self.size:  # if x is ffhq aligned
            x = F.interpolate(x, size=256, mode="bilinear", align_corners=True)  # first resize to 256x256
            m = self.trans_matrix.repeat(b, 1, 1)  # to (B,2,3)
            grid = F.affine_grid(m, size=x.shape, align_corners=True)  # 得到 grid 用于 grid sample
            x = F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="zeros")  # warp affine
            x = F.interpolate(x, size=112, mode="bilinear", align_corners=True)  # resize to 112x112
        else:  # x is arcface aligned
            x = x
        if x.device != self.device:
            x = x.to(self.device)
        embeddings: torch.Tensor = self.model(x)
        return embeddings
