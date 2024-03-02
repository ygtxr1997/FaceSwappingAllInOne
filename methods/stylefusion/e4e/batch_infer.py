import torch
import numpy as np
import sys
import os
import dlib

from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F

# from e4e.configs import data_configs, paths_config
# from e4e.datasets.inference_dataset import InferenceDataset
from .utils.model_utils import setup_model
# from e4e.utils.common import tensor2im
# from e4e.utils.alignment import align_face


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class EmptyArgs(object):
    def __init__(self):
        pass


class E4EBatchInfer(object):
    def __init__(self):
        self.device = "cuda:0"
        self.args = EmptyArgs
        self.args.latents_only = True
        self.args.align = False
        self.args.ckpt = make_abs_path("../../weights/e4e/e4e_ffhq_encode.pt")

        self.net, self.opts = setup_model(self.args.ckpt, self.device)
        self.is_cars = 'cars_' in self.opts.dataset_type
        self.generator = self.net.decoder
        self.generator.eval()
        print(f"[E4EBatchInfer] model loaded from {self.args.ckpt}.")

    @torch.no_grad()
    def infer_batch(self, x: torch.Tensor):
        x_down = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=True)
        codes = self.net.encoder(x_down)
        if self.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1, 1)
        if codes.shape[1] == 18 and self.is_cars:
            codes = codes[:, :16, :]
        return codes


if __name__ == "__main__":
    net = E4EBatchInfer()
    img = torch.randn(2, 3, 1024, 1024).cuda()
    latents = net.infer_batch(img)
    print(latents.shape)

