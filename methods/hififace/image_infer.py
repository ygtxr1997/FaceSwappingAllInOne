import os
import math
import argparse
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from .hififace_pl import HifiFace


make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


class HiFiFaceImageInfer(object):
    def __init__(self):
        self.device = "cuda:0"
        self.model_config = make_abs_path("./config/model.yaml")
        self.model_ckpt_path = make_abs_path("../weights/hififace/hififace_opensouce_299999.ckpt")
        ckpt = torch.load(self.model_ckpt_path, map_location="cpu")
        self.net = HifiFace(OmegaConf.load(self.model_config))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.net.to(self.device)

    @torch.no_grad()
    def infer_batch(self, source_batch, target_batch):
        ori_hw = source_batch.shape[2:]
        source_batch = (source_batch * 0.5 + 0.5).clamp(0, 1)
        target_batch = (target_batch * 0.5 + 0.5).clamp(0, 1)
        source_batch = F.interpolate(source_batch, size=(256, 256), mode="bilinear", align_corners=True)
        target_batch = F.interpolate(target_batch, size=(256, 256), mode="bilinear", align_corners=True)
        res = self.net(source_batch, target_batch)
        res = F.interpolate(res, size=ori_hw, mode="bilinear", align_corners=True)
        res = (res * 2. - 1.).clamp(-1, 1)
        return res

    @torch.no_grad()
    def infer_image(self, source, target, **kwargs):
        def pil_to_tensor(x: Image.Image):
            x = np.array(x.resize((256, 256))).astype(np.float32) / 127.5 - 1.
            x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).cuda()
            return x

        def tensor_to_pil(x: torch.Tensor):
            x = (x[0].permute(1, 2, 0).cpu().numpy() + 1.) * 127.5
            x = Image.fromarray(x.astype(np.uint8))
            return x

        s = pil_to_tensor(source)
        t = pil_to_tensor(target)
        r = self.infer_batch(s, t)

        result_pil = tensor_to_pil(r)
        return result_pil
