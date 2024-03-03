import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import cv2
from PIL import Image
import numpy as np

from .modules.networks.faceshifter import FSGenerator


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class FaceShifterImageInfer(object):
    def __init__(self):
        self.mouth_net_param = {
            "use": False
        }
        model = FSGenerator(
            make_abs_path("../weights/arcface/ms1mv3_arcface_r100_fp16/backbone.pth"),
            mouth_net_param=self.mouth_net_param
        )
        model = model.cuda()
        model = model.eval()

        self.weight_path = make_abs_path("../weights/ReliableSwap/faceshifter_v5.pt")
        model.load_state_dict(torch.load(self.weight_path, map_location="cpu"))

        self.model = model
        print("[FaceShifterImageInfer] model loaded.")

    @torch.no_grad()
    def infer_image(self, source: Image.Image, target: Image.Image, **kwargs):
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
        r = self.model.forward(s, t)[0]  # tuple

        result_pil = tensor_to_pil(r)
        return result_pil
