import os

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision import transforms

from .alignment import (
    norm_crop,
    norm_crop_with_M,
    paste_back,
    save,
    get_5_from_98,
    get_lmk,
)
from .PIPNet.lib.tools import (
    get_lmk_model,
    demo_image,
)


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class FaceAlignImageInfer(object):
    def __init__(self):
        net, detector = get_lmk_model()
        self.net = net
        self.detector = detector

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def crop_ori_to_aligned(self, source: Image.Image, target: Image.Image = None,
                            align_source="arcface", align_target="ffhq", info_str: str = "",
                            ):
        net, detector = self.net, self.detector

        s_h, s_w = source.size
        if s_h < 500 or s_w < 500:
            source = source.resize((int(s_h * 1.5), int(s_w * 1.5)))
        source_img = np.array(source)
        lmk = get_5_from_98(demo_image(source_img, net, detector)[0])
        if lmk is not None:
            source_rgb = norm_crop(source_img, lmk, 256, mode=align_source, borderValue=0.0)
        else:
            print(f"[Warning] PIPNet no face found @{info_str}")
            source_rgb = np.array(source.resize((256, 256)))
        if align_source == "arcface":
            source_rgb = cv2.resize(source_rgb, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)

        if target is not None:
            target = np.array(target)
            original_target = target.copy()
            lmk = get_5_from_98(demo_image(target, net, detector)[0])
            target_rgb, target_affine_m = norm_crop_with_M(target, lmk, 256, mode=align_target, borderValue=0.0)

            return Image.fromarray(source_rgb), Image.fromarray(target_rgb), target_affine_m
        else:
            return Image.fromarray(source_rgb), None, None

    def paste_aligned_to_ori(self, swapped: Image.Image, target_ori: Image.Image,
                             target_affine_m: np.ndarray,
                             ):
        pasted_pil = paste_back(swapped, target_ori, target_affine_m)
        return pasted_pil
