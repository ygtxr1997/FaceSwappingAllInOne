import os
import cv2
import time
import argparse
import logging
import json
from datetime import datetime, timedelta

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from .stylefusion.sf_stylegan2 import SFGenerator
from .stylefusion.sf_hierarchy import SFHierarchyFFHQ
from .e4e.batch_infer import E4EBatchInfer


make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


class StyleFusionImageInfer(object):
    def __init__(self, latents_type: str = "w+"):
        stylegan_type = "ffhq"
        self.device = 'cuda:0'
        self.latents_type = latents_type

        self.stylegan_type = stylegan_type
        if self.stylegan_type == "ffhq":
            self.truncation = 0.7
            self.stylegan_size = 1024
            self.stylegan_layers = 18
        self.stylegan_ckpt_path = make_abs_path(
            "../weights/MegaFS/inference/checkpoint/stylegan2-ffhq-config-f.pth")
        self.original_net, self.mean_latent, self.sf_hierarchy, self.base_blender = self.load_model(
            self.stylegan_ckpt_path
        )

        fusion_nets_weights = make_abs_path("weights/ffhq_weights_modified.json")
        with open(fusion_nets_weights, 'r') as f:
            fusion_nets_paths = json.load(f)
        keys = fusion_nets_paths.keys()

        for key in keys:
            val = fusion_nets_paths[key]
            val = os.path.join(os.path.abspath(make_abs_path("../weights/")), val)
            self.sf_hierarchy.nodes[key].load_fusion_net(val)
            self.sf_hierarchy.nodes[key].fusion_net.to(self.device)
            self.sf_hierarchy.nodes[key].fusion_net.eval()

        self.e4e_infer = E4EBatchInfer()
        print(f"[StyleFusionInfer] model loaded from: {self.stylegan_ckpt_path} and {fusion_nets_weights}.")

    def load_model(self, ckpt_path: str):
        stylegan_ckpt = torch.load(ckpt_path, map_location='cpu')
        original_net = SFGenerator(self.stylegan_size, 512, 8)
        original_net.load_state_dict(stylegan_ckpt['g_ema'], strict=True)
        original_net = original_net.to(self.device)
        with torch.no_grad():
            mean_latent = original_net.mean_latent(4096)
        if self.stylegan_type == "ffhq":
            sf_hierarchy = SFHierarchyFFHQ()
            base_blender = sf_hierarchy.nodes["all"]
        return original_net, mean_latent, sf_hierarchy, base_blender

    @torch.no_grad()
    def infer_batch(self, source_tensor: torch.Tensor, target_tensor: torch.Tensor,
                    ):
        assert source_tensor.shape[0] == 1, "batch size should be 1"
        s_latents = self.e4e_infer.infer_batch(source_tensor)
        t_latents = self.e4e_infer.infer_batch(target_tensor)
        latents_type = self.latents_type

        s_dict = dict()
        parts = self.sf_hierarchy.nodes["all"].get_all_active_parts()
        for part in parts:
            s_dict[part] = self.general_latent_to_s(t_latents, latents_type)  # target: pose, expression

        hair = t_latents
        face = s_latents
        background = t_latents
        mouth = s_latents
        eyes = s_latents
        overall = t_latents

        self.swap(s_dict, hair, ["bg_hair_clothes", "hair"])  # target
        self.swap(s_dict, face, ["face", "eyes", "skin_mouth", "mouth", "skin", "shirt"])  # source
        self.swap(s_dict, background, ["background", "background_top", "background_bottom", "bg"])  # target
        # self.swap(s_dict, overall, ["all"])  # target
        self.swap(s_dict, mouth, ["skin_mouth", "face"])  # source
        self.swap(s_dict, eyes, ["eyes", "face"])  # source

        return self.s_dict_to_image(s_dict)

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

    def swap(self, s_dict, value, keys):
        if value is None:
            return
        for k in keys:
            s_dict[k] = self.general_latent_to_s(value, self.latents_type)
        return s_dict

    def general_latent_to_s(self, l, latent_type):
        assert latent_type in ["z", "w", "w+", "s"]

        if latent_type == "z":
            assert l.size() == (1, 512)
            return self.z_to_s(l)
        elif latent_type == "w" or latent_type == "w+":
            assert l.size() == (1, 512) or l.size() == (1, self.stylegan_layers, 512)
            if l.dim() == 2:
                return self.w_plus_to_s(l.unsqueeze(0).repeat(1, self.stylegan_layers, 1), truncation=1)
            else:
                return self.w_plus_to_s(l, truncation=1)
        else:
            return l

    def z_to_s(self, z):
        return self.original_net([z],
                                     truncation=self.truncation, truncation_latent=self.mean_latent,
                                     randomize_noise=False, return_style_vector=True)

    def z_to_w_plus(self, z):
        _, res = self.original_net([z],
                                     truncation=self.truncation, truncation_latent=self.mean_latent,
                                     randomize_noise=False, return_latents=True)
        return res[0]

    def w_plus_to_s(self, w_plus, truncation):
        return self.original_net([w_plus], input_is_latent=True,
                                     truncation=truncation, truncation_latent=self.mean_latent,
                                     randomize_noise=False, return_style_vector=True)

    def s_to_image(self, s):
        img, _ = self.original_net([torch.zeros(1, 512, device=self.device)],
                                                          randomize_noise=False, style_vector=s)
        return img

    def w_plus_to_image(self, w_plus):
        s = self.w_plus_to_s(w_plus, truncation=1)
        return self.s_to_image(s)

    def z_to_image(self, z):
        s = self.z_to_s(z)
        return self.s_to_image(s)

    def s_dict_to_image(self, s_dict):
        s = self.base_blender.forward(s_dict)
        return self.s_to_image(s)

    def w_plus_dict_to_image(self, w_plus_dict, truncation=1):
        s_dict = dict()
        for key in w_plus_dict.keys():
            s_dict[key] = self.w_plus_to_s(w_plus_dict[key], truncation=truncation)
        return self.s_dict_to_image(s_dict)

    def z_dict_to_image(self, z_dict):
        s_dict = dict()
        for key in z_dict.keys():
            s_dict[key] = self.z_to_s(z_dict[key])
        return self.s_dict_to_image(s_dict)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    s_img = Image.open('tmp_source.jpg')
    t_img = Image.open('tmp_target.jpg')
    TRANSFORMS = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    s_tensor = TRANSFORMS(s_img)[None, :, :, :].cuda()
    t_tensor = TRANSFORMS(t_img)[None, :, :, :].cuda()

    stylefusion_model = StyleFusionInfer(latents_type="w+")
    res = stylefusion_model.infer_batch(s_tensor, t_tensor)
    print(type(res), res.shape)

    # import thop
    #
    # flops, params = thop.profile(stylefusion_model, inputs=(s_tensor, t_tensor,), verbose=False)
    # print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))
