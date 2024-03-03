import os

from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F

from .swapping import BlendSwap
from .align_tools import FaceAlignImageInfer


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class BlendFaceImageInfer(object):
    def __init__(self):
        device = "cuda:0"
        weight_path = make_abs_path("checkpoints/blendswap.pth")
        model = BlendSwap()
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        model.eval()
        model.to(device)

        self.device = device
        self.model = model

        self.source_size = (112, 112)
        in_size = 512
        self.trans_matrix = torch.tensor(
            [[[1.07695457, -0.03625215, -1.56352194 / 512],
              [0.03625215, 1.07695457, -5.32134629 / 512]]],
            requires_grad=False, ).float().to(self.device)

    @torch.no_grad()
    def infer_image(self, source: Image.Image, target: Image.Image, **kwargs):
        target_img = transforms.ToTensor()(target).unsqueeze(0).to(self.device)
        source_img = transforms.ToTensor()(source).unsqueeze(0).to(self.device)

        x = source_img
        b, c, h, w = x.shape
        if (h, w) != self.source_size:  # if x is ffhq aligned
            x = F.interpolate(x, size=512, mode="bilinear", align_corners=True)  # first resize to 512
            m = self.trans_matrix.repeat(b, 1, 1)  # to (B,2,3)
            grid = F.affine_grid(m, size=x.shape, align_corners=True)  # 得到 grid 用于 grid sample
            x = F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="zeros")  # warp affine
            x = F.interpolate(x, size=112, mode="bilinear", align_corners=True)  # resize to 112x112
        else:  # x is arcface aligned
            x = x

        y = F.interpolate(target_img, size=256, mode="bilinear", align_corners=True)  # resize to 112x112

        with torch.no_grad():
            output = self.model(y, x)
        swapped = Image.fromarray(
            (output.permute(0, 2, 3, 1)[0].cpu().data.numpy() * 255).astype(np.uint8)
        )
        return swapped


if __name__ == "__main__":
    target_path = make_abs_path("swapping/examples/target.png")
    source_path = make_abs_path("swapping/examples/source.png")

    target_pil = Image.open(target_path)
    source_pil = Image.open(source_path)

    aligner = FaceAlignImageInfer()
    s_aligned, t_aligned, t_affine = aligner.crop_ori_to_aligned(
        source_pil, target_pil
    )
    s_aligned.save("tmp_s_aligned.png")
    t_aligned.save("tmp_t_aligned.png")

    blend_face = BlendFaceImageInfer()
    swapped = blend_face.infer_image(
        s_aligned, t_aligned
    )

    pasted = aligner.paste_aligned_to_ori(
        swapped, target_pil, t_affine
    )
    pasted.save("tmp_pasted.png")
