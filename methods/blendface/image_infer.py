import os

from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision import transforms

from swapping import BlendSwap
from align_tools import FaceAlignImageInfer


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

    def infer_image(self, source: Image.Image, target: Image.Image):
        target_img = transforms.ToTensor()(target).unsqueeze(0).to(self.device)
        source_img = transforms.ToTensor()(source).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(target_img, source_img)
        print(output.shape)
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
