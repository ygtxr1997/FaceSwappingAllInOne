import os

from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision import transforms

from swapping import BlendSwap

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


if __name__ == "__main__":
    device = "cuda:0"
    weight_path = make_abs_path("checkpoints/blendswap.pth")
    target_path = make_abs_path("swapping/examples/target.png")
    source_path = make_abs_path("swapping/examples/source.png")

    model = BlendSwap()
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    model.to(device)

    target_img = transforms.ToTensor()(Image.open(target_path)).unsqueeze(0).to(device)
    source_img = transforms.ToTensor()(Image.open(source_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(target_img, source_img)
    print(output.shape)
