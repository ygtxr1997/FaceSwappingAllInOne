import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OutputDataset(Dataset):
    def __init__(self,
                 root: str = "/home/yuange/datasets/FaceSwapOutputs",
                 dataset_name: str = "celebahq",
                 method_name: str = None,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 max_len: int = -1,
                 ):
        super(OutputDataset, self).__init__()
        self.root = root
        self.dataset_name = dataset_name
        self.method_name = method_name if method_name is not None else "e4s2023final"  # load default folder
        self.img_root = os.path.join(self.root, self.dataset_name, self.method_name)
        self.target_key = "target"
        self.source_key = "source"

        img_fns = os.listdir(self.img_root)
        img_fns.sort()
        self.img_paths = [os.path.join(self.img_root, fn) for fn in img_fns]

        if max_len < 0:
            max_len = len(self.img_paths)
        self.max_len = min(max_len, len(self.img_paths))

        if method_name is not None:
            print(f'Output/{self.dataset_name}/{self.method_name} dataset loaded, total imgs = {len(self.img_paths)}')
        else:
            print(f'Output/{self.dataset_name}/source&target dataset loaded, total imgs = {len(self.img_paths)}')

        self.transform = transform

    def __getitem__(self, index):
        r_path = self.img_paths[index]
        t_path = r_path.replace(self.method_name, self.target_key)
        s_path = r_path.replace(self.method_name, self.source_key)

        r_pil = Image.open(r_path).convert("RGB")
        t_pil = Image.open(t_path).convert("RGB")
        s_pil = Image.open(s_path).convert("RGB")

        r_pil = r_pil.resize(s_pil.size)  # resize

        r_tensor = self.transform(r_pil)
        t_tensor = self.transform(t_pil)
        s_tensor = self.transform(s_pil)

        return {
            "result_tensor": r_tensor,
            "target_tensor": t_tensor,
            "source_tensor": s_tensor,
        }

    def __len__(self):
        return self.max_len
