import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CelebAHQDataset(Dataset):
    def __init__(self,
                 root: str = "/home/yuange/datasets/CelebAMask-HQ/CelebA-HQ-img",
                 ts_list: np.ndarray = None,  # (N,2)
                 out_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 max_len: int = -1,
                 ):
        super(CelebAHQDataset, self).__init__()
        self.root = root
        img_fns = os.listdir(self.root)
        img_fns.sort()
        self.img_paths = [os.path.join(self.root, fn) for fn in img_fns]
        print('Celeba-HQ dataset loaded, total imgs = %d'
              % (len(self.img_paths)))

        self.out_size = (out_size, out_size)
        self.transform = transform

        if ts_list is None:
            total_cnt = len(self.img_paths)
            t_list = np.arange(total_cnt)
            s_list = (np.arange(total_cnt) + total_cnt // 2) % total_cnt
            ts_list = np.zeros((total_cnt, 2))
            ts_list[:, 0] = t_list
            ts_list[:, 1] = s_list
        ts_list = ts_list.astype(np.int64)
        assert ts_list.shape[-1] == 2 and ts_list.ndim == 2
        self.ts_list = ts_list

        if max_len < 0:
            max_len = self.ts_list.shape[0]
        self.max_len = max_len

    def __getitem__(self, index):
        t_id = self.ts_list[index][0]
        s_id = self.ts_list[index][1]

        t_pil = Image.open(self.img_paths[t_id]).convert("RGB").resize(self.out_size)
        s_pil = Image.open(self.img_paths[s_id]).convert("RGB").resize(self.out_size)

        t_tensor = self.transform(t_pil)
        s_tensor = self.transform(s_pil)

        return {
            "target_arr": np.array(t_pil).astype(np.uint8),
            "source_arr": np.array(s_pil).astype(np.uint8),
            "target_tensor": t_tensor,
            "source_tensor": s_tensor,
            "target_id": t_id,
            "source_id": s_id,
        }

    def __len__(self):
        return self.max_len
