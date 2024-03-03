import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FFPlusDataset(Dataset):
    def __init__(self,
                 root: str = "/media/yuange/EXTERNAL_USB/datasets/ffplus",
                 ts_list: np.ndarray = None,  # (N,2)
                 out_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 use_aligned: bool = False,
                 max_len: int = -1,
                 ):
        super(FFPlusDataset, self).__init__()
        self.root = root
        self.img_folder = os.path.join(self.root, "original_sequences/youtube/c23/images")
        if use_aligned:
            self.img_folder = self.img_folder.replace("youtube/c23/images", "youtube/c23/images_aligned")
        id_folders = os.listdir(self.img_folder)
        id_folders.sort()
        all_img_paths = []
        all_cnt = 0
        for folder in id_folders:
            id_img_paths = [os.path.join(self.img_folder, folder, fn) for fn in os.listdir(
                os.path.join(self.img_folder, folder))]
            all_img_paths.append(id_img_paths)
            all_cnt += len(id_img_paths)
        self.id_folders = id_folders
        self.all_img_paths = all_img_paths
        self.all_cnt = all_cnt
        print('FFPlus dataset loaded, total ids = %d, total imgs = %d'
              % (len(self.all_img_paths), self.all_cnt))

        self.out_size = (out_size, out_size)
        self.transform = transform

        if ts_list is None:
            default_ts_pairs = "ts_pairs.txt"
            with open(os.path.join(self.root, default_ts_pairs)) as f:
                lines = f.readlines()
            pairs = []
            for line in lines:
                p = line.strip().split(" ")
                p = [int(s) for s in p]
                pairs.append(p)
            ts_list = np.array(pairs)
        assert ts_list.shape[-1] == 2 and ts_list.ndim == 2
        self.ts_list = ts_list

        if max_len < 0:
            max_len = self.ts_list.shape[0]
        self.max_len = max_len

    def __getitem__(self, index):
        t_id = self.ts_list[index][0]
        s_id = self.ts_list[index][1]

        t_pil = Image.open(self.all_img_paths[t_id][0]).convert("RGB").resize(self.out_size)
        s_pil = Image.open(self.all_img_paths[s_id][0]).convert("RGB").resize(self.out_size)

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
