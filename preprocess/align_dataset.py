import os.path

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from datasets.ffplus import FFPlusDataset
from .align_tools import FaceAlignImageInfer


def align_ffplus(resume: int = -1):
    aligner = FaceAlignImageInfer()
    dataset = FFPlusDataset()
    out_root = dataset.img_folder.replace("youtube/c23/images", "youtube/c23/images_aligned")
    folders = dataset.id_folders
    img_paths = dataset.all_img_paths
    for i, folder in enumerate(tqdm(folders)):
        if i < resume:
            continue
        os.makedirs(os.path.join(out_root, folder), exist_ok=True)
        id_paths = img_paths[i]
        for path in id_paths:
            fn = os.path.basename(path)
            pil = Image.open(path).convert("RGB")
            cropped, _, _ = aligner.crop_ori_to_aligned(pil, align_source="ffhq", info_str=folder)
            cropped.save(os.path.join(out_root, folder, fn))
