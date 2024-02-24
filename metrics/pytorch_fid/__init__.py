__version__ = '0.2.1'

import os
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from . import fid_score


class FidFolderInfer(object):
    def __init__(self):
        pass

    def infer_folder(self,
                     folder1: str,
                     folder2: str,
                     batch_size=16, device=0, dims=2048, num_workers=4,
                     max_num1=30000, max_num2=30000,
                     ):
        fid = fid_score.calculate_fid_given_paths(
            [folder1, folder2],
            batch_size=batch_size, device=device, dims=dims, num_workers=num_workers,
            max_nums=[max_num1, max_num2]
        )
        print('FID = %.2f' % fid)
        return fid


if __name__ == "__main__":
    folderA = '/gavin/code/FaceSwapping/inference/ffplus/demo_triplet10w_38/source'
    # folderB = '/gavin/datasets/stylegan/stylegan3-r-ffhq-1024x1024'
    folderB = os.path.join('/gavin/code/TextualInversion/exp_eval/db',
                           'all')

    val = fid_score.calculate_fid_given_paths([folderA, folderB], batch_size=16, device=0, dims=2048, num_workers=4)
    print('FID = %.2f' % val)
