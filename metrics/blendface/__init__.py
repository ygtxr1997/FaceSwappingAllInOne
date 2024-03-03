import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

from .iresnet import iresnet100


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class BlendFaceMetricImageInfer(object):
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.size = (112, 112)
        in_size = 512
        self.trans_matrix = torch.tensor(
            [[[1.07695457, -0.03625215, -1.56352194 / 512],
              [0.03625215, 1.07695457, -5.32134629 / 512]]],
            requires_grad=False,).float()

        self.model = iresnet100().to(self.device)
        weights = make_abs_path("../weights/BlendFace/blendface.pt")
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

    def infer_tensor(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if (h, w) != self.size:  # if x is ffhq aligned
            x = F.interpolate(x, size=512, mode="bilinear", align_corners=True)  # first resize to 512
            m = self.trans_matrix.repeat(b, 1, 1)  # to (B,2,3)
            grid = F.affine_grid(m, size=x.shape, align_corners=True)  # 得到 grid 用于 grid sample
            x = F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="zeros")  # warp affine
            x = F.interpolate(x, size=112, mode="bilinear", align_corners=True)  # resize to 112x112
        else:  # x is arcface aligned
            x = x
        if x.device != self.device:
            x = x.to(self.device)
        embeddings: torch.Tensor = self.model(x)
        return embeddings

    def calc_id_retrieval(self,
                          emb_result: torch.Tensor,
                          emb_source: torch.Tensor,
                          idx_source_gt: np.ndarray = None,
                          ):
        if idx_source_gt is None:
            idx_source_gt = np.arange(emb_source.shape[0])

        emb_result = emb_result.cpu().numpy()
        emb_source = emb_source.cpu().numpy()
        dists = pairwise_distances(emb_result, emb_source, metric='cosine')  # (dataset_len,dataset_len)
        idx_source_pred = dists.argmin(axis=1)  # (dataset_len,)
        cos_sims = (1 - dists)

        diff = np.zeros_like(idx_source_pred)
        ones = np.ones_like(idx_source_pred)
        diff[idx_source_pred != idx_source_gt] = ones[idx_source_pred != idx_source_gt]
        acc = 1. - diff.sum() / diff.shape[0]
        cos_sim_mean = cos_sims[idx_source_gt, idx_source_gt].mean()
        print('[BlendFace] id retrieval acc = %.2f %%, cosine_sim = %.4f' % (
            acc * 100., cos_sim_mean
        ))
        return acc, cos_sim_mean
