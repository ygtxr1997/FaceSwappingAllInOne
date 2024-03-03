import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

from .hopenet import Hopenet


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class HopeNetImageInfer(object):
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.size = (224, 224)
        model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        snapshot_path = make_abs_path('../weights/hopenet/hopenet_robust_alpha1.pkl')
        saved_state_dict = torch.load(snapshot_path)
        model.load_state_dict(saved_state_dict)
        model.eval()
        norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model = model.to(self.device)
        self.norm = norm

    @torch.no_grad()
    def infer_tensor(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if (h, w) != self.size:  # if x.size is not 224
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=True)  # resize to 224x224
        else:
            x = x
        if x.device != self.device:
            x = x.to(self.device)
        x = self.norm(x)
        yaw, pitch, roll = self.model(x)  # each is (B,66)
        hope_vec = torch.cat([yaw, pitch, roll], dim=-1)  # (B,198)
        return hope_vec

    def calc_l2(self,
                v1: torch.Tensor,
                v2: torch.Tensor,
                ):
        v1 = v1.cpu().numpy()
        v2 = v2.cpu().numpy()
        yaw_t, pitch_t, roll_t = self._extract_hopenet_yaw_pitch_roll(v1)  # each is (B,66)
        yaw_r, pitch_r, roll_r = self._extract_hopenet_yaw_pitch_roll(v2)  # each is (B,66)

        yaw_mse, yaw_l2, yaw_dim = self._calc_mse_l2(yaw_r, yaw_t)
        pitch_mse, pitch_l2, pitch_dim = self._calc_mse_l2(pitch_r, pitch_t)
        roll_mse, roll_l2, roll_dim = self._calc_mse_l2(roll_r, roll_t)
        mse = (yaw_mse + pitch_mse + roll_mse) / 3
        l2 = (yaw_l2 + pitch_l2 + roll_l2) / 3
        print('[hopenet] average MSE=%.3f, L2=%.3f, dim=%d' % (
            mse,
            l2,
            yaw_dim,
        ))
        return l2

    @staticmethod
    def _extract_hopenet_yaw_pitch_roll(param: np.ndarray):
        yaw, pitch, roll = param[:, :66], param[:, 66:-66], param[:, -66:]
        return yaw, pitch, roll  # (N,66), (N,66), (N,66)

    @staticmethod
    def _calc_mse_l2(vec1: np.ndarray, vec2: np.ndarray):
        mse_dist = mean_squared_error(vec1, vec2)
        ''' L2 = sqrt(MSE * dim) '''
        dim = vec1.shape[-1]
        l2_dist = np.sqrt(mse_dist * dim) / 2
        return mse_dist, l2_dist, dim
