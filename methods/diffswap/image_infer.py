import dlib
import matplotlib.pyplot as plt
import matplotlib
import cv2
import pickle
import os
import json
import torch
import re
import copy
import time
import numpy as np
import scipy
import random
from omegaconf import OmegaConf
from PIL import Image, ImageChops
from tqdm import tqdm
from einops import rearrange, repeat
from skimage import io
from imutils import face_utils
from torchvision import transforms

from einops import rearrange
from scipy.spatial import ConvexHull

from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

from mtcnn import MTCNN

from .data_preprocessing.align.align_trans import get_reference_facial_points, warp_and_crop_face
from .utils.portrait import Portrait
from .utils.blending.blending_mask import gaussian_pyramid, laplacian_pyramid, laplacian_pyr_join, laplacian_collapse
from . import ldm
from .ldm.util import instantiate_from_config
from .ldm.data.portrait import Portrait
from .ldm.models.diffusion.ddim import DDIMSampler


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class DiffSwapImageInfer(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(
            make_abs_path('checkpoints/shape_predictor_68_face_landmarks.dat')
        )
        self.mtcnn_detector = MTCNN()

        self.device = "cuda:0"
        self.config_path = make_abs_path("configs/diffswap/default-project.yaml")
        self.config = OmegaConf.load(self.config_path)
        self.checkpoint = make_abs_path("checkpoints/diffswap.pth")

        self.model = instantiate_from_config(self.config.model)
        self.model.init_from_ckpt(self.checkpoint)
        self.ckpt = os.path.basename(self.checkpoint).rsplit('.', 1)[0]
        self.model.eval()
        self.model = self.model.to(self.device)
        print('[DiffSwapImageInfer] Model built.')
        self.model.cond_stage_model.affine_crop = True
        self.model.cond_stage_model.swap = True

        self.tgt_scale = 0.01
        self.ddim_sampler = DDIMSampler(self.model, tgt_scale=self.tgt_scale)

    def pil_to_bgr(self, pil: Image.Image) -> np.ndarray:
        """
        pil: data/portrait_jpg/source
        bgr: data/portrait/source
        """
        rgb = np.array(pil).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def get_detect(self, image, iter):
        for i in range(iter + 1):  # the bigger, the slower
            faces = self.detector(image, i)
            if len(faces) >= 1:
                break
        return faces

    def get_landmark_ori(self, bgr: np.ndarray):
        """
        bgr: data/portrait/source
        bgr_resized: data/portrait/source
        landmark: data/portrait/landmark/landmark_ori.pkl
        """
        image = bgr
        while image.shape[0] > 2000 or image.shape[1] > 2000:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        while image.shape[0] < 400 or image.shape[1] < 400:
            image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.get_detect(imgray, 2)
        if len(faces) == 0:
            print("No face found!")
            return image, np.random.randn(68, 2).astype(np.float32)
        if len(faces) > 1:
            print("[Warning] More than one face found!")
            face = dlib.rectangle(faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom())
            for i in range(1, len(faces)):
                if abs(faces[i].right() - faces[i].left()) * abs(faces[i].top() - faces[i].bottom()) > abs(
                        face.right() - face.left()) * abs(face.top() - face.bottom()):
                    face = dlib.rectangle(faces[i].left(), faces[i].top(), faces[i].right(), faces[i].bottom())
        else:
            face = faces[0]
        landmark = self.landmark_predictor(image, face)
        landmark = face_utils.shape_to_np(landmark)

        bgr_resized = image
        return bgr_resized, landmark

    def crop_ffhq(self,
                  bgr_resized: np.ndarray,
                  landmark: np.ndarray,
                  output_size=256, transform_size=1024, enable_padding=False, rotate_level=True, random_shift=0,
                  retry_crops=False):
        """
        bgr_resized: data/portrait/source
        landmark: data/portrait/landmark/landmark_ori.pkl
        pil_aligned: data/portrait/align
        affine_rev: data/portrait/affines.json
        """
        print('Recreating aligned images...')
        # Fix random seed for reproducibility
        np.random.seed(12345)

        img_count = 0

        # lm = landmarks[type][img]
        lm = landmark
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        if rotate_level:
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1
        else:
            x = np.array([1, 0], dtype=np.float64)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1

        # Load in ori image.
        # src_file = os.path.join(data_path, type, img)
        # image = Image.open(src_file)
        image = Image.fromarray(cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB))
        quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
        qsize = np.hypot(*x) * 2

        # Keep drawing new random crop offsets until we find one that is contained in the image
        # and does not require padding
        if random_shift != 0:
            for _ in range(1000):
                # Offset the crop rectange center by a random shift proportional to image dimension
                # and the requested standard deviation
                c = (c0 + np.hypot(*x) * 2 * random_shift * np.random.normal(0, 1, c0.shape))
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                crop = (
                int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
                if not retry_crops or not (
                        crop[0] < 0 or crop[1] < 0 or crop[2] >= image.width or crop[3] >= image.height):
                    # We're happy with this crop (either it fits within the image, or retries are disabled)
                    break
            else:
                # rejected N times, give up and move to next image
                # (does not happen in practice with the FFHQ data)
                print('rejected image')
                return
        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
            # print(f'first opretion: resize, from {image.size} to {rsize}')
            image = image.resize(rsize, Image.BICUBIC)
            quad /= shrink
            qsize /= shrink
        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, image.size[0]),
                min(crop[3] + border, image.size[1]))
        IsCrop = False
        if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
            IsCrop = True
            crop = tuple(map(round, crop))
            # print(f'second operation: crop, {crop}')
            image = image.crop(crop)  # (left, upper, right, lower)
            # location = [crop[0], crop[1], crop[2], crop[3]]
            quad -= crop[0:2]
        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.size[0] + border, 0),
               max(pad[3] - image.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            image = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = image.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0,
                                                                                               0.0, 1.0)
            image += (np.median(image, axis=(0, 1)) - image) * np.clip(mask, 0.0, 1.0)
            image = Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform(with rotation)
        quad = (quad + 0.5).flatten()
        assert (abs((quad[2] - quad[0]) - (quad[4] - quad[6])) < 1e-6 and abs(
            (quad[3] - quad[1]) - (quad[5] - quad[7])) < 1e-6)

        if IsCrop:
            quad_new = [quad[0] + crop[0], quad[1] + crop[1], quad[2] + crop[0], quad[3] + crop[1],
                        quad[4] + crop[0], quad[5] + crop[1], quad[6] + crop[0], quad[7] + crop[1]]
        else:
            quad_new = quad
        if shrink > 1:
            quad_new *= shrink
        # print(f'quad_new: {quad_new}', 'type', type, 'img', img)
        affine_rev = ((256 * (quad_new[1] - quad_new[3])) / (
                    quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] * quad_new[5] +
                    quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] - quad_new[3] * quad_new[4]),
                      -(256 * (quad_new[0] - quad_new[2])) / (
                                  quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] *
                                  quad_new[5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] -
                                  quad_new[3] * quad_new[4]),
                      (256 * (quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2])) / (
                                  quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] *
                                  quad_new[5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] -
                                  quad_new[3] * quad_new[4]),
                      -(256 * (quad_new[3] - quad_new[5])) / (
                                  quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] *
                                  quad_new[5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] -
                                  quad_new[3] * quad_new[4]),
                      (256 * (quad_new[2] - quad_new[4])) / (
                                  quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] *
                                  quad_new[5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] -
                                  quad_new[3] * quad_new[4]),
                      (256 * (quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] * quad_new[
                          5] + quad_new[1] * quad_new[4])) / (
                                  quad_new[0] * quad_new[3] - quad_new[1] * quad_new[2] - quad_new[0] *
                                  quad_new[5] + quad_new[1] * quad_new[4] + quad_new[2] * quad_new[5] -
                                  quad_new[3] * quad_new[4]))
        # use affine to transform image
        affine = (-(quad[0] - quad[6]) / transform_size, -(quad[0] - quad[2]) / transform_size, quad[0],
                  -(quad[1] - quad[7]) / transform_size, -(quad[1] - quad[3]) / transform_size, quad[1])
        image = image.transform((transform_size, transform_size), Image.AFFINE, affine,
                                Image.BICUBIC)  # a, b, c, d, e, f

        if output_size < transform_size:
            image = image.resize((output_size, output_size), Image.BICUBIC)

        # Save aligned image.
        # dst_subdir = os.path.join(save_path, type)
        # os.makedirs(dst_subdir, exist_ok=True)
        # image.save(os.path.join(dst_subdir, img))
        pil_aligned = image

        img_count += 1
        # print('type {} finished, processed {} images'.format(type, img_count))
        # All done.
        # json.dump(affine_all, open(affine_path, 'w'), indent=4)

        return pil_aligned, affine_rev

    def get_lmk_256(self,
                    pil_aligned: Image.Image,
                    ):
        """
        pil_aligned: data/portrait/align
        landmark: data/portrait/landmark/landmark_256.pkl
        """
        image = cv2.cvtColor(np.array(pil_aligned), cv2.COLOR_RGB2BGR)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.get_detect(imgray, 2)
        if len(faces) == 0:
            print("No face found!")
            return np.random.randn(68, 2).astype(np.float32)
        if len(faces) > 1:
            print("[Warning] More than one face found!")
            face = dlib.rectangle(faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom())
            for i in range(1, len(faces)):
                if abs(faces[i].right() - faces[i].left()) * abs(faces[i].top() - faces[i].bottom()) > abs(
                        face.right() - face.left()) * abs(face.top() - face.bottom()):
                    face = dlib.rectangle(faces[i].left(), faces[i].top(), faces[i].right(), faces[i].bottom())
        else:
            face = faces[0]
        landmark = self.landmark_predictor(image, face)
        landmark = face_utils.shape_to_np(landmark)

        return landmark

    def detect_faces_portrait(self, aligned_pil: Image.Image):
        """
        aligned_pil: data/portrait/align
        mtcnn_result: data/portrait/mtcnn/mtcnn_gpu.json or mtcnn_256.json
        """
        rgb = np.array(aligned_pil)
        mtcnn_result = self.mtcnn_detector.detect_faces(rgb)
        return mtcnn_result

    def face_align_portrait(self, aligned_pil: Image.Image,
                            mtcnn_256: list,
                            crop_size=112,
                            ):
        """
        aligned_pil: data/portrait/align
        mtcnn_256: data/portrait/mtcnn/mtcnn_256.json
        pil_warped: data/portrait/align112x112/*.png
        affine_theta: data/portrait/affine_theta.json
        """
        scale = crop_size / 112.
        reference = get_reference_facial_points(default_square=True) * scale

        keys = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
        default_tfm = np.array([[2.90431126e-01, 1.89934467e-03, -1.88962605e+01],
                                [-1.90354592e-03, 2.90477119e-01, -1.70081139e+01]])

        H1 = W1 = 256
        H2 = W2 = 112
        A = np.array([[2 / (W1 - 1), 0, -1], [0, 2 / (H1 - 1), -1], [0, 0, 1]])
        B = np.linalg.inv(np.array([[2 / (W2 - 1), 0, -1], [0, 2 / (H2 - 1), -1], [0, 0, 1]]))
        C = np.array([[0, 0, 1]])

        def tfm2theta(tfm):
            ttt = np.concatenate([tfm, C], axis=0)
            ttt = np.linalg.inv(ttt)
            theta = A @ ttt @ B
            return theta[:2]

        def compute_area(item):
            return -np.prod(item['box'][-2:])

        use_torch = False
        to_tensor = ToTensor()
        to_pil = ToPILImage()

        value = sorted(mtcnn_256, key=compute_area)
        if len(value) == 0:
            print("[] face_align_portrait: No face found!")
            return aligned_pil.resize((112, 112)), np.random.randn(2, 3).astype(np.float32)
        value = value[0]
        facial5points = [value['keypoints'][key] for key in keys]
        tfm = warp_and_crop_face(None, facial5points, reference, crop_size=(crop_size, crop_size), return_tfm=True)

        theta = tfm2theta(tfm).tolist()

        image = aligned_pil
        if not use_torch:
            face_img = cv2.warpAffine(np.array(image), tfm, (crop_size, crop_size))
            pil_warped = Image.fromarray(face_img)
        else:
            image = to_tensor(image)[None].float()
            image = torch.nn.functional.interpolate(image, size=(256, 256))
            theta = torch.tensor(tfm2theta(tfm)[None]).float()
            grid = torch.nn.functional.affine_grid(theta, size=(1, 3, crop_size, crop_size))
            image = torch.nn.functional.grid_sample(image, grid)
            pil_warped = to_pil(image[0])

        return pil_warped, theta

    def get_mask(self):
        pass

    def get_batch(self, s_aligned: Image.Image, source_affine_theta: list, s_landmark_256: np.ndarray,
                  t_aligned: Image.Image, target_affine_theta: list, t_landmark_256: np.ndarray,
                  size=256, base_res=256, flip=False,
                  interpolation="bicubic", dilate=False, convex_hull=True,
                  ):
        """
        s_aligned: data/portrait/source
        source_affine_theta: data/portrait/affine_theta.json
        s_landmark_256: data/portrait/landmark/landmark_256.pkl
        """
        # 1. init
        batch = {}
        interpolation = {
            # "linear": Image.LINEAR,  # deprecated
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]
        all_indices = np.arange(0, 68)
        landmark_indices = {
            # 'face': all_indices[:17].tolist() + all_indices[17:27].tolist(),
            'l_eye': all_indices[36:42].tolist(),
            'r_eye': all_indices[42:48].tolist(),
            'nose': all_indices[27:36].tolist(),
            'mouth': all_indices[48:68].tolist(),
        }
        dilate_kernel = np.ones((11, 11), np.uint8)

        def extract_convex_hull(landmark):
            landmark = landmark * size
            hull = ConvexHull(landmark)
            image = np.zeros((size, size))
            points = [landmark[hull.vertices, :1], landmark[hull.vertices, 1:]]
            points = np.concatenate(points, axis=-1).astype('int32')
            mask = cv2.fillPoly(image, pts=[points], color=(255, 255, 255))
            mask = mask > 0
            return mask

        def extract_convex_hulls(landmark):
            mask_dict = {}
            mask_organ = []
            for key, indices in landmark_indices.items():
                mask_key = extract_convex_hull(landmark[indices])
                if dilate:
                    # mask_key = mask_key[:, :, None]
                    # mask_key = repeat(mask_key, 'h w -> h w k', k=3)
                    # print(mask_key.shape, type(mask_key))
                    mask_key = mask_key.astype(np.uint8)
                    mask_key = cv2.dilate(mask_key, dilate_kernel, iterations=1)
                mask_organ.append(mask_key)
            mask_organ = np.stack(mask_organ)  # (4, 256, 256)
            mask_dict['mask_organ'] = mask_organ
            mask_dict['mask'] = extract_convex_hull(landmark)
            return mask_dict

        def mask_organ_src(landmark):
            mask_organ = []
            for key, indices in landmark_indices.items():
                mask_key = extract_convex_hull(landmark[indices])
                if dilate:
                    # mask_key = mask_key[:, :, None]
                    # mask_key = repeat(mask_key, 'h w -> h w k', k=3)
                    # print(mask_key.shape, type(mask_key))
                    mask_key = mask_key.astype(np.uint8)
                    mask_key = cv2.dilate(mask_key, dilate_kernel, iterations=1)
                mask_organ.append(mask_key)
            return np.stack(mask_organ)

        # 2. source
        # image = Image.open(os.path.join(f'{self.root}/align/{type}', self.src_list[src_index])).convert('RGB')
        # affine_theta = np.array(self.affine_thetas[type][self.src_list[src_index]], dtype=np.float32)
        # landmark = torch.tensor(self.landmarks[type][self.src_list[src_index]]) / self.base_res
        image = s_aligned
        affine_theta = source_affine_theta
        landmark = torch.tensor(s_landmark_256) / base_res

        image = image.resize((size, size), resample=interpolation)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        batch['mask_organ_src'] = mask_organ_src(landmark)  # (4,256,256)
        batch['image_src'] = image  # (256,256,3)
        # batch['image_src']'s identity
        batch['affine_theta_src'] = affine_theta  # Tuple(6)
        batch['src'] = "source_fn.jpg"

        # 3. target
        # image = Image.open(os.path.join(f'{self.root}/align/{type}', self.tgt_list[tgt_index])).convert('RGB')
        # affine_theta = np.array(self.affine_thetas[type][self.tgt_list[tgt_index]], dtype=np.float32)
        # landmark = torch.tensor(self.landmarks[type][self.tgt_list[tgt_index]]) / self.base_res
        image = t_aligned
        affine_theta = target_affine_theta
        landmark = torch.tensor(t_landmark_256) / base_res

        image = image.resize((size, size), resample=interpolation)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        batch['image'] = image
        # batch['image']'s identity
        batch['landmark'] = landmark  # (68,2)
        batch['affine_theta'] = affine_theta
        batch['target'] = "target_fn.jpg"

        # 4. mask: data/portrait/mask
        if convex_hull:
            mask_dict = extract_convex_hulls(batch['landmark'])
            batch.update(mask_dict)  # mask:(256,256), mask_organ:(4,256,256)

        # 5. add batch dim
        for k, v in batch.items():
            if isinstance(v, str):
                batch[k] = [v]
            elif isinstance(v, np.ndarray):
                batch[k] = v[np.newaxis, :]
                batch[k] = torch.tensor(batch[k], device=self.device)
            elif isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
                batch[k] = torch.tensor(batch[k], device=self.device)
            elif isinstance(v, tuple) or isinstance(v, list):
                batch[k] = np.array(v)[np.newaxis, :]
                batch[k] = torch.tensor(batch[k], device=self.device).float()
            else:
                raise TypeError("Not supported type!")
        # for k, v in batch.items():
        #     print(f"{k}:{type(v)}")
        #     if isinstance(v, tuple) or isinstance(v, list):
        #         print(v)
        #     elif isinstance(v, torch.Tensor):
        #         print(v.shape, v.device)
        # exit()

        return batch

    @torch.no_grad()
    def perform_swap(self, batch, ckpt, ddim_sampler=None, ddim_steps=200, ddim_eta=0., **kwargs):
        """
        swapped_pil: data/portrait/swap_res_ori
        mask_pil: data/portrait/mask
        """
        # now we swap the faces
        use_ddim = ddim_steps is not None

        log = dict()

        model = self.model

        z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True, swap=True)
        N = x.size(0)

        b, h, w = z.shape[0], z.shape[2], z.shape[3]  # 64 x 64

        mask_key = 'mask'
        mask = (1 - batch[mask_key].float())[:, None]
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask[mask > 0] = 1
        mask[mask <= 0] = 0

        with model.ema_scope("Plotting Inpaint"):
            if ddim_sampler is None:
                ddim_sampler = DDIMSampler(model)
            shape = (model.channels, model.image_size, model.image_size)
            samples, _ = ddim_sampler.sample(ddim_steps, N, shape, c, eta=ddim_eta, x0=z[:N], mask=mask,
                                             verbose=False, **kwargs)

        x_samples = model.decode_first_stage(samples.to(model.device))

        gen_imgs = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        gen_imgs = np.array(gen_imgs * 255).astype(np.uint8)
        gen_imgs = rearrange(gen_imgs, 'b c h w -> b h w c')

        # save swapped images
        # for j in range(N):
        #     src = batch['src'][j][:-4]
        #     save_root = f'swap_res/{ckpt}_{ddim_sampler.tgt_scale}'
        #     os.makedirs(os.path.join(save_dir, save_root, src), exist_ok=True)
        #     Image.fromarray(gen_imgs[j]).save(os.path.join(save_dir, save_root, src, batch['target'][j]))

        swapped_pil = Image.fromarray(gen_imgs[0])
        mask_pil = Image.fromarray(np.array(batch[mask_key][0].cpu() * 255).astype(np.uint8))
        return swapped_pil, mask_pil

    def repair_by_mask(self, swapped_pil: Image.Image,
                       target_aligned_pil: Image.Image, mask_pil: Image.Image,
                       ):
        """
        swapped_pil: data/portrait/swap_res
        target_aligned_pil: data/portrait/align/target
        mask_pil: data/portrait/mask
        repaired_pil: data/portrait/swap_res_repair
        """
        swap_img = cv2.cvtColor(np.array(swapped_pil), cv2.COLOR_RGB2BGR)
        im1 = cv2.cvtColor(swap_img, cv2.COLOR_BGR2RGB)
        tgt_img = cv2.cvtColor(np.array(target_aligned_pil), cv2.COLOR_RGB2BGR)
        im2 = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
        mask = mask_pil
        im1, im2 = np.int32(im1), np.int32(im2)
        mask = np.uint8(np.array(mask).astype(np.float32) / 255.)

        gp_1, gp_2 = [gaussian_pyramid(im) for im in [im1, im2]]
        mask_gp = [cv2.resize(mask, (gp.shape[1], gp.shape[0])) for gp in gp_1]
        lp_1, lp_2 = [laplacian_pyramid(gp) for gp in [gp_1, gp_2]]
        lp_join = laplacian_pyr_join(lp_1, lp_2, mask_gp)
        im_join = laplacian_collapse(lp_join)
        np.clip(im_join, 0, 255, out=im_join)
        im_join = np.uint8(im_join)

        repaired_pil = Image.fromarray(im_join)
        return repaired_pil

    def paste(self, repaired_pil: Image.Image,
              t_ori_pil: Image.Image, t_affine: list,
              ):
        """
        repaired_pil: data/portrait/swap_res_repair
        t_ori_pil: data/portrait/target
        t_affine: data/portrait/affines.json
        pasted_pil: data/portrait/swap_res_ori
        """
        tgt_img = t_ori_pil
        gen_img = tgt_img.copy()
        gen_img256 = repaired_pil  # 256x256
        mask = Image.new('RGBA', (256, 256), (255, 255, 255))
        mask = mask.transform(tgt_img.size, Image.AFFINE, t_affine, Image.BICUBIC)
        affine_img = gen_img256.transform(tgt_img.size, Image.AFFINE, t_affine, Image.BICUBIC)
        gen_img.paste(affine_img, (0, 0), mask=mask)

        pasted_pil = gen_img
        return pasted_pil

    @torch.no_grad()
    def infer_image(self, source: Image.Image, target: Image.Image,
                    need_paste: bool = True, **kwargs) -> Image.Image:
        # 1. jpg to png
        s_bgr = self.pil_to_bgr(source)
        t_bgr = self.pil_to_bgr(target)

        # 2. get landmark
        s_bgr_resized, s_lmk = self.get_landmark_ori(s_bgr)
        t_bgr_resized, t_lmk = self.get_landmark_ori(t_bgr)

        # 3. align
        s_pil_aligned, s_affine = self.crop_ffhq(s_bgr_resized, s_lmk)
        t_pil_aligned, t_affine = self.crop_ffhq(t_bgr_resized, t_lmk)

        # 4. get landmark from aligned
        s_lmk_256 = self.get_lmk_256(s_pil_aligned)
        t_lmk_256 = self.get_lmk_256(t_pil_aligned)

        # 5. mtcnn
        s_mtcnn_256 = self.detect_faces_portrait(s_pil_aligned)
        t_mtcnn_256 = self.detect_faces_portrait(t_pil_aligned)

        # 6. obtain params of affine transformation
        s_pil_112, s_affine_theta_112 = self.face_align_portrait(s_pil_aligned, s_mtcnn_256)
        t_pil_112, t_affine_theta_112 = self.face_align_portrait(t_pil_aligned, t_mtcnn_256)

        # X. get batch
        batch = self.get_batch(
            s_pil_aligned, s_affine_theta_112, s_lmk_256,
            t_pil_aligned, t_affine_theta_112, t_lmk_256,
        )

        # Y. swap
        swapped_pil, mask_pil = self.perform_swap(
            batch=batch,
            ckpt=self.ckpt,
            ddim_sampler=self.ddim_sampler,
            ddim_steps=50,
        )
        swapped_pil.save("tmp_swapped_pil.jpg")
        mask_pil.save("tmp_mask_pil.jpg")

        # Z. paste back
        t_pil_aligned.save("tmp_t_pil_aligned.jpg")
        repaired_pil = self.repair_by_mask(
            swapped_pil, t_pil_aligned, mask_pil
        )
        repaired_pil.save("tmp_repaired_pil.jpg")

        if need_paste:
            repaired_pil = self.paste(
                repaired_pil, target, t_affine
            )

        return repaired_pil


if __name__ == '__main__':
    # s_path = make_abs_path("data/portrait_jpg/source/0.jpg")
    # t_path = make_abs_path("data/portrait_jpg/target/0.jpg")
    s_path = "/home/yuange/program/PyCharmRemote/e4s_v2/outputs/00001_29333_to_28905/S_cropped.png"
    t_path = "/home/yuange/program/PyCharmRemote/e4s_v2/outputs/00001_29333_to_28905/T_cropped.png"

    s_pil = Image.open(s_path).convert("RGB")
    t_pil = Image.open(t_path).convert("RGB")
    image_infer = DiffSwapImageInfer()
    r_pil = image_infer.infer_image(s_pil, t_pil)
    r_pil.save("tmp_final_swapped.jpg")
