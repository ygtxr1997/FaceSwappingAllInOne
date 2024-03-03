import os.path
import glob
import random

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.ffplus import FFPlusDataset
from datasets.celebahq import CelebAHQDataset
from datasets.output import OutputDataset

from preprocess.align_tools import FaceAlignImageInfer
from preprocess.align_dataset import align_ffplus

from methods.diffswap import DiffSwapImageInfer
from methods.blendface import BlendFaceImageInfer
from methods.infoswap import InfoSwapImageInfer
from methods.hires import HiResImageInfer
from methods.megafs import MegaFSImageInfer
from methods.simswap import SimSwapOfficialImageInfer
from methods.reliableswap import FaceShifterImageInfer
from methods.hififace import HiFiFaceImageInfer
from methods.stylefusion import StyleFusionImageInfer

from metrics.cosface import CosFaceImageInfer
from metrics.blendface import BlendFaceMetricImageInfer
from metrics.deep3d import Deep3DImageInfer
from metrics.hopenet import HopeNetImageInfer
from metrics.pytorch_fid import FidFolderInfer


def debug():
    # x = torch.randn(5, 3, 512, 512)
    # infer = CosFaceImageInfer()
    # z = infer.infer_tensor(x)
    # print(z.shape)

    # x = Image.open("28008.jpg")
    # x = np.array(x).astype(np.float32) / 127.5 - 1.
    # x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    # print(x.shape, x.min(), x.max())

    # infer = Deep3DImageInfer()
    # z = infer.infer_tensor(x)
    # print(z["exp"].shape)
    # noise = torch.randn_like(z["exp"]) + z["exp"]
    # loss = infer.calc_l2(z["exp"], noise)

    # infer = HopeNetImageInfer()
    # z = infer.infer_tensor(x)
    # print(z.shape)
    # noise = torch.randn_like(z) + z
    # loss = infer.calc_l2(z, noise)

    s_pil = Image.open("source.jpg")
    t_pil = Image.open("target.jpg")

    # infer = InfoSwapImageInfer()
    # infer = HiResImageInfer()
    # infer = MegaFSImageInfer()
    infer = StyleFusionImageInfer()

    r_pil = infer.infer_image(s_pil, t_pil)
    r_pil.save("tmp_result.jpg")


def fid(dataset_name="celebahq",
        method_name="diffswap",
        ):
    print(f"----------- [FID] ({dataset_name})-({method_name}) calculating --------------")
    f0 = f"/home/yuange/datasets/FaceSwapOutputs/{dataset_name}/{method_name}"
    f1 = f"/home/yuange/datasets/FaceSwapOutputs/{dataset_name}/source"
    f2 = f"/home/yuange/datasets/FaceSwapOutputs/{dataset_name}/target"
    infer = FidFolderInfer()
    s_score = infer.infer_folder(f0, f1, max_num1=25000, max_num2=25000, num_workers=4)
    t_score = infer.infer_folder(f0, f2, max_num1=25000, max_num2=25000, num_workers=4)
    avg_score = (s_score + t_score) / 2.

    print(f"[FID] ({dataset_name})-({method_name}):"
          f"source={s_score:.2f},target={t_score:.2f},avg={avg_score:.2f}")


def start_swap(dataset_name="celebahq",
               method_name="diffswap",
               resume=-1,
               ):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_root = "/home/yuange/datasets/FaceSwapOutputs"
    result_root = os.path.join(save_root, dataset_name, method_name)
    # source_root = os.path.join(save_root, dataset_name, "source")
    # target_root = os.path.join(save_root, dataset_name, "target")
    os.makedirs(result_root, exist_ok=True)
    # os.makedirs(source_root, exist_ok=True)
    # os.makedirs(target_root, exist_ok=True)

    if method_name == "diffswap":
        infer = DiffSwapImageInfer()
    elif method_name == "blendface":
        infer = BlendFaceImageInfer()
    elif method_name == "infoswap":
        infer = InfoSwapImageInfer()
    elif method_name == "hires":
        infer = HiResImageInfer()
    elif method_name == "megafs":
        infer = MegaFSImageInfer()
    elif method_name == "simswap":
        infer = SimSwapOfficialImageInfer()
    elif method_name == "faceshifter":
        infer = FaceShifterImageInfer()
    elif method_name == "hififace":
        infer = HiFiFaceImageInfer()
    elif method_name == "stylefusion":
        infer = StyleFusionImageInfer()
    else:
        raise ValueError(f"Method {method_name} not supported!")

    dataset = OutputDataset(
        save_root, max_len=10000,
        dataset_name=dataset_name,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    for idx, batch in enumerate(tqdm(dataloader)):
        if idx < resume:
            continue
        t_arr = ((batch["target_tensor"][0] + 1.) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        s_arr = ((batch["source_tensor"][0] + 1.) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        s_pil = Image.fromarray(s_arr)
        t_pil = Image.fromarray(t_arr)

        torch.cuda.empty_cache()

        save_name = "%06d" % (idx)

        swap_result = infer.infer_image(
            source=s_pil, target=t_pil,
            need_paste=False,
        )

        gpu_free, gpu_total = torch.cuda.mem_get_info()
        print(f'finished {save_name}. '
              f'GPU Free = {gpu_free / 1024 / 1024 / 1024} GB, '
              f'Total = {gpu_total / 1024 / 1024 / 1024} GB.')

        swap_result.save(os.path.join(result_root, "%06d.png" % idx))


@torch.no_grad()
def start_calc(save_root="/home/yuange/datasets/FaceSwapOutputs",
               dataset_name="ffplus",
               method_name="diffswap"
               ):
    print(("V" * 20) + f" [Metric] ({dataset_name})-({method_name}) " + ("V" * 20))
    dataset = OutputDataset(
        save_root, dataset_name, method_name,
        max_len=10000,
    )
    batch_size = 8

    # id_infer = CosFaceImageInfer()
    id_infer = BlendFaceMetricImageInfer()
    deep3d_infer = Deep3DImageInfer()
    pose_infer = HopeNetImageInfer()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    r_ids, t_ids, s_ids = [], [], []
    r_shapes, t_shapes, s_shapes = [], [], []
    r_exps, t_exps, s_exps = [], [], []
    r_poses, t_poses, s_poses = [], [], []
    for idx, batch in enumerate(tqdm(dataloader)):
        result = batch["result_tensor"]
        target = batch["target_tensor"]
        source = batch["source_tensor"]

        torch.cuda.empty_cache()

        cat = torch.cat([result, target, source], dim=0)

        # ID
        vec = id_infer.infer_tensor(cat).cpu()
        r_vec, t_vec, s_vec = torch.chunk(vec, 3, dim=0)
        r_ids.append(r_vec)
        t_ids.append(t_vec)
        s_ids.append(s_vec)

        # Shape & Expression
        vec = deep3d_infer.infer_tensor(cat)
        r_vec, t_vec, s_vec = torch.chunk(vec["id"].cpu(), 3, dim=0)
        r_shapes.append(r_vec)
        t_shapes.append(t_vec)
        s_shapes.append(s_vec)

        r_vec, t_vec, s_vec = torch.chunk(vec["exp"].cpu(), 3, dim=0)
        r_exps.append(r_vec)
        t_exps.append(t_vec)
        s_exps.append(s_vec)

        # Pose
        vec = pose_infer.infer_tensor(cat).cpu()
        r_vec, t_vec, s_vec = torch.chunk(vec, 3, dim=0)
        r_poses.append(r_vec)
        t_poses.append(t_vec)
        s_poses.append(s_vec)

    r_ids = torch.cat(r_ids, dim=0)
    t_ids = torch.cat(t_ids, dim=0)
    s_ids = torch.cat(s_ids, dim=0)
    id_infer.calc_id_retrieval(
        r_ids, s_ids
    )

    r_shapes = torch.cat(r_shapes, dim=0)
    t_shapes = torch.cat(t_shapes, dim=0)
    s_shapes = torch.cat(s_shapes, dim=0)
    deep3d_infer.calc_l2(
        r_shapes, s_shapes, "shape"
    )

    r_exps = torch.cat(r_exps, dim=0)
    t_exps = torch.cat(t_exps, dim=0)
    s_exps = torch.cat(s_exps, dim=0)
    deep3d_infer.calc_l2(
        r_exps, t_exps, "expression"
    )

    r_poses = torch.cat(r_poses, dim=0)
    t_poses = torch.cat(t_poses, dim=0)
    s_poses = torch.cat(s_poses, dim=0)
    pose_infer.calc_l2(
        r_poses, t_poses
    )
    print(("^" * 20) + f"[Metric] ({dataset_name})-({method_name}) is Finished!" + ("^" * 20) + "\n")


if __name__ == "__main__":
    # debug()

    # start_swap(
    #     dataset_name="celebahq",
    #     method_name="stylefusion",
    #     resume=-1,
    # )
    # start_swap(
    #     dataset_name="ffplus",
    #     method_name="stylefusion",
    #     resume=-1,
    # )

    eval_datasets = [
        "ffplus",
        "celebahq",
    ]
    eval_methods = [
        "blendface", "diffswap", "hififace",
        "infoswap", "megafs",
        "faceshifter", "simswap",
        "stylefusion", "e4s2022", "e4s2023recolor", "e4s2023final"
    ]

    for d in eval_datasets:
        for m in eval_methods:
            start_calc(
                dataset_name=d,
                method_name=m,
            )

    # for m in eval_methods:
    #     fid(
    #         dataset_name="celebahq",
    #         method_name=m,
    #     )

