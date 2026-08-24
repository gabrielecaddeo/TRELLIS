import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import torch.multiprocessing as mp
import numpy as np
import random

from trellis import models, datasets, trainers
from trellis.utils.dist_utils import setup_dist


seed = 1337
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True, warn_only=True)


def find_ckpt(cfg):
    cfg["load_ckpt"] = None
    if cfg.load_dir != "":
        if cfg.ckpt == "latest":
            files = glob.glob(os.path.join(cfg.load_dir, "ckpts", "misc_*.pt"))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split("step")[-1].split(".")[0])
                    for f in files
                ])
        elif cfg.ckpt == "none":
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    model_summary = "Parameters:\n"
    model_summary += "=" * 128 + "\n"
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f"{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n"
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += "\n"
    model_summary += f"Number of parameters: {num_params}\n"
    model_summary += f"Number of trainable parameters: {num_trainable_params}\n"
    return model_summary


def main(local_rank, cfg):
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    setup_rng(rank)

    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f"\n\nBackbone: {name}\n" + model_summary)
            with open(os.path.join(cfg.output_dir, f"{name}_model_summary.txt"), "w") as fp:
                print(model_summary, file=fp)

    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict,
        dataset,
        None,
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        load_dir=cfg.load_dir,
        step=cfg.load_ckpt,
    )

    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/user/TRELLIS/configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill.json",
        help="Distillation config file.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/user/TRELLIS/outputs/flow_conditioned_distilled",
        type=str,
        help="Output directory.",
    )
    parser.add_argument("--load_dir", type=str, default="", help="Load directory, default to output_dir.")
    parser.add_argument("--ckpt", type=str, default="latest", help="Checkpoint step to resume training.")
    parser.add_argument("--data_dir", type=str, default="/home/user/TRELLIS/datasets/Hands")
    parser.add_argument("--teacher_config", type=str, default="", help="Teacher model config path.")
    parser.add_argument("--teacher_ckpt", type=str, default="", help="Teacher denoiser checkpoint path.")
    parser.add_argument("--auto_retry", type=int, default=3)
    parser.add_argument("--tryrun", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12345")
    opt = parser.parse_args()

    opt.load_dir = opt.load_dir if opt.load_dir != "" else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus

    config = json.load(open(opt.config, "r"))
    if opt.teacher_config:
        config["trainer"]["args"]["teacher_config_path"] = opt.teacher_config
    if opt.teacher_ckpt:
        config["trainer"]["args"]["teacher_ckpt_path"] = opt.teacher_ckpt

    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)

    print("\n\nConfig:")
    print("=" * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "command.txt"), "w") as fp:
            print(" ".join(["python"] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, "config.json"), "w") as fp:
            json.dump(config, fp, indent=4)

    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying ({rty + 1}/{cfg.auto_retry})...")
