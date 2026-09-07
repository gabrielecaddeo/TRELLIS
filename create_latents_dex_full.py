"""Encode ss_latents_sdf_pose for the FULL multi-view dex dataset (8 frames
per grasp) in the TRAIN-repo dataset convention.

Derived from create_latents_dex.py with the fixes of 2026-09-07:
- ALL frames encoded (--frames, default 0-7), not just f000;
- output names {instance}_{v}__{hand,object}.npz with PLAIN-INT view (what
  SparseStructureLatentSDFConditioned.get_instance reads; the old script's
  '_f00__' names are unreadable by it);
- strict encoder load;
- unchanged (verified correct): EMA encoder encoder_ema0.9999_step0300000.pt
  (same as the original training latents, dataset_toolkits/
  encode_ss_latents_sdf.py), posterior MEAN, clamp(-2, 2).

Usage:
  python create_latents_dex_full.py --data_root <root>/data_pose_norm \
      --output_dir <root> [--frames 8] [--rank i --world_size N] [--skip_existing]
"""
import os, sys, json, argparse
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
import trellis.models as models

torch.set_grad_enabled(False)


def load_sdf(path):
    sdf = torch.tensor(np.load(path), dtype=torch.float32)
    return torch.clamp(sdf, -2, 2)[None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help=".../data_pose_norm")
    ap.add_argument("--output_dir", required=True, help="dataset root (gets ss_latents_sdf_pose/)")
    ap.add_argument("--model_root", default="outputs")
    ap.add_argument("--enc_model", default="vae_final_all_resume_2")
    ap.add_argument("--ckpt", default="0300000")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--skip_existing", action="store_true")
    opt = edict(vars(ap.parse_args()))

    data_root = Path(opt.data_root).resolve()
    latent_name = f"{opt.enc_model}_{opt.ckpt}"
    latent_dir = Path(opt.output_dir).resolve() / "ss_latents_sdf_pose" / latent_name
    latent_dir.mkdir(parents=True, exist_ok=True)

    cfg = edict(json.load(open(Path("/projects/gcaddeo/train_flow/TRELLIS") / opt.model_root
                               / opt.enc_model / "config.json")))
    encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
    ckpt_path = (Path("/projects/gcaddeo/train_flow/TRELLIS") / opt.model_root / opt.enc_model
                 / "ckpts" / f"encoder_ema0.9999_step{opt.ckpt}.pt")
    encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=True)
    encoder.eval()
    print(f"Loaded EMA encoder (strict): {ckpt_path}")

    instances = sorted(p.name for p in data_root.iterdir() if p.is_dir()
                       and (p / "sdfs" / f"{p.name}_f000__object.npy").exists())
    lo = len(instances) * opt.rank // opt.world_size
    hi = len(instances) * (opt.rank + 1) // opt.world_size
    instances = instances[lo:hi]
    print(f"shard {opt.rank}/{opt.world_size}: {len(instances)} instances x {opt.frames} frames")

    records = []
    for inst in tqdm(instances):
        row = {"instance": inst, f"ss_latent_{latent_name}": True, "error": ""}
        try:
            for v in range(opt.frames):
                outs = {p: latent_dir / f"{inst}_{v}__{p}.npz" for p in ("hand", "object")}
                if opt.skip_existing and all(o.exists() for o in outs.values()):
                    continue
                for part, out in outs.items():
                    src = data_root / inst / "sdfs" / f"{inst}_f{v:03d}__{part}.npy"
                    z = encoder(load_sdf(src).cuda(), sample_posterior=False)
                    assert torch.isfinite(z).all(), f"non-finite latent {inst} f{v} {part}"
                    np.savez_compressed(out, mean=z[0].cpu().numpy())
        except Exception as e:
            print(f"[fail] {inst}: {e!r}", flush=True)
            row[f"ss_latent_{latent_name}"] = False
            row["error"] = repr(e)
        records.append(row)

    csv = Path(opt.output_dir) / f"ss_latent_{latent_name}_{opt.rank}_sdf.csv"
    pd.DataFrame.from_records(records).to_csv(csv, index=False)
    print(f"records -> {csv}\nlatents -> {latent_dir}")


if __name__ == "__main__":
    main()
