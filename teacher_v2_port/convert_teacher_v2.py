"""Convert a teacher_v2 EMA .pt checkpoint into the deployed pipeline dir.

Writes teacher_v2_stage2/ckpts/denoiser_teacher_v2_final.safetensors (the name the
pipeline_conditioned.json references) plus a PROVENANCE.txt recording the source.
"""
import argparse
import os

import torch
from safetensors.torch import save_file

PIPELINE_CKPTS = "/projects/gcaddeo/inference/TRELLIS/teacher_v2_stage2/ckpts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="source denoiser EMA .pt (train repo)")
    ap.add_argument("--out", default=os.path.join(PIPELINE_CKPTS, "denoiser_teacher_v2_final.safetensors"))
    args = ap.parse_args()

    try:
        obj = torch.load(args.pt, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(args.pt, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in obj.items() if torch.is_tensor(v)}
    if not tensors:
        raise ValueError(f"No tensors found in {args.pt}")

    save_file(tensors, args.out)
    prov = os.path.join(os.path.dirname(args.out), "PROVENANCE.txt")
    with open(prov, "w") as f:
        f.write(f"denoiser_teacher_v2_final.safetensors <- {os.path.realpath(args.pt)}\n")
    print(f"Saved {len(tensors)} tensors -> {args.out}")
    print(f"Provenance: {os.path.realpath(args.pt)}")


if __name__ == "__main__":
    main()
