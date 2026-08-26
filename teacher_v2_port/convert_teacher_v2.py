"""Convert a train-repo denoiser EMA .pt checkpoint into a deployed pipeline dir.

Teacher default (unchanged from the parity-tested version): writes
teacher_v2_stage2/ckpts/denoiser_teacher_v2_final.safetensors (the name the
pipeline_conditioned.json references) plus a PROVENANCE.txt recording the source.

Generalized for other archs (e.g. the distilled student): pass --train_config
(the train-repo output dir's config.json) and --pipeline_dir. A fresh pipeline
dir is then assembled: the denoiser arch json is generated from the train
config's models.denoiser (use_checkpoint forced off for inference), the three
decoder ckpts are symlinked from the teacher pipeline dir (they are shared —
the student only replaces the sparse-structure flow model), and a
pipeline_conditioned.json is written pointing at the new denoiser. Example:

  python teacher_v2_port/convert_teacher_v2.py \
    --pt <train_repo>/outputs/distill_teacherv2/ckpts/denoiser_ema0.9999_step0081000.pt \
    --train_config <train_repo>/outputs/distill_teacherv2/config.json \
    --pipeline_dir /projects/gcaddeo/inference/TRELLIS/student_a_stage2 \
    --name denoiser_student_a_final
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file

TEACHER_PIPELINE = "/projects/gcaddeo/inference/TRELLIS/teacher_v2_stage2"
DECODER_STEMS = [
    "decoder_ema0.9999_step0300000",
    "decoder_slat_gs_ema0.9999_step0090000",
    "decoder_slat_mesh_ema0.9999_step0096000",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="source denoiser EMA .pt (train repo)")
    ap.add_argument("--pipeline_dir", default=TEACHER_PIPELINE,
                    help="deployed pipeline dir to write into (created if new)")
    ap.add_argument("--name", default="denoiser_teacher_v2_final",
                    help="ckpt stem inside <pipeline_dir>/ckpts")
    ap.add_argument("--train_config", default=None,
                    help="train-repo config.json; when given, also generates "
                         "<name>.json (arch) from its models.denoiser and, for a "
                         "non-teacher pipeline_dir, assembles the full pipeline dir")
    ap.add_argument("--out", default=None,
                    help="override full output .safetensors path (legacy flag)")
    args = ap.parse_args()

    ckpts_dir = os.path.join(args.pipeline_dir, "ckpts")
    out = args.out or os.path.join(ckpts_dir, args.name + ".safetensors")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    try:
        obj = torch.load(args.pt, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(args.pt, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in obj.items() if torch.is_tensor(v)}
    if not tensors:
        raise ValueError(f"No tensors found in {args.pt}")

    save_file(tensors, out)
    prov = os.path.join(os.path.dirname(out), "PROVENANCE.txt")
    with open(prov, "a") as f:
        f.write(f"{os.path.basename(out)} <- {os.path.realpath(args.pt)}\n")
    print(f"Saved {len(tensors)} tensors -> {out}")
    print(f"Provenance: {os.path.realpath(args.pt)}")

    if args.train_config:
        cfg = json.load(open(args.train_config))
        den = cfg["models"]["denoiser"]
        arch = {"name": den["name"], "args": dict(den["args"])}
        arch["args"]["use_checkpoint"] = False
        arch_path = os.path.splitext(out)[0] + ".json"
        with open(arch_path, "w") as f:
            json.dump(arch, f, indent=4)
        print(f"Arch json ({den['args'].get('num_blocks')} blocks, "
              f"mlp_ratio {den['args'].get('mlp_ratio')}) -> {arch_path}")

    if os.path.realpath(args.pipeline_dir) != os.path.realpath(TEACHER_PIPELINE):
        if not args.train_config:
            raise SystemExit("--train_config is required for a non-teacher --pipeline_dir "
                             "(the arch json cannot be assumed)")
        for stem in DECODER_STEMS:
            for ext in (".safetensors", ".json"):
                dst = os.path.join(ckpts_dir, stem + ext)
                src = os.path.join(TEACHER_PIPELINE, "ckpts", stem + ext)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        pipe = json.load(open(os.path.join(TEACHER_PIPELINE, "pipeline_conditioned.json")))
        pipe["args"]["models"]["sparse_structure_flow_model"] = "ckpts/" + args.name
        pipe_path = os.path.join(args.pipeline_dir, "pipeline_conditioned.json")
        with open(pipe_path, "w") as f:
            json.dump(pipe, f, indent=4)
        print(f"Pipeline dir assembled: {args.pipeline_dir} (decoders symlinked from teacher)")


if __name__ == "__main__":
    main()
