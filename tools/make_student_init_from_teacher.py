"""Build a copy-init checkpoint for the 8-block / mlp_ratio-4 ablation student
(EVAL_GUIDANCE.md 7.8): every 3rd teacher block (0,3,...,21) -> student blocks
0..7, all non-block tensors copied verbatim. Valid ONLY for a student with
model_channels 1024 and mlp_ratio 4 (teacher shapes); the deployed 8x mlp2
student is shape-incompatible on purpose - do not point this at it.
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--teacher_ckpt", default="outputs/teacher_v2_stage2_physics/denoiser_teacher_v2_FROZEN.pt")
ap.add_argument("--out", default="outputs/distill_s8mlp4_copyinit/student_init_from_teacher.pt")
ap.add_argument("--blocks", default="0,3,6,9,12,15,18,21")
args = ap.parse_args()

src = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=True)
pick = [int(b) for b in args.blocks.split(",")]
assert len(pick) == len(set(pick))

out = {}
copied_blocks = 0
for k, v in src.items():
    if k.startswith("blocks."):
        idx = int(k.split(".")[1])
        if idx in pick:
            rest = k.split(".", 2)[2]
            out[f"blocks.{pick.index(idx)}.{rest}"] = v.clone()
            copied_blocks += 1
    else:
        out[k] = v.clone()

n_teacher_blocks = len({int(k.split('.')[1]) for k in src if k.startswith('blocks.')})
per_block = sum(1 for k in src if k.startswith('blocks.0.'))
assert copied_blocks == len(pick) * per_block, (copied_blocks, len(pick), per_block)
print(f"teacher blocks: {n_teacher_blocks}, picked {pick} -> student 0..{len(pick)-1}")
print(f"copied {copied_blocks} block tensors + {len(out)-copied_blocks} non-block tensors = {len(out)} total")
import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save(out, args.out)
print(f"wrote {args.out}")
