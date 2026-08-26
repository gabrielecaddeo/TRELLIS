"""P4 stage-2 precompute (EVAL_GUIDANCE.md §7.19): run the FROZEN best student
over the TRAINING sets' views (subsampled) and save the decoded object SDFs.
These become the prior source for the recursive student's stage-2 curriculum
(dataset arg prior_source='student', prior_student_dir=<out_dir>).

One batched forward per instance (all requested views), unguided, bf16
autocast. Output: <out_dir>/<instance>/f{view:03d}.npy (float16 [64,64,64],
the raw ss_dec SDF in the view's own grid). Resumable: instances whose files
all exist are skipped.

Launch via tools/precompute_student_recons.sbatch (GPU; needs user approval —
multi-GPU-day pass). Shard with --shard i --num_shards N to parallelize.
"""
import os, sys, json, argparse
import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_physics_losses import DiagnosticTrainer  # noqa: E402
from trellis import models, datasets  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--views", default="0,2,4,6,8,10,12,14,16,18,20,22",
                    help="views to reconstruct per grasp (every 2nd of 24 halves the pass)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg_strength", type=float, default=5.0)
    ap.add_argument("--cfg_interval", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--rescale_t", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    views = [int(v) for v in args.views.split(",")]

    cfg = edict(json.load(open(os.path.join(args.model_dir, "config.json"))))
    dataset = getattr(datasets, cfg.dataset.name)(args.data_dir, **cfg.dataset.args)
    dataset.inference = True

    model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()
    state = torch.load(os.path.join(args.model_dir, "ckpts", args.ckpt),
                       map_location="cuda", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    trainer_args = dict(cfg.trainer.args)
    trainer_args["batch_size_per_gpu"] = len(views)
    trainer_args["fp16_mode"] = None
    trainer = DiagnosticTrainer({"denoiser": model}, dataset, dataset,
                                output_dir=os.path.join(args.out_dir, "_trainer_scratch"),
                                load_dir=None, step=None, **trainer_args)
    sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=trainer.sigma_min)

    TENSOR_KEYS = ["x_0", "x0_hand", "cond", "mask_hand", "mask_obj", "cond_mask", "touch"]
    instances = list(dataset.instances)[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(instances)} instances x {len(views)} views, "
          f"ckpt {args.ckpt}, steps={args.steps}")

    n_done = n_skip = n_fail = 0
    for gi, (root, instance) in enumerate(instances):
        inst_out = os.path.join(args.out_dir, instance)
        paths = {v: os.path.join(inst_out, f"f{v:03d}.npy") for v in views}
        if all(os.path.exists(p) for p in paths.values()):
            n_skip += 1
            continue
        packs = []
        try:
            for v in views:
                dataset.force_view = v
                packs.append(dataset.get_instance(root, instance))
        except Exception as e:
            print(f"[skip] {instance}: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        finally:
            dataset.force_view = None

        data = {k: torch.stack([p[k] for p in packs]).cuda() for k in TENSOR_KEYS}
        x_0 = data.pop("x_0")
        g = torch.Generator(device=x_0.device).manual_seed(args.seed * 9973 + gi)
        noise = torch.randn(x_0.shape, generator=g, device=x_0.device, dtype=x_0.dtype)
        cond_args = trainer.get_inference_cond(**data)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = sampler.sample(model, noise, cond=cond_args["cond"],
                               neg_cond=cond_args["neg_cond"],
                               steps=args.steps, rescale_t=args.rescale_t,
                               cfg_strength=args.cfg_strength,
                               cfg_interval=tuple(args.cfg_interval),
                               verbose=False).samples
        with torch.no_grad():
            sdf = trainer.ss_dec(z).float().squeeze(1).cpu().numpy()  # [K,64,64,64]

        os.makedirs(inst_out, exist_ok=True)
        for i, v in enumerate(views):
            np.save(paths[v], sdf[i].astype(np.float16))
        n_done += 1
        if (gi + 1) % 50 == 0:
            print(f"  ... {gi+1}/{len(instances)} (done {n_done}, cached {n_skip}, failed {n_fail})")

    print(f"DONE: {n_done} computed, {n_skip} already cached, {n_fail} failed")


if __name__ == "__main__":
    main()
