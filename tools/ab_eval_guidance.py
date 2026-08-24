"""
A/B evaluation of physics guidance at sampling time, run entirely in the train repo.

Three arms, same noise and same conditioning per sample:
  A  unguided  -- FlowEulerGuidanceIntervalSampler.sample() with the deployment CFG params
  B  guided    -- sample_velocity_conditioned() (DPS-style physics guidance on x0_hat),
                  alpha_vel matching the operative inference-repo value
  GT floor     -- the ground-truth latent decoded through the same frozen decoder; every
                  metric is also reported relative to this, since the decoder itself
                  imposes nonzero penetration/contact error (TEACHER_RETRAIN.md §8.1)

Metrics per decoded SDF (64^3, unit cube, 1 voxel = 1/64):
  pen_frac       fraction of grid voxels with sdf_obj < 0 AND sdf_hand < 0
  pen_depth      mean object depth (-sdf_obj) over penetrating voxels (0 if none)
  contact_abs    mean |sdf_obj| over annotated contact voxels (touch[:,0])
  contact_hit1v  fraction of contact voxels with |sdf_obj| < 1 voxel
  occ_iou_gt     occupancy IoU vs the GT decode (shape fidelity; guidance must not buy
                 physics by destroying geometry)
  occ_frac       occupied fraction of the grid (sanity: does guidance shrink objects?)

Read-only: writes one JSON (+ optional sample SDFs) under outputs/diagnostics/.
"""
import os
import sys
import json
import glob
import argparse
import random

import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_physics_losses import DiagnosticTrainer, to_cuda  # noqa: E402
from mesh_metrics import paired_mesh_metrics  # noqa: E402

from trellis import models, datasets  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402


VOXEL = 1.0 / 64.0


@torch.no_grad()
def sdf_metrics(sdf_obj, sdf_hand, touch, sdf_gt=None):
    """Per-sample physics/shape metrics for a batch of decoded SDFs."""
    B = sdf_obj.shape[0]
    obj = sdf_obj.view(B, -1)
    hand = sdf_hand.view(B, -1)
    pen = (obj < 0) & (hand < 0)

    pen_frac = pen.float().mean(dim=1)
    depth = (-obj).clamp_min(0) * pen.float()
    pen_depth = depth.sum(dim=1) / pen.float().sum(dim=1).clamp_min(1.0)

    cmask = (touch[:, 0].reshape(B, -1) > 0).float()
    cden = cmask.sum(dim=1).clamp_min(1.0)
    contact_abs = (obj.abs() * cmask).sum(dim=1) / cden
    contact_hit1v = ((obj.abs() < VOXEL).float() * cmask).sum(dim=1) / cden

    occ = obj < 0
    occ_frac = occ.float().mean(dim=1)
    out = dict(
        pen_frac=pen_frac, pen_depth=pen_depth,
        contact_abs=contact_abs, contact_hit1v=contact_hit1v,
        occ_frac=occ_frac,
    )
    if sdf_gt is not None:
        gt_occ = sdf_gt.view(B, -1) < 0
        inter = (occ & gt_occ).float().sum(dim=1)
        union = (occ | gt_occ).float().sum(dim=1).clamp_min(1.0)
        out["occ_iou_gt"] = inter / union
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher_dir", type=str,
                    default="outputs/teacher_v2_stage2_physics")
    ap.add_argument("--ckpt", type=str, default="latest_ema",
                    help="Checkpoint filename in <teacher_dir>/ckpts/, or 'latest_ema'")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Comma-separated held-out roots")
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    # deployment sampler params (pipeline_conditioned.json in the inference repo)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg_strength", type=float, default=5.0)
    ap.add_argument("--cfg_interval", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--rescale_t", type=float, default=3.0)
    # guidance params (operative inference-repo values)
    ap.add_argument("--alpha_vel", type=float, default=10.0)
    ap.add_argument("--guidance_skip", type=int, default=5)
    # arms to run
    ap.add_argument("--arms", type=str,
                    default="unguided,guided_asis,guided_v2,oc_flow",
                    help="Comma-separated subset of: unguided,guided_asis,guided_v2,oc_flow")
    # improved-guidance (v2) params
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--time_power", type=float, default=1.0)
    ap.add_argument("--contact_floor", type=float, default=0.011)
    # complete OC-Flow params
    ap.add_argument("--oc_n_outer", type=int, default=4)
    ap.add_argument("--oc_eta_u", type=float, default=0.5)
    ap.add_argument("--oc_u_max_ratio", type=float, default=0.3)
    ap.add_argument("--oc_no_jacobian", action="store_true",
                    help="FlowGrad-style straight-through (skip VJPs through the model)")
    # standard paired mesh metrics (CD / NC / F@tau / EMD), vs the GT decode
    ap.add_argument("--mesh_metrics", type=int, default=1,
                    help="1 = also compute CD/NC/F@tau/EMD per sample (CPU)")
    ap.add_argument("--mesh_points", type=int, default=10000)
    ap.add_argument("--emd_points", type=int, default=1024)
    ap.add_argument("--f_tau", type=float, default=0.02)
    ap.add_argument("--no_hand_pe", action="store_true",
                    help="Build the model with use_hand_pe=False -- required to "
                         "faithfully evaluate pre-fix checkpoints (the old teacher), "
                         "whose forward never added hand positional embeddings.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--data_seed", type=int, default=None,
                    help="Re-seed torch's global RNG with this value right before the "
                         "dataloader iterator is created. The DataLoader shuffles from "
                         "the global RNG, whose state at that point depends on how many "
                         "draws model construction consumed -- so a 24-block teacher and "
                         "an 8-block student otherwise evaluate DIFFERENT samples. Set "
                         "the same value for every run that must be paired across "
                         "models. Default None keeps the legacy (model-dependent) order.")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--save_sdf_samples", type=int, default=-1,
                    help="Save decoded SDFs + input image for the first N samples "
                         "(-1 = all, 0 = disable). Used by tools/visualize_ab_sdfs.py")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = edict(json.load(open(os.path.join(args.teacher_dir, "config.json"))))

    if args.ckpt == "latest_ema":
        cands = sorted(glob.glob(os.path.join(args.teacher_dir, "ckpts", "denoiser_ema*.pt")))
        assert cands, "no EMA checkpoints found"
        args.ckpt = os.path.basename(cands[-1])
    ckpt_path = os.path.join(args.teacher_dir, "ckpts", args.ckpt)

    out_path = args.output or os.path.join(
        "outputs", "diagnostics",
        f"ab_guidance_{os.path.basename(args.teacher_dir)}_{args.ckpt.replace('.pt', '')}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sdf_dir = out_path.replace(".json", "_sdfs")
    if args.save_sdf_samples != 0:
        os.makedirs(sdf_dir, exist_ok=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"checkpoint : {ckpt_path}")
    print(f"data_dir   : {args.data_dir}")
    print(f"output     : {out_path}")
    print(f"arms       : {arms} + gt_floor")
    print(f"sampler    : steps={args.steps} cfg={args.cfg_strength} interval={args.cfg_interval} rescale_t={args.rescale_t}")
    print(f"guided_asis: alpha={args.alpha_vel} skip={args.guidance_skip}")
    print(f"guided_v2  : rho={args.rho} time_power={args.time_power} contact_floor={args.contact_floor}")
    print(f"oc_flow    : n_outer={args.oc_n_outer} eta_u={args.oc_eta_u} u_max={args.oc_u_max_ratio} jacobian={not args.oc_no_jacobian}")

    dataset = getattr(datasets, cfg.dataset.name)(args.data_dir, **cfg.dataset.args)
    if hasattr(dataset, "inference"):
        dataset.inference = True
    print(dataset)

    model_args = dict(cfg.models.denoiser.args)
    if args.no_hand_pe:
        model_args["use_hand_pe"] = False
    model = getattr(models, cfg.models.denoiser.name)(**model_args).cuda()
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"loaded {len(state)} tensors (strict=True)")

    trainer_args = dict(cfg.trainer.args)
    trainer_args["batch_size_per_gpu"] = args.batch_size
    trainer_args["fp16_mode"] = None
    trainer = DiagnosticTrainer(
        {"denoiser": model}, dataset, dataset,
        output_dir=os.path.join(os.path.dirname(out_path), "_trainer_scratch"),
        load_dir=None, step=None, **trainer_args,
    )

    sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=trainer.sigma_min)
    cfg_kwargs = dict(cfg_strength=args.cfg_strength,
                      cfg_interval=tuple(args.cfg_interval))

    acc = {arm: {} for arm in arms + ["gt_floor"]}

    def push(arm, m):
        for k, v in m.items():
            acc[arm].setdefault(k, []).append(v.cpu())

    n_done = 0
    if args.data_seed is not None:
        torch.manual_seed(args.data_seed)
    it = iter(trainer.dataloader)
    while n_done < args.num_samples:
        data = to_cuda(next(it))
        x_0 = data.pop("x_0")
        B = x_0.shape[0]
        g = torch.Generator(device=x_0.device).manual_seed(args.seed * 9973 + n_done)
        noise = torch.randn(x_0.shape, generator=g, device=x_0.device, dtype=x_0.dtype)

        cond_args = trainer.get_inference_cond(**data)
        pos, neg = cond_args["cond"], cond_args["neg_cond"]

        samples = {}
        if "unguided" in arms:
            with torch.no_grad():
                samples["unguided"] = sampler.sample(
                    model, noise, cond=pos, neg_cond=neg,
                    steps=args.steps, rescale_t=args.rescale_t,
                    verbose=False, **cfg_kwargs).samples
        if "guided_asis" in arms:
            samples["guided_asis"] = sampler.sample_velocity_conditioned(
                model, noise, trainer.ss_dec, cond=pos, neg_cond=neg,
                steps=args.steps, rescale_t=args.rescale_t,
                alpha_vel=args.alpha_vel, guidance_skip=args.guidance_skip,
                verbose=False, **cfg_kwargs).samples
        if "guided_v2" in arms:
            samples["guided_v2"] = sampler.sample_guided_v2(
                model, noise, trainer.ss_dec, cond=pos, neg_cond=neg,
                steps=args.steps, rescale_t=args.rescale_t,
                rho=args.rho, time_power=args.time_power,
                guidance_skip=args.guidance_skip,
                contact_floor=args.contact_floor,
                verbose=False, **cfg_kwargs).samples
        if "oc_flow" in arms:
            samples["oc_flow"] = sampler.sample_oc_flow(
                model, noise, trainer.ss_dec, cond=pos, neg_cond=neg,
                steps=args.steps, rescale_t=args.rescale_t,
                n_outer=args.oc_n_outer, eta_u=args.oc_eta_u,
                u_max_ratio=args.oc_u_max_ratio,
                use_model_jacobian=not args.oc_no_jacobian,
                contact_floor=args.contact_floor,
                verbose=True, **cfg_kwargs).samples

        with torch.no_grad():
            sdf_hand = trainer.ss_dec(pos["x0_hand"]).float()
            sdf_gt = trainer.ss_dec(x_0).float()
            touch = pos["touch"]

            sdfs = {arm: trainer.ss_dec(z).float() for arm, z in samples.items()}
            for arm, sdf in sdfs.items():
                push(arm, sdf_metrics(sdf, sdf_hand, touch, sdf_gt))
            push("gt_floor", sdf_metrics(sdf_gt, sdf_hand, touch, sdf_gt))

            if args.mesh_metrics:
                # CD / NC / F@tau / EMD vs the GT decode, per sample (CPU).
                gt_np = sdf_gt[:, 0].cpu().numpy()
                rng_mesh = np.random.default_rng(args.seed * 31337 + n_done)
                for arm, sdf in sdfs.items():
                    rec_np = sdf[:, 0].cpu().numpy()
                    vals = {}
                    for b in range(B):
                        m = paired_mesh_metrics(rec_np[b], gt_np[b], rng_mesh,
                                                n_points=args.mesh_points,
                                                emd_points=args.emd_points,
                                                tau=args.f_tau)
                        if m is None:
                            continue
                        for k, v in m.items():
                            vals.setdefault(k, []).append(v)
                    push(arm, {k: torch.tensor(v) for k, v in vals.items() if v})

            save_all = args.save_sdf_samples < 0
            if save_all or (args.save_sdf_samples > 0 and n_done < args.save_sdf_samples):
                k = B if save_all else min(B, args.save_sdf_samples - n_done)
                blob = {f"sdf_{arm}": sdf[:k].cpu() for arm, sdf in sdfs.items()}
                blob.update({"sdf_gt": sdf_gt[:k].cpu(), "sdf_hand": sdf_hand[:k].cpu(),
                             "touch": touch[:k].cpu(),
                             "image": (data["cond"][:k] * 255).clamp(0, 255)
                                      .to(torch.uint8).cpu()})
                torch.save(blob, os.path.join(sdf_dir, f"samples_{n_done:04d}.pt"))

        n_done += B
        print(f"  ... {n_done}/{args.num_samples} samples")

        # Incremental checkpoint of the aggregates so a wall-clock kill loses nothing.
        partial = {"meta": {"num_samples_done": n_done}}
        for arm in acc:
            partial[arm] = {k: {"mean": float(torch.cat(v).mean()),
                                "std": float(torch.cat(v).std())}
                            for k, v in acc[arm].items()}
        with open(out_path + ".partial", "w") as f:
            json.dump(partial, f, indent=2)

    # ---- aggregate & report -------------------------------------------------------
    results = {"meta": {
        "teacher_dir": args.teacher_dir, "ckpt": args.ckpt, "data_dir": args.data_dir,
        "num_samples": n_done, "steps": args.steps, "cfg_strength": args.cfg_strength,
        "cfg_interval": args.cfg_interval, "rescale_t": args.rescale_t,
        "alpha_vel": args.alpha_vel, "guidance_skip": args.guidance_skip,
        "arms": arms, "rho": args.rho, "time_power": args.time_power,
        "contact_floor": args.contact_floor, "oc_n_outer": args.oc_n_outer,
        "oc_eta_u": args.oc_eta_u, "oc_u_max_ratio": args.oc_u_max_ratio,
        "oc_use_jacobian": not args.oc_no_jacobian,
        "seed": args.seed, "data_seed": args.data_seed,
    }}
    for arm in acc:
        results[arm] = {}
        for k in sorted(acc[arm].keys()):
            v = torch.cat(acc[arm][k]).numpy()
            results[arm][k] = {"mean": float(v.mean()), "std": float(v.std()),
                               "n": int(v.size)}

    metrics = sorted({k for arm in acc for k in acc[arm].keys()})
    print(f"\n{'metric':<16}" + "".join(f"{arm:>16}" for arm in acc))
    for k in metrics:
        row = f"{k:<16}"
        for arm in acc:
            row += (f"{results[arm][k]['mean']:>16.5g}"
                    if k in results[arm] else f"{'-':>16}")
        print(row)

    print("\nFloor-relative (arm - gt_floor), the honest physics numbers:")
    for k in ("pen_frac", "pen_depth", "contact_abs"):
        gt = results["gt_floor"][k]["mean"]
        line = f"{k:<16}"
        for arm in arms:
            line += f"{results[arm][k]['mean'] - gt:>16.5g}"
        print(line + f"    (floor {gt:.5g})")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
