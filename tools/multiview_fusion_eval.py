"""P1 of the multi-view study (EVAL_GUIDANCE.md §7.9): reconstruct a grasp from K
views independently (batched, unguided), warp all SDFs into the reference view's
grid with the validated similarity transforms, fuse, and score against the
reference view's GT — physics metrics (floor-relative) + CD/NC/F/EMD, same
conventions as tools/ab_eval_guidance.py.

Fusion arms per K: `mean`, `median`, `vismean` (per-voxel visibility-weighted
mean: a view votes where the voxel is unoccluded along its own -z viewing axis
through the predicted-object+GT-hand scene; median fallback where no view sees).
`single` (reference view alone) is the K=1 baseline; `gt_floor` as usual.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_physics_losses import DiagnosticTrainer  # noqa: E402
from ab_eval_guidance import sdf_metrics  # noqa: E402
from mesh_metrics import paired_mesh_metrics  # noqa: E402
from multiview_warp import warp_sdf, torch_warp_coords  # noqa: E402

from trellis import models, datasets  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402

TENSOR_KEYS = ["x_0", "x0_hand", "cond", "mask_hand", "mask_obj", "cond_mask", "touch"]


def visibility(scene_sdf):
    """1.0 where the voxel is unoccluded along the view axis (camera at +z looking
    -z, meta pose2d view_axis=-z; grid axis k -> z): no inside-scene voxel with
    larger z in the same (x,y) column. RETIRED for fusion weighting (noisy: built
    from the *predicted* object; see EVAL_GUIDANCE 7.11) — kept for reference."""
    inside = scene_sdf < 0
    occ = np.zeros_like(inside)
    occ[..., :-1] = np.flip(np.maximum.accumulate(np.flip(inside[..., 1:], axis=2), axis=2), axis=2)
    return (~occ).astype(np.float32)


def hand_visibility(sdf_hand, alpha=1.0, floor=0.25):
    """Soft, EXACT hand-occlusion weight (user insight 2026-08-24): the hand SDF
    is ground-truth conditioning, so 'how much hand does this view's ray pass
    through before reaching the voxel' is noise-free. w = exp(-alpha * number of
    inside-hand voxels above along +z), floored so occluded views keep a weak
    vote (their generative completion is still informative)."""
    inside = (sdf_hand < 0).astype(np.float32)
    above = np.zeros_like(inside)
    above[..., :-1] = np.flip(np.cumsum(np.flip(inside[..., 1:], axis=2), axis=2), axis=2)
    w = np.exp(-alpha * above)
    return floor + (1.0 - floor) * w


def load_meta(root, instance, view):
    p = os.path.join(root, "data_pose_norm", instance, f"{instance}_f{view:03d}_meta.json")
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--num_groups", type=int, default=48)
    ap.add_argument("--views", default="0,3,6,9,12,15,18,21",
                    help="views to reconstruct; first is the reference")
    ap.add_argument("--k_subsets", default="1:0;2:0,12;4:0,6,12,18;8:0,3,6,9,12,15,18,21",
                    help="fusion set sizes as K:view,view,...")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg_strength", type=float, default=5.0)
    ap.add_argument("--cfg_interval", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--rescale_t", type=float, default=3.0)
    ap.add_argument("--mesh_points", type=int, default=10000)
    ap.add_argument("--emd_points", type=int, default=1024)
    ap.add_argument("--consistency", action="store_true",
                    help="Also run the P2 cross-view consistency-guided sampler "
                         "(sample_multiview_consistency, all views jointly, same "
                         "noise) and report consist_single / consist_median_K<all> "
                         "rows alongside the unguided ones.")
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--band", type=float, default=0.15)
    ap.add_argument("--guidance_skip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    views = [int(v) for v in args.views.split(",")]
    ref = views[0]
    subsets = {}
    for part in args.k_subsets.split(";"):
        k, vs = part.split(":")
        vs = [int(v) for v in vs.split(",")]
        assert vs[0] == ref and all(v in views for v in vs), (k, vs)
        subsets[int(k)] = vs

    if args.ckpt == "latest_ema":
        # Same resolution as ab_eval_guidance. Only safe on single-purpose
        # student dirs whose training has FINISHED — never the teacher dir
        # (it holds unevaluated post-freeze leftovers).
        import glob
        cands = sorted(glob.glob(os.path.join(args.model_dir, "ckpts", "denoiser_ema*.pt")))
        assert cands, f"no EMA ckpts in {args.model_dir}/ckpts"
        args.ckpt = os.path.basename(cands[-1])
        print(f"latest_ema resolved to {args.ckpt}")

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
                                output_dir=os.path.join(os.path.dirname(args.output), "_trainer_scratch_mv"),
                                load_dir=None, step=None, **trainer_args)
    sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=trainer.sigma_min)

    # Deterministic group list: first N instances of the filtered metadata.
    groups = list(dataset.instances[:args.num_groups])
    print(f"{len(groups)} groups x views {views} (ref {ref}), K subsets {sorted(subsets)}")
    print(f"model {args.model_dir}/{args.ckpt}, steps={args.steps}")

    arms = ["single"] + [f"{m}_K{k}" for k in sorted(subsets) if k > 1
                         for m in ("mean", "median", "vishand", "hybrid")]
    if args.consistency:
        arms += ["consist_single", f"consist_median_K{len(views)}"]
    acc = {a: {} for a in arms + ["gt_floor"]}

    def push(arm, m):
        for k, v in m.items():
            acc[arm].setdefault(k, []).append(v.cpu() if torch.is_tensor(v) else torch.tensor(v))

    n_done = 0
    for gi, (root, instance) in enumerate(groups):
        # ---- load all views of this grasp ----
        packs, metas = [], {}
        try:
            for v in views:
                dataset.force_view = v
                packs.append(dataset.get_instance(root, instance))
                metas[v] = load_meta(root, instance, v)
        except Exception as e:
            print(f"[skip] {instance}: {type(e).__name__}: {e}")
            continue
        finally:
            dataset.force_view = None

        data = {k: torch.stack([p[k] for p in packs]).cuda() for k in TENSOR_KEYS}
        x_0 = data.pop("x_0")
        B = x_0.shape[0]

        g = torch.Generator(device=x_0.device).manual_seed(args.seed * 9973 + gi)
        noise = torch.randn(x_0.shape, generator=g, device=x_0.device, dtype=x_0.dtype)
        cond_args = trainer.get_inference_cond(**data)
        pos, neg = cond_args["cond"], cond_args["neg_cond"]

        with torch.no_grad():
            z = sampler.sample(model, noise, cond=pos, neg_cond=neg,
                               steps=args.steps, rescale_t=args.rescale_t,
                               cfg_strength=args.cfg_strength,
                               cfg_interval=tuple(args.cfg_interval),
                               verbose=False).samples
            sdf_pred = trainer.ss_dec(z).float()            # [B,1,64,64,64]
            sdf_hand = trainer.ss_dec(pos["x0_hand"]).float()
            sdf_gt_ref = trainer.ss_dec(x_0[:1]).float()    # ref view GT decode

        sdf_consist = None
        if args.consistency:
            coords, scales = [], []
            for v in views:
                c, sc = torch_warp_coords(metas[v], metas[ref])
                coords.append(torch.from_numpy(np.ascontiguousarray(c)).float())
                scales.append(sc)
            coords = torch.stack(coords).cuda()
            scales_t = torch.tensor(scales).float().cuda()
            zc = sampler.sample_multiview_consistency(
                model, noise.clone(), trainer.ss_dec, coords, scales_t,
                cond=pos, neg_cond=neg,
                steps=args.steps, rescale_t=args.rescale_t,
                rho=args.rho, band=args.band, guidance_skip=args.guidance_skip,
                cfg_strength=args.cfg_strength,
                cfg_interval=tuple(args.cfg_interval),
                verbose=False).samples
            with torch.no_grad():
                sdf_consist = trainer.ss_dec(zc).float()

        sdf_hand_ref = sdf_hand[:1]
        touch_ref = pos["touch"][:1]

        # ---- warp every view's prediction into the ref grid; visibility in own grid ----
        warped, warped_vis = {}, {}
        for bi, v in enumerate(views):
            pred_np = sdf_pred[bi, 0].cpu().numpy()
            vis = hand_visibility(sdf_hand[bi, 0].cpu().numpy())
            if v == ref:
                warped[v], warped_vis[v] = pred_np, vis
            else:
                warped[v] = warp_sdf(pred_np, metas[v], metas[ref])
                warped_vis[v] = np.clip(warp_sdf(vis, metas[v], metas[ref], cval=0.0), 0.0, 1.0)

        fused = {"single": warped[ref]}
        if sdf_consist is not None:
            cw = []
            for bi, v in enumerate(views):
                cn = sdf_consist[bi, 0].cpu().numpy()
                cw.append(cn if v == ref else warp_sdf(cn, metas[v], metas[ref]))
            fused["consist_single"] = cw[0]
            fused[f"consist_median_K{len(views)}"] = np.median(np.stack(cw), axis=0)
        for k, vs in subsets.items():
            if k == 1:
                continue
            W = np.stack([warped[v] for v in vs])
            V = np.stack([warped_vis[v] for v in vs])
            med = np.median(W, axis=0)
            fused[f"mean_K{k}"] = W.mean(axis=0)
            fused[f"median_K{k}"] = med
            wmean = (W * V).sum(axis=0) / np.clip(V.sum(axis=0), 1e-6, None)
            fused[f"vishand_K{k}"] = wmean
            # Hybrid (user proposal 2026-08-24): trust the visibility-weighted
            # mean where the views agree, fall back to the robust median where
            # they disagree. Smooth gate on across-view std, tau = 1 voxel.
            std = W.std(axis=0)
            tau = 1.0 / 64.0
            g = np.exp(-(std / tau) ** 2)
            fused[f"hybrid_K{k}"] = g * wmean + (1.0 - g) * med

        # ---- metrics in the ref frame ----
        rng_mesh = np.random.default_rng(args.seed * 31337 + gi)
        gt_np = sdf_gt_ref[0, 0].cpu().numpy()
        for arm, sdf_np in fused.items():
            t = torch.from_numpy(np.ascontiguousarray(sdf_np))[None, None].float().cuda()
            push(arm, sdf_metrics(t, sdf_hand_ref, touch_ref, sdf_gt_ref))
            m = paired_mesh_metrics(sdf_np, gt_np, rng_mesh,
                                    n_points=args.mesh_points, emd_points=args.emd_points)
            if m is not None:
                push(arm, {k2: torch.tensor(v) for k2, v in m.items()})
        push("gt_floor", sdf_metrics(sdf_gt_ref, sdf_hand_ref, touch_ref, sdf_gt_ref))

        n_done += 1
        print(f"  ... group {n_done}/{len(groups)} ({instance})")
        partial = {"meta": {"groups_done": n_done}}
        for a in acc:
            partial[a] = {k2: {"mean": float(torch.stack(vv).mean())} for k2, vv in acc[a].items()}
        with open(args.output + ".partial", "w") as f:
            json.dump(partial, f, indent=2)

    results = {"meta": {**vars(args), "groups_done": n_done, "views": views,
                        "subsets": {str(k): v for k, v in subsets.items()}}}
    for a in acc:
        results[a] = {k2: {"mean": float(torch.stack(vv).mean()),
                           "std": float(torch.stack(vv).std()),
                           "n": len(vv)}
                      for k2, vv in acc[a].items()}

    metrics = sorted({k2 for a in acc for k2 in acc[a]})
    order = ["gt_floor", "single"] + [a for a in arms if a != "single"]
    print(f"\n{'metric':<16}" + "".join(f"{a:>14}" for a in order))
    for k2 in metrics:
        row = f"{k2:<16}"
        for a in order:
            row += (f"{results[a][k2]['mean']:>14.5g}" if k2 in results.get(a, {}) else f"{'-':>14}")
        print(row)
    print("\nFloor-relative contact_abs:")
    fl = results["gt_floor"]["contact_abs"]["mean"]
    for a in order[1:]:
        if "contact_abs" in results[a]:
            print(f"  {a:<14} {results[a]['contact_abs']['mean'] - fl:.5g}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
