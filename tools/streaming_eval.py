"""Streaming (recursive) evaluation for P4 (EVAL_GUIDANCE.md §7.19/§7.21).

Simulates a live stream per held-out grasp: views arrive in order (REF view
LAST, so the final frame's metrics live in the same ref grid as the fusion
tables). At frame t the prior is the per-voxel MEDIAN of the model's OWN
previous outputs warped into the current view's grid, encoded by the frozen
VAE encoder and fed as cond['x0_prior'] (prior-aware models only; frame 0 and
prior-less models run with prior_keep=0, which the zero-init/gated channel
turns into exact single-view behavior).

Arms reported (metrics in the ref frame, same harness rows as
multiview_fusion_eval):
  single            ref view, NO prior (single-view mode / copy-shortcut check)
  stream_final      ref view's direct output with prior from the 7 previous
                    frames — THE recursive-integration claim arm
  stream_median     median over all 8 streamed outputs warped to ref
                    (recursion + post-hoc fusion stacked)
  ringbuffer_median median over 8 NO-prior outputs (the training-free
                    baseline; == fusion median_K8 for prior-less models)
Plus per-frame trajectory of the direct streamed output (frame_1..frame_K,
each in its own view's grid vs that view's GT — quality vs #frames seen).

Usage mirrors multiview_fusion_eval.py; --prior self|none (none forces the
baseline behavior for any model).
"""
import os, sys, json, argparse
import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_physics_losses import DiagnosticTrainer  # noqa: E402
from ab_eval_guidance import sdf_metrics  # noqa: E402
from mesh_metrics import paired_mesh_metrics  # noqa: E402
from multiview_warp import warp_sdf  # noqa: E402
from trellis import models, datasets  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402

TENSOR_KEYS = ["x_0", "x0_hand", "cond", "mask_hand", "mask_obj", "cond_mask", "touch"]


def load_meta(root, instance, v):
    with open(os.path.join(root, "data_pose_norm", instance,
                           f"{instance}_f{v:03d}_meta.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--num_groups", type=int, default=48)
    ap.add_argument("--views", default="3,6,9,12,15,18,21,0",
                    help="stream order; the LAST view is the ref frame for metrics")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg_strength", type=float, default=5.0)
    ap.add_argument("--cfg_interval", type=float, nargs=2, default=[0.5, 1.0])
    ap.add_argument("--rescale_t", type=float, default=3.0)
    ap.add_argument("--prior", choices=["self", "none"], default="self")
    ap.add_argument("--ss_enc_ckpt", default="step0300000")
    ap.add_argument("--mesh_points", type=int, default=10000)
    ap.add_argument("--emd_points", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    views = [int(v) for v in args.views.split(",")]
    ref = views[-1]

    cfg = edict(json.load(open(os.path.join(args.model_dir, "config.json"))))
    dataset = getattr(datasets, cfg.dataset.name)(args.data_dir, **cfg.dataset.args)
    dataset.inference = True
    if hasattr(dataset, "prior_dropout"):
        dataset.prior_dropout = 1.0  # harness builds its own priors; dataset's are unused

    model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()
    state = torch.load(os.path.join(args.model_dir, "ckpts", args.ckpt),
                       map_location="cuda", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model_has_prior = bool(getattr(model, "use_prior", False))
    use_prior = model_has_prior and args.prior == "self"

    trainer_args = dict(cfg.trainer.args)
    trainer_args["batch_size_per_gpu"] = 1
    trainer_args["fp16_mode"] = None
    trainer = DiagnosticTrainer({"denoiser": model}, dataset, dataset,
                                output_dir=os.path.join(os.path.dirname(args.output),
                                                        "_trainer_scratch_stream"),
                                load_dir=None, step=None, **trainer_args)
    sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=trainer.sigma_min)

    # frozen VAE encoder for the prior latent (same VAE as the dataset latents)
    ss_enc = None
    if use_prior:
        vcfg = json.load(open(os.path.join(trainer_args["ss_dec_path"], "config.json")))
        enc = getattr(models, vcfg["models"]["encoder"]["name"])(**vcfg["models"]["encoder"]["args"])
        enc.load_state_dict(torch.load(
            os.path.join(trainer_args["ss_dec_path"], "ckpts", f"encoder_{args.ss_enc_ckpt}.pt"),
            map_location="cpu", weights_only=True))
        ss_enc = enc.cuda().eval()
        for p in ss_enc.parameters():
            p.requires_grad_(False)

    groups = list(dataset.instances[:args.num_groups])
    K = len(views)
    print(f"{len(groups)} groups, stream order {views} (ref {ref}), steps={args.steps}, "
          f"model_has_prior={model_has_prior}, use_prior={use_prior}")

    acc = {}
    def push(arm, m):
        for k, v in m.items():
            acc.setdefault(arm, {}).setdefault(k, []).append(
                v.cpu() if torch.is_tensor(v) else torch.tensor(float(v)))

    def run_view(data, prior_np, gen):
        """One forward for one view; prior_np is a [64^3] SDF in this view's grid or None."""
        d = {k: v.cuda() for k, v in data.items()}
        x_0 = d.pop("x_0")
        cond_args = trainer.get_inference_cond(**d)
        pos, neg = cond_args["cond"], cond_args["neg_cond"]
        if model_has_prior:
            if prior_np is not None:
                with torch.no_grad():
                    z = ss_enc(torch.from_numpy(np.ascontiguousarray(prior_np))
                               [None, None].float().cuda())
                keep = torch.ones(1, device="cuda")
            else:
                z = torch.zeros(1, cfg.models.denoiser.args.in_channels,
                                16, 16, 16, device="cuda")
                keep = torch.zeros(1, device="cuda")
            pos["x0_prior"], pos["prior_keep"] = z * keep.view(-1, 1, 1, 1, 1), keep
            neg["x0_prior"], neg["prior_keep"] = torch.zeros_like(z), torch.zeros_like(keep)
        noise = torch.randn(x_0.shape, generator=gen, device=x_0.device, dtype=x_0.dtype)
        with torch.no_grad():
            z_out = sampler.sample(model, noise, cond=pos, neg_cond=neg,
                                   steps=args.steps, rescale_t=args.rescale_t,
                                   cfg_strength=args.cfg_strength,
                                   cfg_interval=tuple(args.cfg_interval),
                                   verbose=False).samples
            sdf = trainer.ss_dec(z_out).float()
        return x_0, pos, sdf  # sdf [1,1,64,64,64] in this view's grid

    n_done = 0
    for gi, (root, instance) in enumerate(groups):
        try:
            packs, metas = {}, {}
            for v in views:
                dataset.force_view = v
                packs[v] = dataset.get_instance(root, instance)
                metas[v] = load_meta(root, instance, v)
        except Exception as e:
            print(f"[skip] {instance}: {type(e).__name__}: {e}")
            continue
        finally:
            dataset.force_view = None

        gen = torch.Generator(device="cuda").manual_seed(args.seed * 9973 + gi)
        datas = {v: {k: packs[v][k][None] for k in TENSOR_KEYS} for v in views}

        # ---- streamed pass (prior from own previous outputs) ----
        outputs = {}     # view -> sdf np in own grid
        traj = []        # per-frame metrics of the direct output (own grid vs own GT)
        for t, v in enumerate(views):
            if use_prior and t > 0:
                warped_prev = [warp_sdf(outputs[pv], metas[pv], metas[v])
                               for pv in views[:t]]
                prior_np = (np.median(np.stack(warped_prev), axis=0)
                            if len(warped_prev) > 1 else warped_prev[0])
            else:
                prior_np = None
            x_0v, posv, sdfv = run_view(datas[v], prior_np, gen)
            outputs[v] = sdfv[0, 0].cpu().numpy()
            with torch.no_grad():
                gt_v = trainer.ss_dec(x_0v).float()
                hand_v = trainer.ss_dec(posv["x0_hand"]).float()
            traj.append(sdf_metrics(sdfv, hand_v, posv["touch"], gt_v))
            if v == ref:
                gt_ref, hand_ref, touch_ref = gt_v, hand_v, posv["touch"]

        # ---- no-prior pass on every view (single-view mode / ring-buffer baseline) ----
        gen2 = torch.Generator(device="cuda").manual_seed(args.seed * 9973 + gi)
        outputs_np_ = {}
        traj_np = []
        for v in views:
            x_0v, posv, sdfv = run_view(datas[v], None, gen2)
            outputs_np_[v] = sdfv[0, 0].cpu().numpy()
            with torch.no_grad():
                gt_v = trainer.ss_dec(x_0v).float()
                hand_v = trainer.ss_dec(posv["x0_hand"]).float()
            # no-prior twin of each frame: the per-frame trajectory delta
            # (frame_t minus frame_t_noprior) isolates the integration gain
            # from per-view difficulty (views are scored in their own grids).
            traj_np.append(sdf_metrics(sdfv, hand_v, posv["touch"], gt_v))

        def to_ref(d):
            return {v: (d[v] if v == ref else warp_sdf(d[v], metas[v], metas[ref]))
                    for v in d}
        w_stream, w_plain = to_ref(outputs), to_ref(outputs_np_)

        fused = {
            "single": w_plain[ref],
            "stream_final": w_stream[ref],
            "stream_median": np.median(np.stack([w_stream[v] for v in views]), axis=0),
            "ringbuffer_median": np.median(np.stack([w_plain[v] for v in views]), axis=0),
        }

        rng_mesh = np.random.default_rng(args.seed * 31337 + gi)
        gt_np = gt_ref[0, 0].cpu().numpy()
        for arm, sdf_np in fused.items():
            tsr = torch.from_numpy(np.ascontiguousarray(sdf_np))[None, None].float().cuda()
            push(arm, sdf_metrics(tsr, hand_ref, touch_ref, gt_ref))
            m = paired_mesh_metrics(sdf_np, gt_np, rng_mesh,
                                    n_points=args.mesh_points, emd_points=args.emd_points)
            if m is not None:
                push(arm, {k2: torch.tensor(v) for k2, v in m.items()})
        for t in range(K):
            push(f"frame_{t+1}", traj[t])
            push(f"frame_{t+1}_noprior", traj_np[t])
        push("gt_floor", sdf_metrics(gt_ref, hand_ref, touch_ref, gt_ref))

        n_done += 1
        print(f"  ... group {n_done}/{len(groups)} ({instance})")
        partial = {"meta": {"groups_done": n_done}}
        for a in acc:
            partial[a] = {k2: {"mean": float(torch.stack(vv).mean())}
                          for k2, vv in acc[a].items()}
        with open(args.output + ".partial", "w") as f:
            json.dump(partial, f, indent=2)

    results = {"meta": {"model_dir": args.model_dir, "ckpt": args.ckpt,
                        "num_groups": n_done, "views": views, "ref": ref,
                        "steps": args.steps, "prior": args.prior,
                        "model_has_prior": model_has_prior,
                        "cfg_strength": args.cfg_strength,
                        "cfg_interval": args.cfg_interval,
                        "rescale_t": args.rescale_t, "seed": args.seed}}
    for a in acc:
        results[a] = {k2: {"mean": float(torch.stack(vv).mean()),
                           "std": float(torch.stack(vv).std())}
                      for k2, vv in acc[a].items()}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
