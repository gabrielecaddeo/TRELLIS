# ICRA plan: real-time physics-aware in-hand reconstruction (updated 2026-08-24)

**Context.** The hand-conditioned + contact/penetration-loss method is ALREADY
PUBLISHED (prior paper). This campaign fixed its implementation and retrained the
teacher (teacher_v2, frozen at EMA step 52000 — TEACHER_RETRAIN.md §8.9). The fixes
themselves are not a contribution. The running results ledger is EVAL_GUIDANCE.md §7.

## Paper framing (revised 2026-08-24, agreed with the user)

**The compute-quality frontier of physics-aware in-hand reconstruction: capacity,
steps, guidance, and views.** Thesis: *distill capacity away for speed; recover
quality from views.* The multi-view work is MERGED with the distillation
contribution, not adjacent to it.

Findings that anchor each axis (details + tables in EVAL_GUIDANCE.md §7):
1. **Steps are nearly free** (§7.1, §7.2): unguided teacher is step-invariant to 4
   steps on held-out synthetic AND real DexYCB (8 steps == 25 to 3 decimals).
2. **Physics guidance is transient scaffolding** (§7.1, §7.6, §7.8): it helps any
   undertrained model (teacher@29k, student@32k) and is absorbed by training on
   every sufficiently-trained one; always harmful at low NFE (large-dt corrections
   on noisy x̂0). Deployment: unguided, both models.
3. **Capacity costs geometry, and the student is REQUIRED for real time** (§7.8):
   student 220M vs teacher 757M; H200 bf16 flow latency student@8 = 0.52 s (~2 Hz),
   teacher@8 = 1.59 s. Student@48k matches teacher physics; geometry gap remains
   (IoU 0.55 vs 0.68).
4. **Views buy the quality back** (§7.10, smoke n=2; full runs 519–522 pending):
   fused student (K=4, median) beat the single-view teacher on every geometry
   metric at a fraction of the compute. Prediction to test at scale: the
   measurement axis (views) does NOT absorb, unlike the prior axis (physics
   guidance) — the paper's scientific claim.

## Multi-view phases (P0–P4)

Dataset decision (user, 2026-08-24): dex lacks usable multi-view groups; use the
24-view synthetic grasps. Held-out `datasets_split/Leap_Hand_test` (319 groups) +
`Hands_test` (32) are UNCONTAMINATED multi-view sets. Machinery:
`tools/multiview_warp.py` (validated similarity warps: x_view = s·Rᵀ·x_canon + t,
voxel centers, SDF values scaled by s_dst/s_src; GT→GT IoU 0.94–0.97, |diff| ~1/10
voxel) and `tools/multiview_fusion_eval.py` (+ `force_view` dataset hook).

- **P0 — warp module + GT validation: DONE** (§7.9).
- **P1 — fusion baseline: RUNNING** (jobs 519–522: {student48k, teacher52k} ×
  {8, 25} steps × 48 groups; K ∈ {2,4,8}, arms mean/median/vismean vs single).
  Smoke verdict §7.10; vismean is the worst arm — drop or rework.
- **P2 — cross-view consistency guidance**: joint K-view sampling on the
  `sample_guided_v2` skeleton, energy = pairwise warped-SDF disagreement on x̂0.
  Run on both models; the teacher run tests measurement-vs-prior absorption.
  25-step mode by default (low-NFE guidance is known-harmful).
- **P3 — frontier assembly**: batch-K latency (one batched forward = the
  deployment mode; extend bench_latency), the capacity × steps × views Pareto
  table, pick + name the deployment operating point. bf16 QUALITY a/b before
  quoting bf16 latency in the paper.
- **P4 (stretch, gate on P1/P2 + deadline)** — recursive fusion distilled into the
  student: warped previous-reconstruction as an extra conditioning channel
  (architecture pattern exists: `input_layer_x0h`), trained with pose-jittered
  priors → a per-frame filter at student@8 cost.

## Training state

- **Teacher**: FROZEN at EMA 52000 (`denoiser_teacher_v2_FROZEN.pt`); ledger
  closed; never resume; 53k/54k ckpts are unevaluated leftovers.
- **Student (8 blocks, mlp_ratio 2, from scratch)**: chain 492/493/498 done →
  step 48000; extension 512/513 running → ~82k. Physics gap to teacher closed at
  48k; geometry improving steeply (§7.8). Final ckpt choice for the paper: best of
  the extension by paired a/b.
- **Copy-init ablation (8 blocks, mlp_ratio 4, teacher-layer init)**: chain
  514/516 running → `outputs/distill_s8mlp4_copyinit/`. Started at distill_mse
  0.60 vs 1.05 from scratch. Read: does copy-init reach the same quality in far
  fewer steps / end higher?
- Distillation machinery: trainer physics block is a faithful stage-2 port
  (audited+fixed 2026-08-20, smoke-passed job 488).

## Execution order (remaining)

1. Read P1 full runs (519–522) → EVAL_GUIDANCE §7.11; decide K and fusion arm.
2. P2 consistency-guidance sampler + runs (both models).
3. Read extension (~82k) + copy-init ablation checkpoints with the paired a/b
   (`--data_seed 1337`); pick the paper's student.
4. P3: batch-K latency, bf16 quality a/b, assemble the frontier table.
5. Real-capture confirmation of the chosen operating point: student dex rows
   (convert via parity-tested port; ICP canonical flags) + rig latency demo
   (user to arrange rig access — without it, target a workshop).
6. P4 if gates pass and time allows.

## Open questions for the user
- Rig access / camera setup timeline (the latency demo on live captures is the
  centerpiece for ICRA vs workshop).
- P4 in or out of scope for this deadline.
- Extend the 8×2 student beyond ~82k if the paired a/b still shows slope?

## Hard rules (unchanged)
- Mesh evals ALWAYS use ICP with the canonical flag set (memory trellis-eval-icp).
- Cross-model a/b runs ALWAYS pass `--data_seed 1337` (§7.6 pairing gotcha) and
  name the frozen ckpts explicitly (never `latest_ema` on the teacher dir).
- GPU work only via sbatch on gpu-h200 (24h cap)/cpu; login node OOMs; prefer
  SLURM --dependency chains; pre-queue successors; STOP_CHAIN to stop.
- Ask the user before anything ≥ multi-day GPU cost.

---
Historical note: the 2026-08-20 version of this plan predicted guidance re-entry
on the capacity axis as the organizing result. Measured outcome (§7.5–§7.8):
re-entry is a *training-progress* effect, absorbed as the student converges — the
organizing result is now the four-axis frontier above.
