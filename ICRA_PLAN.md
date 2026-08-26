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
4. **Views buy the quality back — MEASURED at scale** (§7.11–§7.15): K8-median
   fusion gives +0.10–0.11 IoU and −26–29% CD to BOTH models; teacher full-set
   (n=351) tables final at 8 AND 25 steps (single IoU 0.60 → K8 0.72). Fusion-arm
   ablation CLOSED: mean < vishand (exact hand-occlusion weights) < hybrid ≈
   **median** — robust vote wins because occluded views' generative completions
   are informative. **The measurement-vs-prior claim is MEASURED (§7.13)**:
   consistency guidance helps the same teacher that physics guidance hurts
   (single-view IoU +0.018; student +0.061); guidance and fusion do NOT stack
   (consensus correlates errors). bf16 == fp32 quality → deployment mode =
   student@8 bf16 = 0.52 s.

## Multi-view phases (P0–P4)

Dataset decision (user, 2026-08-24): dex lacks usable multi-view groups; use the
24-view synthetic grasps. Held-out `datasets_split/Leap_Hand_test` (319 groups) +
`Hands_test` (32) are UNCONTAMINATED multi-view sets. Machinery:
`tools/multiview_warp.py` (validated similarity warps: x_view = s·Rᵀ·x_canon + t,
voxel centers, SDF values scaled by s_dst/s_src; GT→GT IoU 0.94–0.97, |diff| ~1/10
voxel) and `tools/multiview_fusion_eval.py` (+ `force_view` dataset hook).

- **P0 — warp module + GT validation: DONE** (§7.9).
- **P1 — fusion baseline: DONE** (§7.11 n=48 both models; §7.13/§7.15 full-set
  n=351 teacher tables at 8+25 steps; §7.14 fusion-arm ablation closed → median).
- **P2 — consistency guidance: DONE** (§7.13; sampler
  `sample_multiview_consistency` in flow_euler.py; helps single-view output,
  does not stack with fusion; measurement-vs-prior contrast measured).
- **P3 — frontier assembly: NEXT.** Remaining pieces: batch-K latency
  (extend bench_latency: K views in one batched forward, bf16), the final
  capacity × steps × views Pareto table on the CHOSEN student, and the
  deployment operating point. bf16 quality parity already verified (§7.13).
- **P4 — recursive student: APPROVED by the user (2026-08-26), scheduled AFTER
  the current tests/trainings finish.** Full recipe in EVAL_GUIDANCE.md §7.19:
  zero-init prior-latent input channel (input_layer_x0h pattern), warm-start
  from the best student, simulated-streaming data from the 24-view grasps
  (curriculum corrupted-GT → precomputed frozen-student reconstructions, prior
  dropout 30%, pose jitter), losses unchanged, ~4-6 GPU-days, NO base
  re-pretraining. Interim rig demo: ring-buffer median streaming (no training).
- **P5 — visual (silhouette) loss (user idea, 2026-08-26): VALIDATED +
  fine-tune RUNNING** (§7.18): mask↔column correspondence pixel-grade for
  visible objects; presence/carving hinges implemented in the distillation
  trainer; job 616 fine-tunes A@81k 1 segment. THESIS RESULT §7.17: fused A@81k
  beats the single-view teacher on all metrics (IoU 0.642 vs 0.607).

## Training state

- **Teacher**: FROZEN at EMA 52000 (`denoiser_teacher_v2_FROZEN.pt`); ledger
  closed; never resume; 53k/54k ckpts are unevaluated leftovers.
- **Student A (8 blocks, mlp_ratio 2, from scratch)**: 492/493/498 → 48k; ext
  segment 4 (512) → 65k (distill 0.0493, slope flattening); segment 5 (513)
  running → ~82k, ends 2026-08-26 ~10:00. Physics gap closed at 48k (§7.8).
- **Student B (copy-init, 8 blocks, mlp_ratio 4, teacher layers 0,3,...,21)**:
  segment 1 (514) → 16k with distill 0.052/target 0.193 = from-scratch's 48–65k
  level (~3× compute win, §7.15; capacity confound: 1.5× params of A). Segment 2
  (516) running → ~32k, ends 2026-08-26 ~10:10. LIKELY the paper's student if
  the paired a/b confirms the loss-side advantage.
- Distillation machinery: trainer physics block is a faithful stage-2 port
  (audited+fixed 2026-08-20, smoke-passed job 488).

## Execution order (remaining)

1. DONE 2026-08-26: student decision = A@81k (§7.16); A81k paired fusion
   (§7.17, thesis passes); visual loss validated+implemented+launched (§7.18).
2. 2026-08-27/28: read visual-ft (616) vs A-extension (610/612 → ~114k) vs
   B-extension (611/613 → ~64k) — triple comparison, paired a/b + fusion.
3. Full-set (n=351) fusion @8 for the final chosen ckpt (599 = A@81k full-set,
   running; redo only if a later ckpt replaces A@81k).
4. P3: batch-K bf16 latency for the chosen student; assemble the final
   capacity × steps × views frontier table; name the operating point.
5. Real-capture confirmation: chosen student dex rows (convert via the
   parity-tested port teacher_v2_port/convert_teacher_v2.py — NOTE it must be
   generalized for the student arch; ICP canonical flags) + rig latency demo
   (user to arrange rig access — without it, target a workshop).
6. P4 / learned-aggregator if time allows (user decision).

## Open questions for the user
- Rig access / camera setup timeline (the latency demo on live captures is the
  centerpiece for ICRA vs workshop).
- (P4: DECIDED — in scope, after current tests/trainings; see §7.19.)

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
