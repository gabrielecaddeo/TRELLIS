# Prompt for a fresh session (copy-paste from here down)

Read ICRA_PLAN.md and EVAL_GUIDANCE.md §7 (LIVE results ledger, §7.1–§7.19)
before doing anything. TEACHER_RETRAIN.md is historical background only.

State as of 2026-08-26 ~evening:
- Teacher FROZEN at EMA 52000; ALL its results final (dex §7.2, Pareto §7.6,
  full-set n=351 fusion tables at 8+25 steps §7.13/§7.15). Never resume it;
  53k/54k ckpts are unevaluated leftovers.
- **Paper's student (decided §7.16): A@81k EMA**
  (`outputs/distill_teacherv2/ckpts/denoiser_ema0.9999_step0081000.pt`);
  physics beats teacher (contact excess 1.70e-3 vs 3.31e-3 @8 steps), guidance
  fully absorbed. Copy-init B = "2-3× faster convergence at equal compute"
  ablation (§7.16).
- **THESIS RESULT (§7.17): fused A@81k (median K8, 8 images, one batched
  forward) BEATS the single-view teacher** — IoU 0.642/CD 0.0466/F.02 0.646 vs
  0.607/0.0501/0.606, paired. Deployment mode: unguided, 8 steps, bf16
  (== fp32 quality; 0.52 s/frame batch-1 H200).
- Multi-view P0–P2 DONE; fusion arm = MEDIAN (ablation ladder §7.14);
  consistency guidance helps single view only, never stack with fusion (§7.13).
- **Visual silhouette loss (user idea) VALIDATED + implemented (§7.18)**:
  presence+carving hinges in distillation.py `_add_visual_losses` (per-view
  calibration, erode-presence/dilate-carving margins, min-mask gate);
  validator = tools/validate_mask_column_correspondence.py.
- **P4 recursive student APPROVED (§7.19)** — run AFTER current tests/trainings.
  Interim streaming demo = ring-buffer median (no training needed).

IN FLIGHT (check `sacct -j 599,610,611,612,613,616` FIRST):
- 599 = A@81k FULL-SET fusion (n=351 @8, L40S) → `outputs/diagnostics/
  mv_fusion_studentA81k_s8_full.json` — confirms §7.17 at scale.
- 610→612 = student-A extension → ~114k (`outputs/distill_teacherv2/`),
  ends ~2026-08-28 morning. 611→613 = student-B → ~64k
  (`outputs/distill_s8mlp4_copyinit/`), same. Graceful wall = exit 138/FAILED.
- 616 = VISUAL-LOSS fine-tune of A@81k, 1×24h → `outputs/distill_visual_ft/`.
  On first read verify log.txt: presence_raw/carving_raw/visual_valid_frac
  present, no NaN, distill_mse stays ≈0.049 (warm start must not jump), and
  presence_raw/carving_raw magnitudes sane (order 1e-2/1e-4, not 10×).

Tasks, in order:
1. Read 599 → append full-set A@81k fusion numbers to §7.17.
2. When 616 ends (~Thu am): paired a/b (25/8/4, 3 arms, `--data_seed 1337`) +
   48-group fusion (`tools/multiview_fusion_eval.sbatch`, L40S) on its final
   EMA. When 612/613 end (~Fri): same for A@~114k and B@~64k.
3. **TRIPLE COMPARISON** → §7.20: A@81k vs A+visual-ft vs A@114k (and B@64k):
   separates visual-loss gain from more-training gain. Prediction: carving cuts
   CD/EMD outliers. Re-pick the final ckpt if any beats A@81k; rerun full-set
   fusion for the winner only (pattern of job 599).
4. P3 frontier assembly: extend tools/bench_latency.py with batch-K (K views in
   ONE forward, bf16 autocast, K∈{1,2,4,8}) on H200 → the claim "8-view student
   batch < 1 teacher forward" needs this number. Then the final capacity ×
   steps × views Pareto table; name the deployment operating point.
5. Dex rows for the final student (generalize inference-repo
   teacher_v2_port/convert_teacher_v2.py for the student arch first; then the
   486→487 chain pattern, canonical ICP flags).
6. **P4 build (user-approved)**: follow the §7.19 recipe exactly — precompute
   pass (frozen best student over training views, subsampled), zero-init prior
   channel, warm-start, prior dropout 30%, curriculum. Ask the user before the
   precompute+training launch (multi-day GPU).
7. Keep EVAL_GUIDANCE §7 / ICRA_PLAN / NEXT_SESSION_PROMPT / memory updated
   after each result; commit at phase boundaries.

Open user decisions: rig access timeline (real-time demo = ICRA centerpiece;
without it, workshop). Everything else is decided — do not re-litigate.

Hard rules:
- Mesh evals ALWAYS use ICP with the canonical flag set (memory trellis-eval-icp).
- Cross-model a/b runs ALWAYS pass `--data_seed 1337`; name teacher ckpt
  explicitly (never `latest_ema` on the teacher dir — grabs the 54k leftover).
- Never evaluate an EMA ckpt below ~30k steps unless copy-init/warm-start-seeded.
- GPU via sbatch only; login node OOMs. TRAINING → gpu-h200 (4 jobs/96 CPUs/24h,
  self-chaining sbatch + STOP_CHAIN); EVALS → gpu-l40s
  (`sbatch --partition=gpu-l40s <sbatch>`, 2-job quota).
- Prefer SLURM --dependency chains over session watchers for must-run steps.
- Ask the user before launching anything ≥ multi-day GPU cost.
- Git: commit locally on `main`; NO GitHub credentials here — the user pushes.
  Inference repo (/projects/gcaddeo/inference/TRELLIS): local main is ahead of
  origin — push to a branch, never its main.
