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

DONE 2026-08-26 evening session:
- 599 COMPLETED → §7.17 full-set table appended: fused A@81k beats teacher
  single on IoU/F@0.02/NC/contact at n=351, but CD TIES (0.0597 vs 0.0586) and
  EMD is worse — the n=48 across-the-board win was subset luck on distance
  metrics. Paper phrasing recorded in §7.17.
- 616 early-log checklist PASSED (§7.19 note): 0 NaN @100 steps, distill_mse
  0.048–0.050 (no warm-start jump), valid_frac 0.94–0.97; presence_raw ~2e-4 /
  carving_raw ~1e-6 = tiny (near-no-op risk stands). 610 resumed A@81000,
  611 resumed B@32000 clean. NOTE: extension configs log with i_log=500
  (buffered flush every ~40 min) — a silent log.txt is NORMAL for them.
- bench_latency.py EXTENDED with --batch_k (per-K cond encode / flow / decode,
  K views in one forward, pattern of the fusion eval's stacked batch).
- multiview_fusion_eval.py now accepts --ckpt latest_ema (records resolved
  name; only for FINISHED single-purpose student dirs, never the teacher dir).
- Inference repo (NOT committed there yet — see rules): convert_teacher_v2.py
  generalized (--train_config/--pipeline_dir/--name assembles a student
  pipeline dir, decoders symlinked, use_checkpoint off); inference_dex uses
  DEX_PIPELINE_DIR env. New tools/dex_student.sbatch +
  tools/dex_eval_cm_student.sbatch (convert→infer→canonical-ICP, 486→487
  pattern, tags e.g. a81k).

IN FLIGHT / PRE-QUEUED (check `squeue -u gcaddeo` + `sacct` FIRST):
- 610→612 = student-A extension → ~114k (`outputs/distill_teacherv2/`,
  segment 7/7 = final). 611→613 = student-B → ~64k
  (`outputs/distill_s8mlp4_copyinit/`, segment 4/4 = final). Graceful wall =
  exit 138/FAILED. 616 = visual-ft, ends Thu ~17:00.
- Pre-queued on dependencies (all latest_ema, L40S evals):
  - after 616: 618/619/620 = ab_vft 25/8/4, 621 = mv_vft_8, and 617 =
    batch-K bf16 latency bench (H200, teacher+A-arch+B-arch, steps 25/8/4 ×
    K 1/2/4/8 → `latency_h200_batchK_bf16.json`).
  - after 612: 622/623/624 = ab_aext 25/8/4, 625 = mv_aext_8.
  - after 613: 626 = ab_bext_8, 627 = mv_bext_8. (B's 25/4-step a/b NOT yet
    queued — l40s MaxSubmitPU=10 was full; submit Thu once vft suite drains:
    same pattern, `--teacher_dir outputs/distill_s8mlp4_copyinit`.)
  Outputs: `ab_guidance_student{Avft,Aext,Bext}_ds_steps*.json`,
  `mv_fusion_student{Avft,Aext,Bext}_s8.json`.

Tasks, in order:
1. **TRIPLE COMPARISON** → new §7.20: A@81k (§7.16/§7.17) vs A+visual-ft
   (Avft) vs A@~114k (Aext), plus Bext: separates visual-loss gain from
   more-training gain. Prediction: carving cuts CD/EMD outliers (§7.17's
   full-set CD/EMD tie is exactly the target). Watch the near-no-op risk
   (§7.19): if Avft ≈ A@81k, record the absorption interpretation honestly.
   Re-pick the final ckpt if any beats A@81k; rerun full-set fusion for the
   winner only (pattern of job 599: mv_fusion sbatch, n=351, L40S, ~1h45).
2. P3 frontier assembly once 617 lands: verify "8-view student batch < 1
   teacher forward" from `latency_h200_batchK_bf16.json`; assemble the final
   capacity × steps × views Pareto table; name the deployment operating point.
3. Dex rows for the FINAL student: `sbatch tools/dex_student.sbatch
   <train_dir> <ckpt> <tag> 8` then `sbatch --dependency=afterany:<id>
   tools/dex_eval_cm_student.sbatch <tag> 8` (25 too if wanted for the table).
4. **P4 build (user-approved)**: follow the §7.19 recipe exactly — precompute
   pass (frozen best student over training views, subsampled), zero-init prior
   channel, warm-start, prior dropout 30%, curriculum. Ask the user before the
   precompute+training launch (multi-day GPU).
5. Keep EVAL_GUIDANCE §7 / ICRA_PLAN / NEXT_SESSION_PROMPT / memory updated
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
