# Prompt for a fresh session (copy-paste from here down)

Read ICRA_PLAN.md (updated 2026-08-25: framing, phase statuses, execution order)
and EVAL_GUIDANCE.md §7 (the LIVE results ledger, §7.1–§7.15) before doing
anything. TEACHER_RETRAIN.md is historical background only.

State as of 2026-08-25 ~10:30:
- Teacher FROZEN at EMA 52000 (`outputs/teacher_v2_stage2_physics/
  denoiser_teacher_v2_FROZEN.pt`). Never resume; 53k/54k ckpts are leftovers.
  Its results are COMPLETE: dex table (§7.2), Pareto rows (§7.6), full-set
  (n=351) fusion tables at 8+25 steps (§7.13/§7.15).
- Multi-view phases P0–P2 DONE (§7.9–§7.15): warps validated
  (`tools/multiview_warp.py`), fusion harness (`tools/multiview_fusion_eval.py`),
  fusion arm = MEDIAN (ablation closed: mean < vishand < hybrid ≈ median),
  consistency guidance measured (`sample_multiview_consistency` in
  flow_euler.py; helps single view, does NOT stack with fusion).
- bf16 == fp32 quality (§7.13); deployment mode = student@8 bf16 (0.52 s H200).
- STUDENT DECIDED (§7.16, 2026-08-26): **A@81k EMA** = the paper's student
  (`outputs/distill_teacherv2/ckpts/denoiser_ema0.9999_step0081000.pt`); physics
  beats teacher, guidance absorbed. B (copy-init) = equal-compute-wins ablation.
  Both chains ENDED; no training running. Open user decisions: extend A and/or B.
- IN FLIGHT (check `sacct -j 599,600` FIRST; the older list below is DONE):
  - 513 = student-A (8×2) extension segment 5 → ~82k in
    `outputs/distill_teacherv2/`, wall-ends 2026-08-26 ~09:50 (exit 138/FAILED
    in sacct = normal graceful wall kill). NO successor queued after it.
  - 516 = student-B (copy-init 8×4) segment 2 → ~32k in
    `outputs/distill_s8mlp4_copyinit/`, wall-ends 2026-08-26 ~10:10. Last
    segment; no successor.
  - 532/533 = copy-init 16k RAW ckpt paired a/b @8/@25 (L40S) →
    `outputs/diagnostics/ab_guidance_copyinit16kraw_ds_steps{8,25}.json`
  - 534 = student-A 65k EMA paired a/b @8 (H200) →
    `outputs/diagnostics/ab_guidance_student65k_ds_steps8.json`

NEW in flight 2026-08-26 evening (check `sacct -j 599,610,611,612,613,616`):
- 599 = A81k full-set fusion (n=351 @8, L40S) → `mv_fusion_studentA81k_s8_full.json`
- 610→612 = student-A extension segments 6-7 → ~114k (ends 2026-08-28 ~morning)
- 611→613 = student-B extension segments 3-4 → ~64k (same)
- 616 = VISUAL-LOSS fine-tune of A@81k (§7.18): silhouette presence+carving
  terms (user idea, validated §7.18), 1×24h, `outputs/distill_visual_ft/`.
  Verify early log: presence_raw/carving_raw/visual_valid_frac keys, no NaN,
  distill_mse stays ~0.049 (warm start must not jump).
DONE 2026-08-26: §7.16 student decision = A@81k; §7.17 THESIS RESULT (fused
A@81k beats single-view teacher: IoU 0.642 vs 0.607, paired); §7.18 visual loss
validated + implemented (`_add_visual_losses` in distillation.py, calibration +
erode/dilate margins; validator: tools/validate_mask_column_correspondence.py).

Tasks, in order (details in ICRA_PLAN.md "Execution order"):
1. When 616 ends (~Thu morning): paired a/b (25/8/4, `--data_seed 1337`) +
   48-group fusion on `outputs/distill_visual_ft/ckpts/<final EMA>`; when
   610/612 and 611/613 end (~Fri): same for A@~114k and B@~64k. TRIPLE
   comparison vs A@81k rows (§7.16/§7.17) separates visual-loss gain from
   more-training gain. Prediction: carving cuts CD/EMD outliers.
2. Re-pick the final ckpt if any of the three beats A@81k; rerun full-set
   fusion only for the winner (pattern of job 599).
4. P3 frontier assembly: extend tools/bench_latency.py with batch-K (K views in
   ONE batched forward, bf16 autocast) on H200; build the final capacity × steps
   × views table; name the deployment operating point.
5. Dex rows for the chosen student: the converter
   (inference repo teacher_v2_port/convert_teacher_v2.py) assumes the TEACHER
   arch — generalize it for the student config before converting; then the
   486→487 chain pattern with canonical ICP flags.
6. Keep EVAL_GUIDANCE.md §7, ICRA_PLAN.md, and the memory file updated after
   each result; commit at phase boundaries (user pushes — no git credentials on
   this machine; see below).

Open user decisions (ask, don't assume): P4 / learned fusion aggregator in or
out; rig access timeline; extend student A past 82k if 534 still shows slope.

Hard rules:
- Mesh evals ALWAYS use ICP with the canonical flag set (memory trellis-eval-icp).
- Cross-model a/b runs ALWAYS pass `--data_seed 1337`; name teacher ckpt
  explicitly (never `latest_ema` on the teacher dir — grabs the 54k leftover).
- Never evaluate an EMA ckpt below ~30k steps unless it was copy-init-seeded
  (random-init residue artifact, §7.6).
- GPU work only via sbatch; login node OOMs on anything sizable. Partition
  routing: TRAINING → gpu-h200 (QOS: 4 jobs / 96 CPUs / 24h wall, self-chaining
  sbatch pattern with STOP_CHAIN sentinel); EVALS → gpu-l40s
  (`sbatch --partition=gpu-l40s <sbatch>`, 2-job quota, validated §7.12).
- Prefer SLURM --dependency chains over session watchers for must-run steps.
- Ask the user before launching anything ≥ multi-day GPU cost.
- Git: work is committed locally on `main` (latest commits include the
  multi-view phase); the machine has NO GitHub credentials — the user pushes.
  Two repos: this one (train) and /projects/gcaddeo/inference/TRELLIS
  (its local main is ahead of origin; push it to a branch, not main — see the
  2026-08-24 conversation note in EVAL_GUIDANCE §7 header context).
