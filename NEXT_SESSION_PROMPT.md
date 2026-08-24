# Prompt for a fresh session (copy-paste from here down)

Read ICRA_PLAN.md, EVAL_GUIDANCE.md §3–§4, and TEACHER_RETRAIN.md §8.9 (entries
"Boundary 4", "TEACHER FROZEN") in this repo before doing anything.

State as of 2026-08-20 ~22:00:
- The teacher is FROZEN at EMA step 52000:
  outputs/teacher_v2_stage2_physics/denoiser_teacher_v2_FROZEN.pt (see
  FROZEN_TEACHER.txt there; STOP_CHAIN is set; ckpts 53k/54k are unevaluated
  leftovers — never use them). Do NOT resume stage-2 training.
- Jobs 486 (GPU: converts 52000 into the inference repo's teacher_v2_stage2/
  pipeline and regenerates dex meshes into
  /projects/gcaddeo/inference/TRELLIS/meshes_results_marching_cubes_teacher_v2_52k)
  and 487 (CPU, afterany:486: canonical ICP eval →
  summary_teacher_v2_52k_dex_total_total_icp.json) were queued at ~21:30 Aug 20.
  Check sacct -j 486,487 first; on failure read slurm-dex_t2-486.out /
  slurm-dexcm_t2_52k-487.out. Compare 487's output against
  summary_ours_ablation_dex_total_total_icp.json (old teacher, job 475: CD²
  0.0458/0.0181 mean/median) — this is the paper's old-vs-new table.
- The distillation trainer (trellis/trainers/flow_matching/distillation.py) was
  audited and FIXED on 2026-08-20: its physics block is now a faithful port of the
  stage-2 (fixed) physics. Config:
  configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill_teacherv2.json
  (frozen-teacher paths, λ_ni 30 relative+margined warmup 10k, λ_contact 1.0
  relative, fp32 student, 8 blocks / mlp_ratio 2). It has NOT been run.

Your task, in order (from ICRA_PLAN.md "Execution order"):
1. Read 486/487 results; write the frozen-teacher dex table into EVAL_GUIDANCE.md.
2. Smoke-test the fixed distillation trainer: write an sbatch modeled on
   tools/train_teacher_v2.sbatch (same conda/env/PYTHONPATH pattern; train.py with
   the distill_teacherv2 config; 1 GPU, ~1h, output dir
   outputs/distill_teacherv2_smoke). Verify in the log: teacher loads from the
   FROZEN ckpt (strict), ni_floor/contact_floor keys appear, loss magnitudes sane
   (mse ~0.15–0.35 territory, physics terms small), distill_mse decreasing, no NaN.
3. Report smoke results and ASK THE USER before launching the full distillation
   (do not launch it on your own), proposing chained 24h segments per the
   train_teacher_v2.sbatch pattern with a fresh output dir.
4. In parallel with any GPU waits, start ICRA_PLAN.md item 4: add a --steps sweep
   to tools/ab_eval_guidance.py runs (8 and 4 steps) and produce the teacher rows
   of the capacity×steps×guidance Pareto table on the frozen teacher.

Hard rules:
- Mesh evals ALWAYS use ICP with the canonical flag set (memory trellis-eval-icp).
- GPU work only via sbatch files on gpu-h200 (24h cap) / cpu partitions; login node
  OOMs on anything sizable; prefer SLURM --dependency chains over session watchers;
  pre-queue successors for long chains, cancel on a STOP decision.
- Ask the user before launching distillation or anything ≥ multi-day GPU cost.
- The stopping-rule ledger is CLOSED; do not relaunch teacher training.
