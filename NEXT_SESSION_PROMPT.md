# Prompt for a fresh session (copy-paste from here down)

Read ICRA_PLAN.md (updated 2026-08-24: merged framing + multi-view phases) and
EVAL_GUIDANCE.md §7 (the running results ledger, §7.1–§7.10) before doing anything.
TEACHER_RETRAIN.md §8.9 is historical background (teacher frozen, ledger closed).

State as of 2026-08-24 ~evening:
- Teacher FROZEN at EMA 52000 (`outputs/teacher_v2_stage2_physics/
  denoiser_teacher_v2_FROZEN.pt`). Never resume it; 53k/54k ckpts are leftovers.
- Student 8×mlp2 trained to 48k (chain 492/493/498, done); **extension 512→513
  running** to ~82k in `outputs/distill_teacherv2/` (ckpts every 1000).
- **Copy-init ablation 514→516 running** (8 blocks × mlp4, teacher-layer init) in
  `outputs/distill_s8mlp4_copyinit/`.
- **P1 multi-view fusion full runs 519–522 in flight**: {student48k, teacher52k} ×
  {8,25} steps, 48 held-out groups → `outputs/diagnostics/
  mv_fusion_{student48k,teacher52k}_s{8,25}.json` (+ `.partial` safety copies).
  Smoke (§7.10): fused student K=4-median BEAT the single-view teacher on all
  geometry metrics; vismean arm is broken-ish — drop or rework.
- Latency (H200, batch 1): §7.8 fp32 table + §7.10 bf16 (≈2× faster; student@8 =
  0.52 s). bf16 QUALITY not yet verified.

Tasks, in order:
1. `sacct -j 519,520,521,522`; read the four `mv_fusion_*.json` → write §7.11
   (fusion at n=48: does the smoke conclusion hold? pick K + fusion arm). On
   failure read `slurm-mv_*-<id>.out`.
2. Check the training chains (`sacct -j 512,513,514,516`; log.txt line count ≈
   step). When the extension ends (~82k): paired a/b on its final EMA
   (`--data_seed 1337`, steps 25/8/4, arms unguided,guided_v2,oc_flow — pattern of
   jobs 509–511). Same for the copy-init ablation's final ckpt. Compare vs
   student48k rows (§7.8) and teacher (§7.6 table).
3. Build P2 (consistency guidance) per ICRA_PLAN.md: new sampler method on the
   `sample_guided_v2` skeleton (`trellis/pipelines/samplers/flow_euler.py`),
   energy = pairwise warped-SDF disagreement of the K x̂0 decodes
   (`tools/multiview_warp.py` has the validated warps — torch port needed for
   in-loop use). Run both models at 25 steps on the same 48 groups; the teacher
   run is the measurement-vs-prior absorption test.
4. bf16 quality a/b (cheap): one a/b run with autocast bf16 vs the fp32 numbers.
5. Keep EVAL_GUIDANCE.md §7 and the memory file up to date after each result.

Hard rules:
- Mesh evals ALWAYS use ICP with the canonical flag set (memory trellis-eval-icp).
- Cross-model a/b runs ALWAYS pass `--data_seed 1337`; name teacher ckpt
  explicitly (never `latest_ema` on the teacher dir — it grabs the 54k leftover).
- GPU work only via sbatch on gpu-h200 (24h cap)/cpu partitions; login node OOMs
  on anything sizable; prefer SLURM --dependency chains over session watchers.
- Ask the user before launching anything ≥ multi-day GPU cost.
- Both repos were committed+pushed 2026-08-24 (train repo `main` = eab364c + this
  session's later work may be uncommitted — check `git status` and propose a
  commit when a phase completes).
