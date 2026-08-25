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
- IN FLIGHT (check `sacct -j 513,516,532,533,534` FIRST):
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

Tasks, in order (details in ICRA_PLAN.md "Execution order"):
1. Read 532/533/534 → §7.16: does copy-init's loss advantage (distill 0.052@16k
   = from-scratch's 48–65k level) show up in QUALITY? Compare against §7.6/§7.8
   rows (teacher single 0.607 IoU @8; student-A 48k 0.553).
2. After the chains end (Wed ~10:00): paired a/b on A's ~82k EMA and B's ~32k
   EMA (B's EMA is copy-init-seeded → residue benign) at 25/8/4, 3 arms,
   `--data_seed 1337` → PICK THE PAPER'S STUDENT, record the decision.
3. Full-set (n=351) fusion for the chosen student:
   `sbatch --partition=gpu-l40s tools/multiview_fusion_eval.sbatch --model_dir
   <dir> --ckpt <ema> --num_groups 351 --steps 8 --output ...` (pattern of jobs
   529/528).
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
