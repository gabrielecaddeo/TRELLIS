# ICRA plan: real-time physics-aware in-hand reconstruction (2026-08-20)

**Context.** The hand-conditioned + contact/penetration-loss method is ALREADY
PUBLISHED (prior paper). This campaign fixed its implementation and retrained the
teacher (teacher_v2, frozen at EMA step 52000 — TEACHER_RETRAIN.md §8.9). The fixes
themselves are not a contribution. What remains novel from this campaign:

1. **The guidance-absorption finding** (measured, 5 checkpoints, pre-registered
   metric): inference-time physics guidance (complete OC-Flow, discrete adjoint)
   goes from −31% excess contact at ckpt 29k to +45% (harmful) at 52k as
   training-time physics losses do their work. Guidance is a *substitute* for
   training/capacity, not a supplement. Data: `ab_guidance_2arm_final{,2,3,4}.json`,
   `ab_guidance_4arm.json`.
2. **Nothing architectural.** Honest position agreed with the user 2026-08-20.

**Paper framing (agreed): deployment/systems study.**
"Real-time physics-aware in-hand reconstruction: distillation, guidance, and the
compute-quality frontier."
- (i) Real-time demonstration on hardware (user to arrange rig access; the latency
  test on live captures is the centerpiece — without it, target a workshop instead).
- (ii) The capacity × steps × guidance Pareto frontier, on held-out synthetic (a/b
  harness) AND real DexYCB captures (dex benchmark):
  arms = teacher-unguided / student-unguided / student+greedy-v2 / student+OC-Flow,
  each at 25 steps and reduced steps (8, 4). Prediction from the absorption curve:
  the weakened student re-enters the regime where guidance helps — measuring WHERE
  it re-enters is the paper's organizing result.
- (iii) The absorption analysis (already measured) as the explaining insight.

**Assets already in place.**
- Frozen teacher EMA 52000 + provenance (`FROZEN_TEACHER.txt`).
- Inference-repo deployment, parity-verified bit-exact; dex benchmark harness with
  canonical ICP flags (memory: trellis-eval-icp); teacher_v2@37k already beats the
  old teacher on real captures: CD² 0.0349/0.0119 vs 0.0458/0.0181, n=989/992.
  52k refresh running (jobs 486/487).
- a/b harness (`tools/ab_eval_guidance.py`) works for ANY checkpoint incl. students.
- Distillation trainer + config: audited and FIXED 2026-08-20 (see EVAL_GUIDANCE.md
  §3 note; trainer physics now identical to stage-2's; config
  `..._distill_teacherv2.json` points at the frozen teacher, λ 30/1.0 relative,
  fp32 student). Needs a GPU smoke run before a real launch.

**Execution order (next session).**
1. Read 486/487 (frozen-teacher dex numbers → final old-vs-new table).
2. GPU smoke-test the fixed distillation trainer (few hundred steps, 1 GPU:
   watch ni_floor/contact_floor appear in the log, loss magnitudes sane,
   distill_mse decreasing). Template: tools/train_teacher_v2.sbatch pattern with
   the distill config; a dedicated sbatch does not exist yet.
3. ASK USER, then launch full distillation (24h chained segments, 2 GPUs).
4. While distilling: extend the a/b harness with a --steps sweep (8, 4) and run
   the teacher rows of the Pareto table.
5. Student boundary evals with the SAME harness (unguided + guided_v2 + oc_flow):
   does guidance help the student? At which capacity/steps does it re-enter?
6. Convert best student via the parity-tested port (teacher_v2_port/) and run the
   dex benchmark rows; latency measurements (steps × blocks) on H200 + the rig.
7. Optional step-distillation (consistency/shortcut style) if time allows —
   the only genuinely new machinery in the plan.

**Open questions for the user.**
- Rig access / camera setup timeline for (i).
- Student design: keep 8 blocks / mlp_ratio 2 (prior runs) or sweep capacity too.
