# Prompt for a fresh session (copy-paste from here down)

Read ICRA_PLAN.md and EVAL_GUIDANCE.md §7 (LIVE ledger, now through §7.20)
before doing anything. TEACHER_RETRAIN.md is historical background only.

State as of 2026-08-28 ~22:30 (everything below is COMMITTED locally; user
pushes — train repo main; inference repo local main is ahead of origin, push
to a branch, never its main):
- Teacher FROZEN at EMA 52000; all its results final. Never resume it.
- **FINAL STUDENT (§7.20 triple comparison): A@113k EMA**
  (`outputs/distill_teacherv2/ckpts/denoiser_ema0.9999_step0113000.pt`) —
  beats A@81k / A+vft@16k / B@64k on EVERY metric, single AND fused
  (@8 unguided: IoU 0.633, CD 0.0584, contact excess 9.15e-4 = 3.6× BELOW
  teacher; fused K8: IoU 0.667/CD 0.0430/F.02 0.676). Still climbing —
  stopped by calendar, not convergence.
- **Visual loss verdict (§7.18/§7.20)**: real + more compute-efficient per
  step than plain distillation, gain AMPLIFIES under fusion (+0.010 single →
  +0.020 fused IoU), but plain 2×-steps extension beats it outright → paper
  framing: cheap accelerant / measurement-guidance candidate. Stacked test
  RUNNING (below).
- B@64k: absorption complete (oc_flow hurts now), fused 0.651 > A@81k but
  physics stuck at ~4e-3; stays the §7.16 copy-init ablation. NOTE §7.20
  latency correction: B costs only +3% vs A (mlp width ~free) — latency was
  wrongly used against B in §7.16's open-options text.
- P3 batch-K bench DONE (§7.20): near-linear scaling (throughput-bound at
  batch 1, ~10% peak util — NOT saturation; see corrected wording). Batched
  K8@8 = 4.09 s total; "beats 1 teacher forward" only vs teacher@25 (4.73 s).
  LEAD with the streaming operating point: 0.56 s/frame ≈ 1.8 Hz, ring-buffer
  median fuses cached SDFs free. Future-work note in ICRA_PLAN: token count =
  5th axis (patch-2 student ≈10 Hz, same VAE; latent shrink out of scope).

IN FLIGHT (check `squeue -u gcaddeo` + `sacct` FIRST):
- **640 = P4 stage-1 chain** (USER-APPROVED launch 2026-08-28 ~22:15;
  segment 2/2 auto-queued afterany) → `outputs/p4_recursive/`, warm-start
  A@113k, gt_corrupt curriculum. On first read verify log.txt: distill_mse
  starts ≈0.045-0.05 (no jump — smoke 634 PASSED with 0.046-0.050), no NaN.
  Graceful wall = exit 138.
- **641/642 = P4 precompute shards 0/2, 1/2** → `outputs/p4_prior_recons/`
  (frozen A@113k over training views 0,2,...,22; resumable — if a shard hits
  its 24h wall before "DONE", RESUBMIT same command). When both DONE:
  continue the chain with the stage-2 config:
  `sbatch tools/train_p4_recursive.sbatch 3 4 _stage2` (segments 3-4;
  STOP_CHAIN sentinel stops everything).
- **643 = stacked visual-ft on A@113k** (USER-APPROVED) →
  `outputs/distill_visual_ft113k/`, 1×24h, ends Sat ~22:30. When done:
  paired a/b 25/8/4 + 48-group fusion (latest_ema is safe — single-purpose
  finished dir; pattern of jobs 618-621). **If its fused K8 IoU > 0.667 it
  becomes the paper's final ckpt** → rerun full-set fusion + dex for it and
  update P4 stage-2... (P4 warm start stays A@113k regardless — decided).
- **637 = full-set n=351 fusion for A@113k** (~1h45, L40S) →
  `mv_fusion_studentA113k_s8_full.json` → append to §7.20; compare vs
  teacher full-set (§7.13) and A@81k full-set (§7.17) — the honest CD/EMD
  caveat check at scale.
- **638→639 = dex rows chain for A@113k** (tag a113k, @8 steps, canonical
  ICP flags) → `summary_student_a113k_s8_dex_total_total_icp.json` in the
  INFERENCE repo root → the paper's real-capture student row vs §7.2 table.

Tasks, in order:
1. Read 637 → §7.20 full-set table. Read 639 → dex student row (§7.2
   companion). Verify 640's first log + 641/642 progress.
2. Sat eve: 643 ends → eval suite (a/b + fusion, --data_seed 1337) → stacked
   verdict → possibly re-pick final ckpt (full-set fusion + dex rerun for it
   only, patterns above).
3. Sun/Mon: precompute DONE → launch stage-2 chain (command above).
4. ~Tue: first P4 eval — needs a prior-aware eval path (the a/b harness
   passes no prior; the P4 model degrades to single-view when cond lacks
   x0_prior, so baseline evals work as-is; the STREAMING eval — prior from
   previous frames' recons — needs a new harness: simulate a view sequence,
   feed warped fused prior of earlier outputs via
   trainer.get_inference_cond(prior_sdf=..., prior_keep=...)). Compare vs
   ring-buffer median at equal K (the §7.19 success criterion).
5. Paper assembly: final frontier table (capacity × steps × views ×
   streaming), operating point = student@8 bf16 streaming 1.8 Hz.
6. Keep EVAL_GUIDANCE §7 / ICRA_PLAN / this file / memory updated; commit at
   phase boundaries.

Open user decisions: rig access timeline (ICRA centerpiece vs workshop).
P4 launched + stacked vft launched (both approved 2026-08-28). Everything
else decided — do not re-litigate.

Hard rules (unchanged):
- Mesh evals ALWAYS ICP with canonical flags (memory trellis-eval-icp).
- Cross-model a/b ALWAYS `--data_seed 1337`; name teacher ckpt explicitly.
- Never evaluate EMA below ~30k steps unless warm-start-seeded (vft/P4 dirs
  are warm-started → their EMAs are valid from ~16k).
- GPU via sbatch only; login node OOMs (can't even alloc 4 MB — run even
  tiny torch tests as cpu-partition jobs). TRAINING → gpu-h200 (4 jobs/96
  CPUs/24h cap), EVALS → gpu-l40s (2 concurrent/10 submitted).
- Prefer SLURM --dependency chains; ask user before ≥ multi-day GPU; P4
  future decisions: report first, launch only on explicit user go.
- Git: commit locally; user pushes (inference repo → branch only).
