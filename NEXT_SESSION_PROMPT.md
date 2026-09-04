# Prompt for a fresh session (copy-paste from here down)

Read ICRA_PLAN.md and EVAL_GUIDANCE.md §7 (the LIVE results ledger, §7.1–§7.25,
COMPLETE) before doing anything. TEACHER_RETRAIN.md is historical only.

## State as of 2026-09-04 evening — MEASUREMENT PROGRAM CLOSED, WRITING PHASE

**DEPLOYED MODEL FROZEN (user decision 2026-09-03): warm-P4 final**
`outputs/p4_recursive/ckpts/denoiser_ema0.9999_step0064000.pt`
(= A@113k + 64k P4-recipe steps: stage-1 corrupted-GT priors → stage-2
self-recon priors, visual losses λ2/2, 30% prior dropout; FROZEN_STUDENT.txt
+ STOP_CHAIN in the dir; never resume). Headline credentials:
- **Streaming (n=351, §7.25)**: stream_median IoU 0.721 / CD 0.0399 / EMD
  0.0624 — BEATS the 757M teacher's K8 fusion ceiling (0.716/0.0427/0.0710);
  F@0.02 0.699 vs 0.716 is the one metric under (quote honestly).
  stream_final 0.683/0.0430 at one forward/frame. Per-frame integration
  delta +0.061@2 → +0.109@8 (THE demo figure). 1.8 Hz (0.56 s/frame bf16).
- **Dex real captures (n=989, §7.25)**: CD² 0.0295/0.0104 — beats teacher
  (0.0336/0.0112) AND A@113k (0.0307/0.0110); NC 0.841/0.885 best. This is
  single-image prior-less mode: the P4 recipe improved the raw backbone.
- **Pose-free (§7.23)**: hand-volume registration 0.28°/0.07vox/0.48% scale;
  fusion penalty ~0.010 IoU; no camera extrinsics needed anywhere.

Other frozen actors: teacher FROZEN@52k; A@113k = the paper's plain student
(§7.20); B@64k = copy-init ablation; visual loss = fine-tune amplifier for
fusion (equal-compute controls: tie mid-training, no early-training win —
§7.21/§7.22); copy-shortcut measured (from-scratch P4 fails, §7.22 — the
warm-start is data-justified).

## ONE remaining measurement (Sunday)

- **687 (running) → 691 (queued) = A plain-extension segments 9-10-(11)**
  toward A@165k+ in `outputs/distill_teacherv2/` — the total-compute-matched
  control for the P4-recipe backbone claim (user requirement).
  **NEVER latest_ema on that dir — it is training again; explicit steps only.**
- SUNDAY (or when `ckpts/denoiser_ema0.9999_step0165000.pt` exists):
  1. `touch outputs/distill_teacherv2/STOP_CHAIN` + scancel the remaining
     segments (683+ GPU-days are not needed past 165k).
  2. Eval A@165k: `sbatch --partition=gpu-l40s tools/ab_eval_guidance.sbatch
     --teacher_dir outputs/distill_teacherv2 --ckpt
     denoiser_ema0.9999_step0165000.pt --num_samples 64 --steps 8
     --guidance_skip 2 --data_seed 1337 --output
     outputs/diagnostics/ab_guidance_studentA165k_ds_steps8.json`
     (+ optionally tools/streaming_eval.sbatch on it for the matched
     ringbuffer row).
  3. Compare vs the BANKED warm-P4@+52k rows (§7.25: single 0.641 /
     c_ex 5.3e-4; stream_median 0.710/0.0356) → write §7.26: does the P4
     recipe beat plain extension at matched 165k total? Prediction on file:
     geometry ≈ tie, physics P4 ahead ~2×.

## The writing phase (the actual work now)

Paper skeleton = the ledger. Framing rules accumulated (MUST follow):
- B rows always carry compute labels; B-vs-A claims only at equal compute.
- Visual loss = "fine-tune amplifier for fusion" (equal-compute controls
  §7.21; early-training hypothesis FALSIFIED §7.22 despite loss-curve hints).
- Streaming lead = stream_median at unchanged deployment cost; F@0.02
  asterisk; stacking-law reversal story (stage-2 makes recursion+fusion
  stack, §7.24).
- Batch-K: throughput-bound at batch 1 (NOT "saturated"); "8-view batch <
  1 teacher forward" only vs teacher@25; lead with streaming 1.8 Hz.
- Copy-shortcut negative justifies warm-start; zero-init graft = the
  method's own lineage (hand conditioning was added the same way).
- Dex = single-view real-capture validation; multi-view/streaming =
  synthetic held-out (rig demo or workshop = user's open decision).
- Token count = 5th-axis future work (patch-2 student ≈10 Hz, ICRA_PLAN).
- EMD/F@0.02 caveats quoted wherever they apply. Mesh evals ICP canonical.

Figures/tables to build (data all exists in outputs/diagnostics/):
1. Frontier table (capacity × steps × views × streaming; §7.1/7.2/7.13/7.25).
2. Per-frame integration trajectory (stream_warpm4_final_full.json
   frame_k vs frame_k_noprior) — the demo/centerpiece figure.
3. Dex table (old teacher / teacher-v2 / A@113k / frozen; §7.2+§7.25).
4. Fusion ladder + pose-free row (§7.13-7.15, §7.23).
5. Absorption plot (guidance effect vs training progress; §7.1/7.5-7.8).
6. Factorial 2×2×2 table (§7.22) + copy-shortcut bar.
Qualitative mesh figures: meshes exist under the inference repo
(meshes_results_marching_cubes_student_wp4fin_s8/ etc.) — rendering/selection
still to do if wanted.

## Side project (user, other cluster): smaller-latent VAE
User is retraining the sparse-structure VAE at 8³×16 (config: extra 512
stage in channels + latent_channels 16 both sides; loader one-liner
data_pose→data_pose_norm; metadata needs sdf=True column; clamp ±2 kept).
GO/NO-GO gate on return: reconstruction contact-floor vs current ~0.011.
If it passes → patch-2/8³ re-distillation = post-deadline work.

## Hard rules (unchanged)
- ICP canonical flags for mesh evals (memory trellis-eval-icp); --data_seed
  1337 for cross-model a/b; explicit ckpt names everywhere now (THREE dirs
  are dangerous for latest_ema: teacher, distill_teacherv2, p4_recursive).
- GPU via sbatch only (login node cannot alloc 4 MB); EVALS → gpu-l40s,
  TRAINING → gpu-h200; dependency chains for must-run steps.
- Ask user before ≥ multi-day GPU. Git: commit locally, USER pushes (train
  repo main; inference repo local main ahead of origin → push to a BRANCH).
  PUSH BACKLOG IS LARGE (~20 commits) — remind the user.
- Open user decisions: rig access (ICRA live demo vs workshop framing —
  now the ONLY open scope question).
