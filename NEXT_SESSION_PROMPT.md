# Prompt for a fresh session (copy-paste from here down)

Read ICRA_PLAN.md and EVAL_GUIDANCE.md §7 (the LIVE results ledger, §7.1–§7.26,
COMPLETE) before doing anything. TEACHER_RETRAIN.md is historical only.

## State as of 2026-09-08 — MEASUREMENT PROGRAM FULLY CLOSED (§7.1–§7.26 + dex attribution); WRITING + dex-full real-capture multi-view

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
  single-image prior-less mode. (Attribution CORRECTED 2026-09-08: the gain
  was training time — A@165k ties/edges it; see §7.26 block below.)
- **Pose-free (§7.23)**: hand-volume registration 0.28°/0.07vox/0.48% scale;
  fusion penalty ~0.010 IoU; no camera extrinsics needed anywhere.

Other frozen actors: teacher FROZEN@52k; A@113k = the paper's plain student
(§7.20); B@64k = copy-init ablation; visual loss = fine-tune amplifier for
fusion (equal-compute controls: tie mid-training, no early-training win —
§7.21/§7.22); copy-shortcut measured (from-scratch P4 fails, §7.22 — the
warm-start is data-justified). A@165k = best single-view student (0.663
synthetic / dex 0.0295-0.0100; ckpt step0165000 in distill_teacherv2).
Footnote: P4's prior encoder used RAW step0300000 while dataset latents used
the EMA encoder — internally consistent within P4 (train+eval both raw), do
NOT change on the frozen model; new latents (dex-full) use EMA like training.

## Measurement program FULLY CLOSED (2026-09-08)

- §7.26 (matched 165k): plain A@165k BEATS the P4 recipe single-view (IoU
  0.663 vs 0.641, CD 0.0519 vs 0.0594); the P4 recipe wins streaming
  (stream_median 0.710 vs plain ring-buffer 0.691). Verdict: the recipe
  trades ~0.02 single-view IoU for ~0.02 streaming IoU.
- Dex attribution (695/696): A@165k ties/edges frozen wp4 on real captures
  single-image (CD 0.0295/0.0100 vs 0.0295/0.0104, F@0.02 0.170 vs 0.164) —
  the recipe's dex gain was training time. WINNER BY MODE: single image →
  A@165k (best student; teacher wins synthetic-only at 3× latency); streaming
  → frozen wp4 (0.721, beats teacher ceiling). DEPLOYED MODEL UNCHANGED:
  frozen wp4 (deployment = streaming).
- Final real-capture table rows: old teacher / teacher-v2@52k / A@165k
  (single-view student) / frozen wp4 (deployed streaming).
- A-chain STOPPED at ~174.5k (ckpts to ~174k exist; ~177k matchpoint free if
  a reviewer asks). THREE dirs dangerous for latest_ema: teacher,
  distill_teacherv2, p4_recursive.

## NEW TOP TASK: dex-full — REAL-CAPTURE multi-view (8 frames/grasp)

User is copying the FULL DexYCB export to
`/projects/gcaddeo/inference/TRELLIS/dex-full/` (in progress; currently
stage1_meshes=renders_cond, stage2_sdf=data_pose_norm — user renames after
copy — plus logs/, supervise_build.sh, view_groups.json manifest). Exactly
8 frames per object/hand pose. PRIZE: moves the streaming/fusion claims to
REAL captures (possibly pose-free) — supersedes the rig-gap concern.

Plan when the user confirms the copy is done:
1. Inspect `view_groups.json` + one instance: metas must have the `pose`
   block (similarity convention) for meta-warps — if absent/different, use
   the HAND-REGISTRATION path (§7.23, tools/hand_pose_registration.py:
   meta-free, validated 0.28°/0.07vox/0.48%; fusion cost ~0.01 IoU).
2. Completeness checklist (train-repo dataset class needs ALL of):
   metadata.csv (sha256, aesthetic_score, cond_rendered,
   ss_latent_vae_final_all_resume_2_0300000 cols); renders_cond/<inst>/
   transforms.json + images + NNN_mask_1.png (hand) + NNN_mask_2.png (obj);
   data_pose_norm/<inst>/<inst>_fNNN_meta.json (+ pose2d_meta);
   contacts/<inst>_fNNN_{contact_coords,dist_to_contact}.npy (REQUIRED by
   get_instance — verify dex-full has them!); sdfs/<inst>_fNNN__{object,hand}.npy.
3. Latents: `/projects/gcaddeo/inference/TRELLIS/create_latents_dex_full.py`
   (COMMITTED, FIXED version — all 8 frames, train-convention names
   {instance}_{v}__{object,hand}.npz, strict EMA-encoder
   encoder_ema0.9999_step0300000 = same as training latents; the OLD
   create_latents_dex.py is f000-only with broken _f00__ names — do not use).
   Run via sbatch gpu-l40s; shardable --rank/--world_size; --skip_existing.
4. Smoke: streaming_eval + fusion on ~4 dex-full groups (views 0-7 →
   --views "1,2,3,4,5,6,7,0" streaming / "0,...,7" fusion; adjust k_subsets
   "1:0;2:...;4:...;8:0,...,7"). Watch: frame ordering semantics (are the 8
   frames simultaneous multi-cam or temporal?), meta warp validity (GT->GT
   hand warp check like §7.9), grid/scale conventions.
5. Full runs: frozen wp4 streaming + fusion (+ pose-free variant) on all
   dex-full groups → the paper's real-capture streaming table.

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
  synthetic held-out UNLESS dex-full lands (then real-capture streaming
  rows supersede; rig demo becomes nice-to-have).
- Token count = 5th-axis future work (patch-2 student ≈10 Hz, ICRA_PLAN).
- EMD/F@0.02 caveats quoted wherever they apply. Mesh evals ICP canonical.

Figures/tables to build (data all exists in outputs/diagnostics/):
1. Frontier table (capacity × steps × views × streaming; §7.1/7.2/7.13/7.25).
2. Per-frame integration trajectory (stream_warmp4_final_full.json
   frame_k vs frame_k_noprior) — the demo/centerpiece figure.
3. Dex table (old teacher / teacher-v2 / A@165k / frozen wp4; §7.2+§7.26).
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
