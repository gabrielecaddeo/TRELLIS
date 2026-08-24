# Brief for the presentation session: the published hand-conditioned reconstruction work

**Purpose.** You (a fresh session) will build a PowerPoint presentation about the
PREVIOUSLY PUBLISHED method — physics-aware in-hand object reconstruction from a
single image with a hand-conditioned generative model. The user will feed you the
paper PDF: **the PDF is the authority on narrative, claims, notation, and figures.**
This brief gives you (a) the system-level ground truth as embodied in the code,
(b) exactly which code files to read and what to look for in each, (c) validated
benchmark numbers and figure-making assets, and (d) pitfalls so you do not mix the
published system up with the follow-up campaign that lives in this repo.

Written 2026-08-24 by the campaign session, from the code and campaign docs — NOT
from the paper PDF (never read here). Where this brief and the PDF disagree on
terminology or emphasis, follow the PDF.

---

## 1. What the published system is (one paragraph)

Single RGB image of a hand holding an object → reconstruct the object's 3D mesh,
*aware of the hand*: the hand's known geometry both conditions the generation and
constrains it physically (no hand–object interpenetration; the object should
actually touch the hand at annotated contact regions). The generative backbone is
a TRELLIS-style **sparse-structure flow model**: a rectified-flow (flow-matching)
DiT operating on a 16³×8 latent of a 64³ SDF grid, decoded by a frozen
convolutional VAE decoder. Conditioning: DINOv2 image tokens (cross-attention),
plus three hand channels — the hand's own SDF latent, a contact/"touch" volume,
and the 2D hand mask — injected through dedicated cross-attentions and an input
fusion path. Physics enters twice: as **training losses** (non-interpenetration +
contact) and as optional **sampling-time guidance** (energy gradient on the
decoded x̂0 during Euler sampling). Output SDF → marching cubes → mesh.

## 2. Theory checklist for the slides

Cover these, in the paper's notation (the PDF defines the exact symbols):

1. **Flow matching / rectified flow**: x_t = (1−t)·x₀ + t·ε (linear path), model
   predicts velocity v = ε − x₀ (v-prediction); training target MSE on v;
   x̂₀ = x_t − t·v̂ (the estimator used everywhere physics needs geometry);
   logitNormal(1.0, 1.0) timestep sampling; Euler integration at inference with a
   **t-rescaling** (rescale_t = 3: more resolution near t=1) and 25 steps.
2. **Classifier-free guidance**: p_uncond = 0.1 conditioning dropout at training;
   at sampling, CFG strength 5.0 applied only on the **interval t ∈ [0.5, 1]**
   (GuidanceIntervalSampler) — an important detail, often on a slide of its own.
3. **The TRELLIS sparse-structure latent**: 64³ SDF grid ↔ 16³×8 latent via a
   3D-conv VAE (trained separately, frozen for flow training). Everything —
   object, hand, GT — lives on the same 64³ unit grid (voxel = 1/64 of the
   [-1,1] cube), pose-normalized per view.
4. **Hand conditioning** (the paper's core novelty):
   - hand SDF latent x0_hand: same VAE encoding of the hand's SDF, fed through a
     dedicated input pathway and per-block cross-attention;
   - touch volume (2×64³: contact-voxel mask + distance-to-contact), encoded by a
     small 3D CNN and fused with the input latent;
   - 2D hand mask tokens with their own positional embedding and cross-attention;
   - "weighted attention" variant in the DiT blocks.
5. **Physics losses** (present the PAPER's formulation — see pitfall P2):
   - non-interpenetration: penalize predicted-object SDF being negative (inside)
     where the hand interior is; hinge/soft-clamped penetration depth;
   - contact: penalize |SDF_object| at annotated contact voxels (the object
     surface should pass through the contact set);
   - both computed on the DECODED x̂₀ during training (decoder in the loop),
     weighted toward low-t (clean) samples.
6. **Sampling-time physics guidance** (deployed variant): DPS-style — decode x̂₀
   at each Euler step, take the gradient of the physics energy w.r.t. x_t, add a
   scaled correction to the velocity (α = 10, guidance from step 5 on,
   λ_inter = 500, λ_contact = 50 inside the energy).
7. **Evaluation**: DexYCB real-capture benchmark (994 in-hand instances, 1 view),
   mesh metrics after **ICP alignment** in a bbox-normalized frame — Chamfer-²,
   Normal Consistency, F-score@τ ladder, EMD; plus physics metrics (penetration
   fraction/depth, contact distance) on the SDF grid.

## 3. Architecture facts (for a "model card" slide)

- DiT: **24 blocks, 1024 channels, 16 heads, mlp_ratio 4** (~757M params);
  patch_size 1 on the 16³ latent → 4096 tokens; qk-RMS-norm; absolute PE.
- Conditioning dims: DINOv2 ViT-L/14-reg tokens (1024-d); hand-mask tokens 37×37
  = 1369 with learned PE; touch encoder: Conv3D 2→16→32→8, fused 16→8 (1×1×1).
- VAE decoder (frozen): `ss_dec_conv3d_16l8`, custom-trained
  (`vae_final_all_resume_2`, step 300k). Decoder has a measured "floor": even GT
  latents decode with ~0.011 mean |SDF| at contact voxels — relevant if the talk
  shows physics numbers.
- Sampler defaults (deployment): 25 Euler steps, CFG 5.0 on [0.5,1], rescale_t 3,
  seed-controlled; one view, one sample.

## 4. Data & benchmark facts

- Training data (synthetic renders, three sources): Leap_Hand (10,327 train
  instances), Hands (912; YCB objects in-hand), Hands_Google (8,022). Each grasp
  instance has **24 posed views**; per view: RGBA render, hand/object 2D masks,
  64³ SDFs of object and hand, contact annotations, VAE latents. Held-out test
  splits exist (319/32/248).
- Real benchmark: DexYCB-derived set, 994 instances (8 calibrated cameras × 158
  captures at one frame), through the inference pipeline.
- **Validated published-era benchmark numbers** (reproduced 2026-08 with the
  canonical eval — safe to quote): ICP-aligned, bbox[-1,1]³, n=992:
  **CD² 0.0458 mean / 0.0181 median, NC 0.806, F@0.02 0.130, F@0.10 0.702.**
  (Raw-frame, no-ICP CD is ≈0.195 and dominated by pose error — never quote it.)

## 5. Code reading list (ordered; repo = /projects/gcaddeo/train_flow_conditioned/TRELLIS)

1. `trellis/models/sparse_structure_flow.py` — `SparseStructureFlowModelConditioned`:
   the whole conditioning story (input fusion of latent+touch, x0_hand pathway,
   mask-hand tokens + PE, forward pass). NOTE the `use_hand_pe` flag: **False**
   reproduces the published model's forward exactly (the flag exists because the
   follow-up campaign added hand positional embeddings — see P1).
2. `trellis/modules/transformer/modulated.py` — the DiT block: adaLN modulation,
   self-attn + THREE cross-attentions (image / mask-hand / hand), weighted
   attention, MLP.
3. `trellis/trainers/flow_matching/flow_matching.py` —
   `ImageConditionedFlowMatchingCFGTrainerConditioned.training_losses`: flow-MSE
   + the physics-loss block (decoder-in-the-loop, time weighting, CFG-drop
   gating). See P2 before presenting formulas from here.
4. `trellis/trainers/flow_matching/mixins/image_conditioned.py` — DINOv2
   conditioning (how images become tokens; the p_uncond dropout).
5. `trellis/pipelines/samplers/flow_euler.py` — `FlowEulerSampler.sample` +
   `FlowEulerGuidanceIntervalSampler` (Euler loop, rescale_t, CFG interval);
   `sample_velocity_conditioned` = the deployed physics guidance (lines ~278–470).
   Ignore `sample_guided_v2`, `sample_oc_flow`, `sample_multiview_consistency` —
   post-publication work (P1).
6. `trellis/datasets/components.py` (`ImageConditionedMixinRotationConditioned.
   get_instance`) + `trellis/datasets/sparse_structure_latent.py`
   (`SparseStructureLatentSDFConditioned`) — exactly what one training sample
   contains and how views/masks/contacts/latents are loaded.
7. Inference repo `/projects/gcaddeo/inference/TRELLIS`:
   `trellis/pipelines/trellis_image_to_3d_conditioned.py` (`run_velocity`,
   `sample_sparse_structure_velocity` — the deployment path),
   `teacher_v2_port/inference_dex_teacher_v2.py` (dex benchmark runner incl. the
   marching-cubes convention: level 0.0, ZYX transpose, half-voxel shift), and
   `eval_meshes_paired_emd_voxel_dex_cm.py` (the metric definitions; canonical
   flags: `--cd_squared --normalize bbox_-1_1 --pt2tri --icp --icp_pca
   --icp_random_restarts 10 --emd --voxel_iou ...`).
8. Optional background: TEACHER_RETRAIN.md §1–§4 (what each loss/term is and why,
   written while auditing the published implementation).

## 6. Figure assets you can generate

- **3D galleries**: `tools/visualize_ab_sdfs.py` renders rotatable
  object+hand+contacts panels (object blue, hand gray, contact voxels red) from
  saved SDF blobs in `outputs/diagnostics/*_sdfs/`
  (`cd outputs/diagnostics && python3 -m http.server 8877`, then screenshot).
  Existing blobs cover the published-era model too
  (`ab_guidance_4arm_ycb_oldteacher_sdfs/` = the published checkpoint's outputs).
- Input-conditioning examples: any `datasets_split/*_test/renders_cond/<inst>/`
  (RGBA render + `NNN_mask_1/2.png` hand/object masks).
- Meshes from the published model on real captures:
  `/projects/gcaddeo/inference/TRELLIS/meshes_results_marching_cubes_ablation_dex/`
  (sample.ply per instance; GT models in `models/ycb/`).
- Environment for running anything: conda env `trellis_hopper`
  (`source /opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2/etc/profile.d/conda.sh`),
  GPU work via sbatch only (partitions gpu-h200 / gpu-l40s; login node OOMs).

## 7. Pitfalls — read before writing a single slide

- **P1 — two generations of the model live in this repo.** The published model =
  the "old teacher" (`outputs/flow_conditioned_all_losses_resume_32k_resume3_LEAP`,
  runs with `use_hand_pe=False`). The repo's current default configuration and
  most recent outputs belong to **teacher_v2 and a distilled student — a
  POST-publication campaign** (retraining, fixed physics, guidance studies,
  distillation, multi-view fusion; ledger in EVAL_GUIDANCE.md §7). None of that
  goes into this presentation unless the user explicitly asks for an outlook
  slide.
- **P2 — present the paper's loss formulation, not the audited implementation.**
  The campaign found implementation-level deviations in the published physics
  code (x̂₀ reconstruction formula, normalization, masking details). For a talk
  about the published work, the PDF's equations are what you present; do not
  surface the audit unless the user asks.
- **P3 — guidance versions.** The published/deployed guidance is
  `sample_velocity_conditioned` (α=10, DPS-style). The other samplers in
  flow_euler.py (guided_v2, oc_flow, multiview) are post-publication.
- **P4 — benchmark numbers.** Quote only the ICP-aligned numbers in §4 (they
  reproduce the paper's harness); never raw-frame CD; never mix in teacher_v2 /
  student numbers from EVAL_GUIDANCE.md §7 without labeling them as follow-up.
- **P5 — check facts against the PDF.** Anything here that the PDF contradicts
  (dataset counts, λ values, architecture details as *published*) — the PDF wins;
  this brief describes the code as it exists, which may postdate the paper.
