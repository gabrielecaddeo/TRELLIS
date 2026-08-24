# Teacher retrain plan — review findings, diagnostics, fixes, and the fresh run

**Status:** written 2026-08-06, executed 2026-08-06. See §8 for the execution log: the
pre-change code is backed up, probe C has been run and **confirms finding A**, the fixes are
applied, and the retrain is set up as a self-chaining `gpu-h200` job.

This file is a self-contained handoff. Read it top to bottom in a new session; it assumes no
prior conversation context.

---

## 0. The three-step plan

| step | what | state |
|---|---|---|
| **1** | Run the diagnostics on the current teacher (§3) | probe C **done**; A/B/D run in §8 |
| **2** | Apply the code fixes (§4) | **done** — fixes 1–5, 7, 8 and finding G/H |
| **3** | Retrain the teacher from the pre-conditioning base checkpoint (§5) | see §8 |

The reason for step 1 before step 2: fix A (positional embedding on the hand tokens) is the
expensive one and is based on a static reading of the code. Probe C measures whether the current
teacher is actually blind to hand position, which either confirms the fix is the highest-value
change or falsifies it before a multi-day retrain is committed. **Probe C confirmed it** — see §8.

---

## 1. Where everything is

### Repos
- **This repo (conditioned):** `/projects/gcaddeo/train_flow_conditioned/TRELLIS`
- **Base repo (unconditioned):** `/projects/gcaddeo/train_flow/TRELLIS` — where the
  pre-conditioning flow model and the SS-VAE were trained. Datasets live under
  `/projects/gcaddeo/train_flow/TRELLIS/datasets/`.

### The base checkpoint the retrain starts from
```
/projects/gcaddeo/train_flow/TRELLIS/outputs/flow_final_norm_resume330/ckpts/denoiser_ema0.9999_step0350000.pt
```
- Model `SparseStructureFlowModel` (vanilla TRELLIS): 24 blocks, `model_channels=1024`,
  `patch_size=1`, `pe_mode=ape`, `qk_rms_norm=true`, `use_fp16=true`.
- **489 tensors, all float32** (`save()` writes the fp32 master params, so the fp16 torso is not
  a problem when loading into a `use_fp16=false` model).
- Trained on ABO, 3D-FUTURE, YCB, Google, ObjaverseXL_sketchfab. `mse ≈ 0.306` at step 350k,
  lr 1e-4, batch 16.
- **Same latent space as the conditioned runs:** `latent_model = vae_final_all_resume_2_0300000`,
  `ss_dec_path = /projects/gcaddeo/train_flow/TRELLIS/outputs/vae_final_all_resume_2`,
  `ss_dec_ckpt = step0300000`. No `normalization` block in either config, so `x_0` is the raw VAE
  posterior mean in both. **No latent-space mismatch — the warm start is clean.**
- Despite the name, `flow_final_norm_*` has nothing to do with latent normalization
  (`calculate_norm.py` in the base repo computes stats over SLAT `feats`, not SS latents).

### The current teacher (to be replaced)
```
outputs/flow_conditioned_all_losses_resume_32k_resume3_LEAP/ckpts/denoiser_ema0.9999_step0054000.pt
```
Lineage, all with `SparseStructureFlowModelConditioned` (24 blocks, `use_fp16=false`):

| run | steps | data | λ_ni | λ_contact |
|---|---|---|---|---|
| `flow_conditioned_no_losses` | 0–10.5k | Hands, Hands_Google | 0 | 0 |
| `..._resume_10k` / `_24k` / `_32k` | →32k | Hands, Hands_Google | 0 | 0 |
| `flow_conditioned_all_losses_resume_32k` | 32k–36k | Hands, Hands_Google | 50→200 | 0.1 |
| `..._resume1` | 36k–40k | " | 200 | 0.1 |
| `..._resume2` | 40k–44k | " | 200 | 0.1 |
| `..._resume3_LEAP` | 44k–55.5k | **Leap_Hand only** | 200 | 0.1 |

The first run warm-started from the base checkpoint exactly the way §5 proposes — see
`outputs/flow_conditioned_no_losses/command.txt`. That run went `mse 0.177 → 0.160` over 10.5k
steps with batch 4 / lr 1e-5, i.e. **no warm-start blow-up**: the zero-initialised branches make
step 0 numerically identical to the base model.

### Datasets (all under `/projects/gcaddeo/train_flow/TRELLIS/datasets/`)

| dataset | instances | frames | `data_pose_norm/<inst>/contacts/` |
|---|---|---|---|
| `Hands` | 944 | 24 | yes (48 files/inst) |
| `Hands_Google` | 8270 | 24 | yes |
| `Leap_Hand` | 10648 | 24 | yes |

All three carry the same `latent_model` column and are directly usable. `--data_dir` takes a
**comma-separated** list (`components.py:24` → `roots.split(',')`).

Measured data facts used throughout this document:
- object latent `x_0`: std **0.382**; hand latent `x0_hand`: std **0.578** (300 Leap_Hand samples).
- `touch` is `[2, 64, 64, 64]`: ch0 = binary contact grid, ch1 = `dist_to_contact`.
- **75 contact voxels per sample on average** (median 63, range 26–222) out of 64³ = 262,144,
  i.e. **0.03% of the grid**.
- `dist_to_contact` maxes at ~1.73 = √3 ⇒ the grid is **unit-cube normalized**, so
  **1 voxel = 1/64 = 0.0156** in SDF units.

### Environment / cluster
- conda env **`trellis_hopper`** (`~/.conda/envs/trellis_hopper`, py3.10, torch 2.4.0+cu121).
  Conda root: `/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2`.
- SLURM partition **`gpu-h200`** (2 GPUs + 32–64 CPUs per node, **1-day time limit**).
  `gpu-l40s` also exists with a 7-day limit. The `--partition=gpuh` in `~/test.job.sh` is stale.
- Use `--cpus-per-task=16`; the dataset is CPU-bound (1024×1024 RGBA render + 2 masks + two 64³
  float grids per sample).
- `#SBATCH --export=NONE` drops the interactive env — job bodies must `source
  ${CONDA_ROOT}/etc/profile.d/conda.sh` explicitly.
- DINOv2 hub cache **is warm** at `~/.cache/torch/hub/facebookresearch_dinov2_main` +
  `checkpoints/dinov2_vitl14_reg4_pretrain.pth`, so compute nodes without egress are fine.

---

## 2. Review findings

### 2a. The physics losses did essentially nothing in the current teacher

Parsed from `outputs/flow_conditioned_*/log.txt` over 23,500 steps at full
`lambda_ni=200`, `lambda_contact=0.1`:

| run | steps | mse | `ni_loss_raw` | `contact_raw` |
|---|---|---|---|---|
| `..._resume_32k`   | 32k–36k   | 0.108 → 0.104   | 4.5e-6 → 3.4e-6 | 0.0445 → 0.0434 |
| `..._resume1`      | 36k–40k   | 0.105 → 0.102   | 3.3e-6 → 3.1e-6 | 0.0437 → 0.0421 |
| `..._resume2`      | 40k–44k   | 0.102 → 0.102   | 3.1e-6 → 3.0e-6 | 0.0422 → 0.0417 |
| `..._resume3_LEAP` | 44k–55.5k | 0.1545 → 0.1516 | 6.5e-6 → 5.3e-6 | 0.0257 → 0.0251 |

(The discontinuity at step 44001 is the dataset switch to Leap_Hand, not a training effect.)

- Over 11,500 steps `contact_raw` improved **2.3%** while `mse` improved **1.9%** — they
  co-move, so contact is tracking general model improvement rather than being optimized.
- Median loss contribution: `ni_loss/mse` **0.5%**, `contact_loss/mse` **1.7%**.
- **NI has no margin.** `relu(-sdf_obj)` ([flow_matching.py:671](trellis/trainers/flow_matching/flow_matching.py#L671))
  is exactly zero, with zero subgradient, unless the predicted object genuinely overlaps the
  hand. `ni_loss_raw ≈ 5e-6` against the `max_pen=0.1` cap is 5e-5 of saturation, and 91.6% of
  logged steps are below 1e-5 — so ~no gradient reaches the denoiser.
- **Contact is at its resolution floor.** `contact_raw ≈ 0.025` = **|sdf| ≈ 1.6 voxels**, about
  as close as a 64³ SDF passing within a voxel of all 75 contact points can get. Only ~75 of
  4096 latent cells can receive any gradient at all.

### 2b. Conceptual errors, severity order

**A. [High] The hand branch has no positional embedding.**
[sparse_structure_flow.py:369](trellis/models/sparse_structure_flow.py#L369) adds `pos_emb` to
the noisy-latent tokens; [sparse_structure_flow.py:392-396](trellis/models/sparse_structure_flow.py#L392-L396)
does **not** add it to `x0h_tokens`. `cross_attn_hand`
([modulated.py:265-268](trellis/modules/transformer/modulated.py#L265-L268)) is therefore
permutation-invariant over the 4096 hand grid cells: permuting the hand latent leaves the output
unchanged. The model gets a *bag of local hand features* and cannot know **where** the hand is.
`touch` is fused into `x0h` before patchify
([sparse_structure_flow.py:387-389](trellis/models/sparse_structure_flow.py#L387-L389)) so the
contact grid loses its position the same way.

Same class of bug, milder: `mask_hand` goes through `nn.Linear(1, 1024)`
([sparse_structure_flow.py:377](trellis/models/sparse_structure_flow.py#L377)), so all 1369
tokens lie on one affine line in feature space with no positional identity — that branch can only
convey a scalar summary of mask-coverage. (Contrast `mask_obj`, which is applied as an attention
*bias* over positional DINOv2 tokens — that one works correctly.)

This undercuts the premise of hand conditioning and is the likely reason the physics constraints
cannot be satisfied through the conditioning path: the network is never told where to avoid.

**B. [High] `x0_pred` is built from the ground-truth noise.**
[flow_matching.py:662](trellis/trainers/flow_matching/flow_matching.py#L662):
```python
x0_pred = (1.0 - self.sigma_min) * noise - pred
```
Since `v_target = (1-σ)·ε − x_0`, this gives `x0_pred − x_0 = v_target − pred` — the **full
v-prediction error at every t**. The correct estimator already exists, unused, in the same file
at [flow_matching.py:397-410](trellis/trainers/flow_matching/flow_matching.py#L397-L410)
(`_v_to_xstart_eps`, matching [flow_euler.py:53-57](trellis/pipelines/samplers/flow_euler.py#L53-L57)):
`x̂_0 − x_0 = (σ + (1-σ)t)(v_target − pred) ≈ t·(v_target − pred)`, which vanishes as t → 0.

With `mse ≈ 0.15` the RMS error of `x0_pred` is **0.387** against a measured object-latent std of
**0.382** — the "predicted object" handed to `ss_dec` has error as large as the object itself, at
every timestep. It also uses privileged information (`noise`) unavailable at inference, and it
makes the `(1-t)²` weighting pointless, since that weighting exists precisely to exploit
t-dependent accuracy this estimator throws away.

**C. [High] CFG dropout desyncs the physics losses.**
`get_cond` ([image_conditioned.py:189-224](trellis/trainers/flow_matching/mixins/image_conditioned.py#L189-L224))
zeroes `x0_hand` and `touch` for ~`p_uncond=0.1` of the batch, but
[flow_matching.py:667](trellis/trainers/flow_matching/flow_matching.py#L667) and
[flow_matching.py:697](trellis/trainers/flow_matching/flow_matching.py#L697) use the **original,
un-dropped** tensors. Those samples are penalized for interpenetrating and failing to touch a
hand the network cannot see — an impossible objective — and the bias lands on the unconditional
branch that CFG extrapolates away from.

**D. [Medium] The `(1-t)^p` weighting is normalized away.**
[flow_matching.py:656-658](trellis/trainers/flow_matching/flow_matching.py#L656-L658) divides by
`w.sum()`, making it a scale-invariant weighted average. The comment says "smoothly downweight
physics at high t", but a batch where every `t ≈ 0.95` still yields a full-magnitude physics
loss. It only reweights *within* a batch.

**E. [Low, by design] `cond` is encoded and discarded.** No transformer block reads
`context['cond']` — only the object-masked `cond_mask` reaches the network. **Confirmed
intentional.** But a full ViT-L/14 forward at 518² still runs every step
([image_conditioned.py:194](trellis/trainers/flow_matching/mixins/image_conditioned.py#L194)) for
a tensor nothing consumes.

**F. [Low] Snapshots run on the training set.** `--data_dir_test` defaults to the same LEAP path
as `--data_dir` ([train.py:153](train.py#L153)), so `dataset_test == dataset` and snapshot
metrics carry no held-out signal.

**G. [Low] `forward()` mutates `cond` in place** and the transforms are not idempotent
([sparse_structure_flow.py:376-396](trellis/models/sparse_structure_flow.py#L376-L396)). This
only survives 50-step sampling because `_inference_model`
([flow_euler.py:76-86](trellis/pipelines/samplers/flow_euler.py#L76-L86)) happens to rebuild
`new_cond` as a fresh dict on every call. Anything that reuses a cond dict across two forwards
will break.

**I. [High — found during execution, 2026-08-06] The hand branches are NOT zero-initialised.**
`ModulatedTransformerCrossBlockConditioned.__init__` zeroes `cross_attn_mask_hand.to_out` and
`cross_attn_hand.to_out` ([modulated.py:221-222, 235-236](trellis/modules/transformer/modulated.py#L221-L236)),
but that runs while the blocks are being constructed — i.e. **before**
`SparseStructureFlowModelConditioned.initialize_weights()`, whose
`self.apply(_basic_init)` xavier-initialises *every* `nn.Linear` in the model and wipes both
zeros out. Only `adaLN_modulation[-1]` and `out_layer` are re-zeroed afterwards.

Consequence: the claim in §4/§5 that "warm start is unaffected, both branches are no-ops at
step 0" was **false in the shipped code**. Warm-starting from the pre-conditioning checkpoint
injected randomly-weighted hand-branch output into the residual stream from step 0, so the
whole existing lineage started from a perturbed model rather than from the base model exactly.

Measured: with the base checkpoint loaded, swapping the hand latent changed the output by
`9.4e-3` at "init" — impossible if `to_out` were zero (`tools/verify_fixes.py` check 3).

Fixed by re-zeroing both `to_out` layers at the end of `initialize_weights()`.

**H. [Low] The dataset swallows errors.** `StandardDatasetBase.__getitem__`
([components.py:51-57](trellis/datasets/components.py#L51-L57)) catches any exception and retries
a random index, hiding missing/corrupt contact or pose files.

---

## 3. Step 1 — run the diagnostics

**Files (already written, compile-checked, never run):**
- `tools/diagnose_physics_losses.py`
- `tools/diagnose_physics_losses.sbatch`

The script is read-only: it writes only under `outputs/diagnostics/` and never calls
`optimizer.step()`. It reuses the real training code (`encode_image` / `get_cond` / `diffuse` /
`get_v` / `_v_to_xstart_eps` / `ss_dec`) via a `DiagnosticTrainer` subclass that skips only the
optimizer, EMA, master params and checkpoint load.

### Submit
```bash
cd /projects/gcaddeo/train_flow_conditioned/TRELLIS
sbatch tools/diagnose_physics_losses.sbatch                            # full run, ~1h
sbatch tools/diagnose_physics_losses.sbatch --probes C --num_batches 20   # 10-min smoke test
```
Extra args are forwarded to the python script. Useful flags: `--probes A,B,C,D`,
`--num_batches`, `--num_batches_grad`, `--batch_size`, `--ckpt`, `--data_dir`.

### What each probe answers

| probe | question | how to read it |
|---|---|---|
| **A** | Is the NI/contact number real, or an artifact of a bad `x0_pred`? Recomputes both terms on three x0 estimates (`as_trained`, `correct`, `gt_floor`), bucketed into 10 t-bins. | If `frac_obj_inside/as_trained ≈ 0` while `gt_floor` is not, the decoded "object" has no interior and NI was measuring nothing. If `contact/gt_floor ≈ contact/as_trained`, contact never had headroom. `x0_rmse/correct` should fall sharply as t→0 while `x0_rmse/as_trained` stays flat — that flatness **is** finding B. |
| **B** | Gradient attribution: ‖∇mse‖ vs ‖∇ni‖ vs ‖∇contact‖ at fixed t per bin, for both x0 variants. | The definitive ratio. If `gn_ni/mse` and `gn_contact/mse` are both ≲1e-2 in every bin *even for the `correct` variant*, the physics terms need redesigning, not reweighting. |
| **C** | Does the model use hand **position**? Paired comparison at identical t/noise: baseline vs. hand latents rolled by 1 along the batch vs. hand zeroed. | If `shuffled_hand ≈ baseline`, the model is insensitive to *which* hand it sees — confirming finding A. If `zeroed_hand ≈ baseline` too, the hand branch contributes nothing. **A large gap would falsify finding A** and is the outcome worth watching for. |
| **D** | How large is the CFG desync (finding C)? Reports NI/contact on dropped vs. kept samples at `p_uncond=0.1`. | The dropped/kept gap is the size of the bias injected into the unconditional branch. |

### Sanity gate
With the `as_trained` variant, the script's batch-level `mse`, `ni_loss_raw` and `contact_raw`
must reproduce
`outputs/flow_conditioned_all_losses_resume_32k_resume3_LEAP/log.txt` near step 54000 —
**mse ≈ 0.152, ni_loss_raw ≈ 5e-6, contact_raw ≈ 0.025** — to within batch noise. If they don't,
the harness is wrong, not the teacher. Fix that before trusting anything else.

Output lands in `outputs/diagnostics/physics_probe_<run>_<ckpt>.json` plus printed tables.

---

## 4. Step 2 — the code fixes

Apply after the diagnostics land; the numbers may change the priority of fixes 5 and 6.

### Fix 1 — positional embedding on the hand branch *(finding A)*

`trellis/models/sparse_structure_flow.py`, in `SparseStructureFlowModelConditioned.forward`,
around line 395:
```python
        x0h_tokens = self.input_layer_x0h(x0h_patches)
        if self.pe_mode == "ape":
            x0h_tokens = x0h_tokens + self.pos_emb[None]      # <-- ADD
        cond['x0_hand'] = x0h_tokens.type(self.dtype)
```
`self.pos_emb` is already the right shape (4096 × `model_channels` for `resolution=16,
patch_size=1`) and is registered only under `pe_mode == "ape"`, hence the guard.

For `mask_hand`, add a 2-D APE over the 37×37 DINOv2 patch grid. In `__init__`:
```python
        self.mask_hand_embedder = nn.Linear(1, self.cond_channels)
        mh_pos = AbsolutePositionEmbedder(self.cond_channels, 2)(
            torch.stack(torch.meshgrid(torch.arange(37), torch.arange(37), indexing='ij'),
                        dim=-1).reshape(-1, 2).float()
        )                                                     # [1369, cond_channels]
        self.register_buffer("mask_hand_pos_emb", mh_pos)
```
and in `forward`, after the embedder:
```python
        embedded_mask_hand = self.mask_hand_embedder(cond['mask_hand'].view(x.shape[0], -1, 1))
        embedded_mask_hand = embedded_mask_hand + self.mask_hand_pos_emb[None]   # <-- ADD
```
(`AbsolutePositionEmbedder` is already imported in this module;
`trellis/modules/transformer/blocks.py:8-46`. 37 = 518/14; hardcoding it is consistent with the
`k.shape[1] - 5` assert in `weighted_scaled_dot_product_attention`.)

**Warm start is unaffected:** `cross_attn_hand.to_out` and `cross_attn_mask_hand.to_out` are
zero-initialised ([modulated.py:221-222, 235-236](trellis/modules/transformer/modulated.py#L221-L236)),
so both branches are no-ops at step 0 regardless of what enters them.

### Fix 2 — correct v→x0 conversion *(finding B)*

`trellis/trainers/flow_matching/flow_matching.py:662`:
```python
-            x0_pred = (1.0 - self.sigma_min) * noise - pred
+            x0_pred, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred)
```

### Fixes 3 + 4 — CFG-gate the physics losses and drop the `w_sum` normalization

In `training_losses`, read the dropout mask off `cond_dict` **before** the forward (the model
mutates the dict in place — finding G), then fold it into `w`:
```python
        cond_dict = self.get_cond(cond, mask_hand, mask_obj, cond_mask, x0_hand, touch, **kwargs)
        B = x_0.shape[0]
        # 1 where the CFG mixin kept the conditioning, 0 where it zeroed it.
        # NOTE: must read a key the CFG mixin actually toggles per sample. Do NOT use
        # cond_dict['cond'] -- after fix 7 that tensor is all-zeros for every sample, so a
        # cond-based mask would silently zero `kept` and disable physics for the whole run.
        kept = (cond_dict['x0_hand'].reshape(B, -1).abs().sum(dim=1) > 0).float()
        pred = self.training_models["denoiser"](x_t, t * 1000, cond_dict, **kwargs)
        ...
        p = getattr(self, "physics_time_power", 2.0)
        w = (1.0 - t).clamp(0.0, 1.0).pow(p) * kept          # zero for dropped samples
        ...
        ni_loss_raw  = (w * ni_per_sample).mean()            # NOT / w.sum()
        contact_raw  = (w * contact_per_sample).mean()
```
Deriving `kept` from the returned dict rather than from the RNG keeps it correct regardless of
how `ClassifierFreeGuidanceMixin` draws its mask. Note the `.mean()` change makes the physics
terms genuinely smaller at high t, which is what the existing comment claims but the current
`/ w_sum` cancels.

### Fix 5 — give NI a margin *(finding 2a)*

Currently NI only exists on real overlap. A margin gives gradient just before contact:
```python
        margin = getattr(self, "ni_margin", 1.0 / 64.0)      # 1 voxel, unit-cube units
        obj_inside  = max_pen * torch.tanh(F.relu(margin - sdf_obj) / max_pen)
        hand_mask   = (sdf_hand < -margin).float()           # strictly interior
```
**Caution:** widening the margin makes NI fight the contact loss, which wants `|sdf_obj| = 0`
exactly at the hand surface. Excluding the contact shell via `sdf_hand < -margin` is what keeps
the two terms from cancelling. Keep the margin at 1–2 voxels; do not go wider without checking
probe B again.

### Fix 6 — reconsider the contact term *(finding 2a)*

At 75 voxels/sample and a 1.6-voxel floor the current binary-mask formulation cannot move.
Either use the dense `touch[:,1]` distance field — the commented-out variant at
[flow_matching.py:486-512](trellis/trainers/flow_matching/flow_matching.py#L486-L512) already
sketches an `exp(-dist/sigma)` soft neighbourhood — or drop the term and save a decoder pass.
Decide from probe A's `contact/gt_floor`: if the ground-truth x0 already scores ≈0.025, there is
nothing to learn and the term should be redesigned or removed.

### Fixes 7–8 — housekeeping
- Skip the unused `encode_image(cond)` call in `get_cond` / `get_inference_cond`
  ([image_conditioned.py:194, 233](trellis/trainers/flow_matching/mixins/image_conditioned.py#L194)),
  or wire `context['cond']` into the blocks. Confirmed the model never reads it, so removing the
  call is free speed. Keep the dict key (the CFG mixin uses it for the batch size, and the fix-3
  `kept` mask above reads it) — populate it from `cond_mask_enc` or a cheap placeholder if the
  encode is removed.
- Point `--data_dir_test` at a genuinely held-out split so snapshots mean something.
- Optional: stop `StandardDatasetBase.__getitem__` from silently swallowing exceptions
  ([components.py:51-57](trellis/datasets/components.py#L51-L57)) — at minimum count and report
  the retries.

---

## 5. Step 3 — retrain the teacher from the base checkpoint

### Warm-start mechanics (already supported by `train.py`)

`train.py:96-121` handles it:
```python
missing, unexpected = model_dict['denoiser'].load_state_dict(torch.load(cfg.ckpt_flow_path), strict=False)
if cfg.initialize_layers:
    model_dict['denoiser'].initialize_input_layer_x0h()      # copies input_layer -> input_layer_x0h
if cfg.partial_freeze:
    ...  # freeze everything, then unfreeze only the keys reported `missing`
```
Verified: the base checkpoint's block keys (`norm2`, `self_attn.*`, `cross_attn.{to_q,to_kv,to_out}`,
`mlp.*`, `adaLN_modulation.*`) match the conditioned model exactly. `cross_attn` is
`MultiHeadAttentionWeighted` in the conditioned model but keeps the same parameter names, so it
loads. `qk_rms_norm_cross` is False in both configs, so there are no `cross_attn.*_rms_norm`
keys on either side.

The keys reported **missing** (i.e. the genuinely new modules) are:
`input_layer_x0h.*`, `mask_hand_embedder.*`, `contact_encoder.*`, `fuse_x0_contact.*`, and per
block `norm_hand.{weight,bias}`, `cross_attn_mask_hand.*`, `cross_attn_hand.*`.
After fix 1 add `mask_hand_pos_emb` (a buffer, so it will not appear in `named_parameters`).

**`--load_dir` interaction:** `opt.load_dir` defaults to `opt.output_dir`, and `find_ckpt`
(`train.py:20-35`) finds no `misc_*.pt` in a fresh output dir, so `load_ckpt` stays `None` and
`Trainer.__init__` skips `load()`. The warm start survives. Just make sure the output dir is new.

### Stage 1 — conditioning only, no physics

Rationale: adding the occlusion conditioning and the physics losses simultaneously makes failures
impossible to attribute. This mirrors the lineage that already worked
(`flow_conditioned_no_losses` → `flow_conditioned_all_losses_*`).

Config: copy `configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned.json` to
`..._stage1.json` with `lambda_non_interpenetration_start: 0`, `lambda_non_interpenetration_max: 0`,
`lambda_contact: 0.0`. Keep `use_fp16: false`, `batch_size_per_gpu: 16`, lr `1e-5`.

```bash
python train.py \
  --config     configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage1.json \
  --output_dir outputs/teacher_v2_stage1_cond \
  --load_flow_weights --initialize_layers \
  --ckpt_flow_path /projects/gcaddeo/train_flow/TRELLIS/outputs/flow_final_norm_resume330/ckpts/denoiser_ema0.9999_step0350000.pt \
  --data_dir      /projects/gcaddeo/train_flow/TRELLIS/datasets/Leap_Hand,/projects/gcaddeo/train_flow/TRELLIS/datasets/Hands,/projects/gcaddeo/train_flow/TRELLIS/datasets/Hands_Google \
  --data_dir_test <HELD-OUT SPLIT>
```

Run until the CP3a hand-branch gate passes — **not** until `mse` plateaus; see §9 CP3-0 for why
that criterion is wrong here (mse is ~99% independent of the hand branch, and stage 2 keeps
optimising it anyway). For reference the old lineage reached `mse ≈ 0.16` on
Hands+Hands_Google by 10k steps and `≈ 0.102` by 44k.

### Stage 2 — add the fixed physics losses

Config `..._stage2.json` with `lambda_non_interpenetration_max: 200`, `lambda_contact: 0.1`
(revisit both from probe B's gradient ratios).

```bash
python train.py \
  --config     configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage2.json \
  --output_dir outputs/teacher_v2_stage2_physics \
  --load_dir   outputs/teacher_v2_stage1_cond --ckpt latest \
  --data_dir      <same three roots> \
  --data_dir_test <HELD-OUT SPLIT>
```

**Gotcha — the λ warmup is a no-op on resume.** `get_lambda_ni()`
([flow_matching.py:315-320](trellis/trainers/flow_matching/flow_matching.py#L315-L320)) compares
`self.step` against `lambda_non_interpenetration_warmup`, and `self.step` resumes at stage 1's
final step, which is far past 1000. λ therefore jumps straight to its max on the first stage-2
step. Either set `lambda_non_interpenetration_start == max` and accept that, or set
`lambda_non_interpenetration_warmup = <stage1_final_step> + 1000` so the ramp actually happens.
This bit the previous lineage too — every `all_losses` run ran at λ=200 from step one.

### Things to decide before submitting
- **Dataset mix.** The old teacher's last 11.5k steps saw Leap_Hand only, which is why its `mse`
  jumped from 0.102 to 0.155 at step 44001. Training on all three roots from the start is the
  cleaner choice, but confirm that is what you want the student to inherit.
- **Learning rate.** The base ran at 1e-4; every conditioned run used 1e-5. With new zero-init
  branches, 1e-5 is safe but slow; 5e-5 is worth a short trial.
- **fp16.** The teacher configs set `use_fp16: false` while `fp16_mode: 'inflat_all'`, which makes
  the 2^log_scale loss scaling a mathematical no-op and the run pure fp32 — roughly 2× slower but
  stable. Note `model.convert_to_fp16()` on resume is commented out at
  [basic.py:201-202](trellis/trainers/basic.py#L201-L202), so flipping `use_fp16` to true needs
  that path re-checked first. `outputs/flow_conditioned_distilled_all/nan_debug_autograd.log`
  records a NaN gradient at step 48407 in the fp16 distill run, so fp32 for the teacher is
  defensible.
- **Held-out split.** Needs creating; currently `--data_dir_test` defaults to the training path.

### SLURM
Model a training `.sbatch` on `tools/diagnose_physics_losses.sbatch` (partition `gpu-h200`,
`trellis_hopper`, `--export=NONE` so conda must be sourced). Two differences: the partition's
**1-day limit** means training must checkpoint and resubmit (`i_save` is 2000 steps and
`--load_dir <out> --ckpt latest` resumes cleanly), and multi-GPU needs `--gres=gpu:2` plus
`--num_gpus 2` (train.py spawns via `mp.spawn`; `batch_size = batch_size_per_gpu * world_size`).

---

## 6. Verification

1. **Diagnostics sanity gate** — §3, reproduce log.txt's step-54000 numbers.
2. **Warm start is a no-op at step 0** — stage 1's first logged `mse` should sit near the base
   model's loss on hand data (the old run started at 0.177), not spike. A spike means the
   zero-init branches were disturbed or the wrong checkpoint was loaded.
3. **Fix 1 actually landed** — re-run probe C on the stage-1 checkpoint. `mse/shuffled_hand`
   should now be clearly worse than `mse/baseline`. If it is still equal, the model is *still*
   ignoring hand position and the problem is elsewhere.
4. **Fix 2 actually landed** — probe A's `x0_rmse/as_trained` should now match
   `x0_rmse/correct` (they become the same code path), and `ni`/`contact` should vary with t.
5. **The physics losses now do work** — in stage 2, `ni_loss_raw` and `contact_raw` should fall
   *faster than* `mse`, unlike the current teacher where they co-move (§2a).
6. **Held-out snapshots** — compare `run_snapshot` renders on the held-out split, not the
   training set.

---

## 8. Execution log — 2026-08-06

### 8.0 Backup of the pre-change tree

```
/projects/gcaddeo/train_flow_conditioned/TRELLIS_backup_20260806_pre_teacher_v2
```
An rsync of the whole repo *except* `outputs/` (1 TB) and `__pycache__/`, including `.git`,
plus `BACKUP_GIT_HEAD.txt`, `BACKUP_GIT_STATUS.txt` and `BACKUP_UNCOMMITTED.patch` (the full
`git diff HEAD` at the time, 1350 lines). Every pre-fix source file is recoverable from there.
`outputs` inside it is a symlink back to the live tree, added so the probes could run from the
pristine copy while the working tree was being edited.

### 8.1 Step 1 — diagnostics

**Probe C (job 410, `--num_batches 20`) — finding A CONFIRMED, and worse than predicted.**

| variant | mse | vs baseline |
|---|---|---|
| baseline | 0.15193 | — |
| shuffled_hand | 0.15495 | +1.99% |
| zeroed_hand | 0.15350 | +1.03% |

The sanity gate passes: baseline `mse = 0.1519` reproduces `log.txt`'s 0.1516 at step 54000.

Reading: **deleting the hand entirely costs 1%**, and giving the model a *different* sample's
hand is barely distinguishable from giving it none. The hand branch is not merely
position-blind — it contributes almost nothing at all. Fix 1 is justified.

**Probes A, B, D (job 411)** — run from the backup copy against the same checkpoint,
`--num_batches 150 --num_batches_grad 5`. Output:
`outputs/diagnostics/physics_probe_flow_conditioned_all_losses_resume_32k_resume3_LEAP_denoiser_ema0.9999_step0054000.json`.
These only inform stage 2 (the λ values and the contact formulation), so stage 1 did not
wait on them.

**Probe A — finding B confirmed, and both physics terms shown to have no headroom.**

Batch-level, comparable to `log.txt` (`x_0` std measured at 0.430):

| metric | as_trained | correct | gt_floor |
|---|---|---|---|
| `x0_rmse` | 0.373 | 0.257 | 0 |
| `ni_loss_raw` | 1.37e-5 | 4.09e-5 | **1.09e-4** |
| `contact_raw` | 0.0199 | 0.0127 | **0.0113** |

*Finding B is confirmed exactly as predicted.* `x0_rmse/as_trained` is **flat** in t
(0.696 → 0.383 from t~0.05 to t~0.95, i.e. it does not improve as the problem gets easier),
while `x0_rmse/correct` falls from **0.357 at t~0.95 to 0.0344 at t~0.05** — a 10× improvement
precisely where the `(1-t)²` weighting concentrates. Against `x_0` std 0.430, the old estimator's
error was 87% of the signal at every timestep; the correct one is 8% at low t. Fix 2 is the
single largest change to what the physics losses actually see.

*The contact term is mostly measuring the decoder, and it overshoots at low t.* The contact
loss is `mean |sdf_obj|` over the ~75 annotated contact voxels, so in principle it is strong,
direct supervision on where the object surface must lie. The problem is what fraction of its
value carries that information.

`contact/gt_floor = 0.0113` — the **ground-truth latent**, decoded, already scores 0.0113 at its
own contact voxels. That is VAE round-trip plus discretisation, not model error: 0.0113 × 64 =
**0.72 voxels**, i.e. sub-voxel, and the contact points are themselves voxel indices. The truth
cannot score zero through this decoder.

Weighted by the `(1-t)²` the loss actually applies (bin masses from the real
`logitNormal(1.0, 1.0)` schedule):

| estimator | time-weighted contact |
|---|---|
| as_trained | 0.01992 |
| correct | 0.01260 |
| gt_floor | 0.01130 |

So after fix 2 there is **~11% real headroom** — small, but not zero. About 90% of the loss
value is a constant the frozen decoder imposes, which means a λ tuned against the total is
mostly weighting that constant.

The sharper problem is the sign at low t. Across bins with 57–189 samples each:

| t | n | correct | gt_floor | headroom |
|---|---|---|---|---|
| 0.25 | 57 | 0.01053 | 0.01143 | −8% |
| 0.35 | 114 | 0.01055 | 0.01108 | −5% |
| 0.45 | 189 | 0.01138 | 0.01141 | −0% |
| 0.65 | 391 | 0.01354 | 0.01139 | +19% |
| 0.75 | 507 | 0.01604 | 0.01125 | +43% |

Below t ≈ 0.45 the model is already at or slightly past the ground truth's own score, so the
objective is **not uniquely minimised at the truth** — driving it lower there moves the
prediction away from the correct answer. All the genuine headroom sits at t ≥ 0.55, where the
object is still noise-dominated.

(An earlier draft of this section claimed the term had *no* headroom, quoting the t~0.05 bin.
That bin holds ~2 samples — the schedule almost never draws t that low — and the claim was
wrong. The 11% figure above is the defensible one.)

**Recommendation: subtract the floor rather than drop the term.** Penalise only being worse
than what a perfect latent achieves through this decoder:
```python
with torch.no_grad():
    sdf_gt = self.ss_dec(x_0).float()
    floor = (contact_w * sdf_gt.abs().squeeze(1)).view(B, -1).sum(1) / den
contact_per_sample = F.relu(contact_per_sample - floor)
```
This keeps all of the real supervision, removes the constant decoder artifact, and makes the
term go silent exactly where the model has already matched the truth instead of pushing past
it. Cost: one extra no-grad `ss_dec` pass per step — the same computation probe A uses for
`gt_floor`.

*NI's target is already beaten by ground truth.* `ni/gt_floor = 1.09e-4` is **8× larger** than
`ni/as_trained = 1.37e-5`, and `frac_obj_inside` runs 0.0147 (gt) > 0.0119 (correct) > 0.0109
(as_trained). The ground-truth object interpenetrates the hand *more* than the model's
prediction does under this measure, so most of what NI penalises is VAE/decoder artifact.
Driving NI toward zero pushes the model **away** from ground truth.

This is the same defect as the contact term, and it is much more severe here: contact sits 11%
above its floor, NI sits **87% below** it. The same remedy applies — make the penalty relative
to what the ground-truth latent scores through the same decoder:
```python
with torch.no_grad():
    sdf_gt = self.ss_dec(x_0).float()
    ni_floor = (max_pen * torch.tanh(F.relu(margin - sdf_gt) / max_pen) * hand_mask) \
                   .view(B, -1).sum(1) / den
ni_per_sample = F.relu(ni_per_sample - ni_floor)
```
Note the probe measured the **un-margined** formulation. Fix 5's 1-voxel margin changes what is
measured (near-contact rather than overlap only), so re-run probe A against the margined version
before committing stage 2 — both the floor and the model's value will move.

**Probe B — gradient attribution (measured on the OLD code: `/w.sum()` normalisation,
λ_ni=200, λ_contact=0.1, un-margined NI, 5 batches per t-bin).** Ratios of physics gradient
norm to mse gradient norm:

| ratio vs mse | overall | t~0.05 | t~0.45 | t~0.65 | t~0.95 |
|---|---|---|---|---|---|
| `gn_ni/correct` | **4.60** | 0.45 | 6.05 | 6.40 | 4.65 |
| `gn_contact/correct` | 0.25 | 0.006 | 0.149 | 0.409 | 0.469 |

Two readings, one of them surprising:

1. **NI at λ=200 was not a weak nudge — it was 4.6× the mse gradient**, in every t bin, yet
   over 23,500 training steps `ni_loss_raw` barely moved (§2a) and the loss value stayed at
   ~5e-6. A gradient that large which produces no loss movement is *churn*: it points at
   decoder artifacts (consistent with probe A's finding that the ground truth scores 8× worse
   than the model — the term's minimum is not at the truth, so mse keeps pulling the model
   back). This upgrades the case for `ni_relative` from "cleaner" to "necessary": at any
   useful λ, an absolute NI injects large gradients that fight mse without reducing anything.
2. **Contact, with the correct estimator, has the profile a physics loss should have**: tiny
   gradient at low t (0.006 — the model is already at the floor there, nothing to push) and
   meaningful gradient at mid/high t (0.15–0.55) where probe A showed real headroom. This is
   the second, independent argument for keeping contact (floor-relative) rather than dropping it.

**Provisional stage-2 λ, to be finalised at CP4/CP5** (target: physics gradient ≈ 5–20% of
mse gradient; two scale changes stack — the fix-4 `mean()` shrinks terms ~8.9×, and the
floor-relative variants shrink them further by an amount only a re-run probe can measure):

- λ_contact: old-normalisation gradient ratio was 0.25 at λ=0.1 ⇒ ~0.1 gradient share needs
  λ≈0.04 old-scale ≈ **0.3–0.4** after the 8.9× mean() correction, *before* the floor-relative
  reduction. Start at **0.5** and check the logged `contact_raw`-to-`mse` gradient share.
- λ_ni: 4.6 at λ=200 ⇒ ~0.1 gradient share needs λ≈4 old-scale ≈ **35–45** new-scale, again
  before the floor-relative reduction — and with `ni_relative` on, the term is mostly zero, so
  the effective gradient will be far smaller. Start at **50**, only with `ni_relative: true`.
  **Do not run an absolute (non-relative) NI at any λ** — probe B says it is 96%+ artifact churn.

**Probe D — finding C confirmed, magnitude modest but free to remove.** With `p_uncond=0.1`
(`frac_dropped = 0.0946`, a clean sanity check):

| | dropped (n=227) | kept (n=2173) | ratio |
|---|---|---|---|
| `mse` | 0.1689 | 0.1474 | 1.15× |
| `contact` | 0.0541 | 0.0245 | **2.20×** |
| `ni` | 1.54e-5 | 1.93e-5 | 0.80× |

CFG-dropped samples predict worse in general (mse 1.15×), but their contact penalty is
**2.2×** — far beyond what the general degradation explains. Those samples are being scored
against a hand the network was not shown, which is an impossible objective, and the excess
lands on the unconditional branch that CFG extrapolates *away* from. Netting out the mse
effect, roughly **9% of the batch contact loss was pure impossible-objective bias**. NI shows
no such gap (0.80×), consistent with it being artifact noise (probe B). Fix 3 removes this for
free.

**Step 1 is complete.** All four probes have run; results are in
`outputs/diagnostics/physics_probe_flow_conditioned_all_losses_resume_32k_resume3_LEAP_denoiser_ema0.9999_step0054000.json`.

### 8.2 Step 2 — fixes applied

| fix | finding | where |
|---|---|---|
| 1a | A | `sparse_structure_flow.py` — `pos_emb` added to `x0h_tokens` under `pe_mode == "ape"` |
| 1b | A | `sparse_structure_flow.py` — 2-D APE buffer `mask_hand_pos_emb` over the 37×37 patch grid, added after `mask_hand_embedder` |
| 2 | B | `flow_matching.py` — `x0_pred = self._v_to_xstart_eps(x_t, t, pred)[0]` |
| 3 | C | `flow_matching.py` — `kept` mask read off `cond_dict['x0_hand']`, folded into `w` so physics is silent on CFG-dropped samples |
| 4 | D | `flow_matching.py` — `(w * per_sample).mean()` instead of `/ w.sum()` |
| 5 | 2a | `flow_matching.py` — `ni_margin` (default 1 voxel = 1/64); `obj_inside` uses `relu(margin - sdf_obj)`, `hand_mask` uses `sdf_hand < -margin` |
| 6 | 2a | `flow_matching.py` — `contact_mode: 'mask' \| 'soft'`; 'soft' weights `\|sdf_obj\|` by `exp(-touch[:,1]/contact_sigma)`. **Which one to use is still open — decide from probe A's `contact/gt_floor`.** |
| 7 | E | `image_conditioned.py` — `get_cond` no longer runs `encode_image(cond)`; the key is kept as a zeros tensor of the same shape. Verified no block reads `context['cond']`. `get_inference_cond` unchanged. |
| 8 | F | held-out split, see 8.3 |
| — | G | `sparse_structure_flow.py` — `forward()` builds a fresh `ctx` dict instead of mutating the caller's `cond` |
| — | H | `components.py` — failed reads are counted and named; recursion bounded at depth 10 |
| — | **I** | `sparse_structure_flow.py` — both hand `to_out` layers re-zeroed at the end of `initialize_weights()` |

Also new: `self.ni_margin`, `self.physics_time_power`, `self.contact_mode`, `self.contact_sigma`
read from the trainer config, and the physics block is **skipped entirely when both λ are 0**,
so stage 1 does not pay for two `ss_dec` passes per step.

**Scale note on fix 4.** Dropping the `/ w.sum()` normalisation reduces the physics loss
magnitude by roughly `E[w] ≈ E[(1-t)^2] · (1-p_uncond)`, which for the `logitNormal(1.0, 1.0)`
t-schedule is on the order of 0.1. λ_ni = 200 under the new normalisation is therefore
comparable to λ_ni ≈ 20 under the old one. Set the stage-2 λ from probe B's gradient ratios
rather than carrying 200 over unchanged.

### 8.3 Held-out split (fix 8 / finding F)

`tools/make_heldout_split.py` builds split roots that symlink the payload directories
(`renders_cond`, `data_pose_norm`, `ss_latents_sdf_pose`) back to the originals and carry their
own filtered `metadata.csv`, so nothing under `/projects/gcaddeo/train_flow/TRELLIS/datasets/`
is modified and no data is copied. Seeded, disjoint by `sha256`:

| dataset | train | test |
|---|---|---|
| Leap_Hand | 10329 | 319 |
| Hands | 912 | 32 |
| Hands_Google | 8022 | 248 |

Roots live in `datasets_split/<NAME>_{train,test}`.

### 8.4 Verification harness

`tools/verify_fixes.py` + `.sbatch` — five model-level checks (base checkpoint loads with only
the new modules missing; `initialize_input_layer_x0h` copies correctly; the hand branch is a
no-op at init; permuting the hand latent changes the output; `forward()` does not mutate the
cond dict). Check 4 builds a separate small model with `use_touch=False`, because the contact
encoder fuses `touch` into `x0h` before patchify and would otherwise confound a pure
permutation. Run `--no_hand_pe` to watch check 4 fail, which is the pre-fix behaviour.

Check 1's result on the base checkpoint: **489 tensors loaded, 355 missing, 0 unexpected**, and
every missing key belongs to the new conditioning modules — the warm start's key mapping is
exactly as §5 describes.

### 8.5 SLURM — the 72-hour request

**`gpu-h200` has a hard 24-hour limit for this account.** Verified rather than assumed:

```
$ sbatch --test-only --partition=gpu-h200 --time=72:00:00 ...
allocation failure: Requested time limit is invalid (missing or exceeds some limit)
$ sbatch --test-only --partition=gpu-h200 --qos=gpuh-long --time=72:00:00 ...
allocation failure: Invalid qos specification
```
`scontrol show partition gpu-h200` gives `MaxTime=1-00:00:00 AllowQos=gpu-h200`, and
`sacctmgr show assoc user=gcaddeo` lists only `cpu`, `gpu-h200`, `gpu-l40s` — the `gpuh-long`
QOS exists cluster-side but is not in our association. The only 7-day partition is `gpu-l40s`,
which would be a large throughput downgrade from H200s.

So 72 hours is delivered as **3 chained 24-hour segments** by `tools/train_teacher_v2.sbatch`:
each segment queues its successor with `--dependency=afterany:$SLURM_JOB_ID` *before* training
starts (so a SIGKILL at the wall still leaves a successor queued) and resumes with
`--load_dir <output_dir> --ckpt latest`. `touch outputs/teacher_v2_<stage>/STOP_CHAIN` ends the
chain. `--gres=gpu:2 --cpus-per-task=32 --mem=160G`; `batch_size = 16 × 2 = 32`.

Gotcha found the hard way: `--export=NONE` leaves a PATH without the Slurm bin dir, so the
self-resubmission has to call `sbatch` by absolute path
(`/opt/share/sw/amd/gcc-11.4.1/slurm-24.11.7/bin/sbatch`). Segment 1 of the first stage-1 run
(job 417) hit `sbatch: command not found` and its successor was queued by hand as job 418.

Stage 2's first segment runs `tools/set_stage2_warmup.py`, which rewrites
`lambda_non_interpenetration_warmup` to `<stage1 final step> + 2000` so the λ ramp actually
happens on resume instead of jumping to max on the first step (the §5 gotcha).

### 8.6 Throughput — the one open risk on stage 1

The old teacher's `log.txt` records one entry **per step**, including timings: the
`..._resume3_LEAP` run had a **median step time of 20.7 s** at the same
`batch_size_per_gpu = 16`, same 24-block fp32 model, same `use_checkpoint: true`.

Stage 1 should be somewhat faster — physics is off (two `ss_dec` passes saved) and fix 7 drops
one of the two DINOv2 forwards — so expect roughly **13–18 s/step**, i.e. **200–275 steps/h**
and **14k–20k steps in the 72-hour budget**. For reference the old lineage needed ~44k steps
to reach mse ≈ 0.102.

That comparison is less alarming than it first looks, because the old lineage spent its first
~32k steps at `batch_size_per_gpu = 4`. Counting *samples seen*:

- old lineage total ≈ 32k × 4 + 23.5k × 16 ≈ **504k samples**
- stage 1 at 17k steps × 32 ≈ **544k samples**

So the 72 h budget buys about the same data exposure as the entire previous lineage — but only
**~1/3 the number of gradient updates**. With batch 32 against the old batch 4, the linear
scaling rule argues the lr should rise accordingly; §5 already flagged "1e-5 is safe but slow;
5e-5 is worth a short trial". **This is the decision to make at CP1**, once the per-step mse
trend over the first 500 steps is visible — that is the cheapest possible evidence, and it
arrives ~2 h into the run rather than after a checkpoint.

### 8.7 Measured: what can and cannot be made faster (`tools/bench_tf32.py`, jobs 419–421)

Benchmarked on the real denoiser, fwd+bwd+optimizer step, so these are not spec-sheet numbers.

**TF32 is the only available lever, and it is worth +40%.** `train.py` never sets
`torch.backends.cuda.matmul.allow_tf32`, and PyTorch has defaulted it to `False` since 1.12 —
so every fp32 matmul avoids the tensor cores (~67 TFLOPS instead of ~495 dense TF32 on an
H200). TF32 keeps the fp32 exponent range and accumulates in fp32; only the matmul input
mantissa drops 23 → 10 bits. It is in a completely different risk class from fp16, which is
what produced the NaN in `flow_conditioned_distilled_all`.

| setting | s/step | speedup | steps in 72 h |
|---|---|---|---|
| as configured (fp32, deterministic) | 18.30 | 1.00× | 14,164 |
| **+ TF32 matmuls** | **13.06** | **1.40×** | **19,849** |
| + TF32 + non-deterministic | 12.95 | 1.41× | 20,008 |

Note the determinism setting costs only **0.7%** — not worth touching, despite
`use_deterministic_algorithms(True, warn_only=True)` not actually delivering determinism
(the memory-efficient attention backward warns and stays non-deterministic).

**VRAM is at 29–46% and that is correct, not waste.** Sweeping batch × checkpointing with TF32 on:

| config | s/step | **samples/s** | peak GiB | VRAM |
|---|---|---|---|---|
| batch 16/GPU, checkpoint=True | 13.05 | **1.23** | 41.0 | 29% |
| batch 32/GPU, checkpoint=True | 26.00 | **1.23** | 73.4 | 53% |
| batch 48/GPU, checkpoint=True | 38.85 | **1.24** | 105.8 | 76% |
| batch 16/32/48, checkpoint=False | OOM | — | — | — |

Two conclusions, both firm:

1. **samples/s is flat at 1.23 across 29%, 53% and 76% VRAM** — step time scales perfectly
   linearly with batch. The GPU is fully compute-saturated at batch 16; it is compute-bound,
   not memory-bound. The idle VRAM is not convertible into throughput. A larger batch buys
   only lower gradient noise while costing proportionally fewer optimizer steps, which is the
   resource this run is short of. **Keep `batch_size_per_gpu: 16`.**
2. **`use_checkpoint` is mandatory, not a tunable.** `checkpoint=False` OOMs at batch **16**,
   the smallest size tested, on a 140 GiB card — re-verified in isolation (job 421) in a fresh
   process to rule out an unreleased-memory artefact from the sweep loop. It is not a
   30%-compute-for-memory trade one could opt out of; it is what makes this configuration
   trainable on this hardware at all. (An earlier note in this file claimed there was headroom
   to disable it. That was wrong.)

**Open decision.** Adopting TF32 means one line in `train.py`
(`torch.backends.cuda.matmul.allow_tf32 = True`) and restarting stage 1, forfeiting the steps
done so far to gain ~40% on everything after. Weigh against CP1's actual loss trend before
deciding — if lr 1e-5 is barely moving mse, the lr change and the TF32 change should be made
in the same restart rather than in two.

**Incidental find:** the new dataset error reporting (finding H) surfaced a corrupt render on
its very first firing —
`datasets_split/Leap_Hand_train/MIRACLE_POUNDING_4: OSError: image file is truncated` — the
same "image file is truncated" that probe A printed mid-run. The retry path handles it; if the
counter climbs beyond a handful of instances, audit `renders_cond` for truncated PNGs.

### 8.8 Stage-1 run 2 gate trajectory, CP4 measurement, and the stage-2 λ — 2026-08-08

Run 2 (TF32, batch 32, `i_save` 1000 from segment 2) gated on raw weights per CP3a:

| step | hand/image ratio | verdict |
|---|---|---|
| 2000 | 0.054 | NOT READY |
| 4000 | 0.131 | EMERGING |
| 5000 | 0.177 | EMERGING |
| 7000 | **0.250** | **READY** (job 432; shuffled_hand +4.77%, zeroed_image +19.08%) |

(Step 6000 went ungated — session restart killed the watcher; no decision hinged on it.
Segment-boundary detail: job 423 was cancelled at step ~5100 once it was clear it could not
reach its next `i_save=2000` checkpoint before the 24h wall — everything after step 4000 was
unsalvageable, so waiting only delayed segment 2.)

**CP4-2 (job 433):** `tools/diagnose_physics_losses.py` extended with `--ni_margin`
(default 1/64, matching fix 5) and floor-relative reporting: probe A adds `ni_rel/*`,
`contact_rel/*` and `*_rel_stage2/*` (the exact stage-2 objective — margined, floor-relative,
fix-4 `mean()`), probe B adds `gn_ni_rel@1` / `gn_contact_rel@1` at unit λ (stage-1 config
λ=0 would have zeroed the legacy measurement). Run on `denoiser_step0007000.pt`, 3-root
train mix. Findings:

- The relative terms are well-posed: `contact_rel/correct` is ~1e-4 at t=0.05 (silent where
  the model already matches the floor) rising to ~0.02 at t=0.95 where the headroom is.
  `ni_rel/correct` has the same shape. Expected stage-2 raw magnitudes:
  `contact_rel_stage2/correct ≈ 1.8e-4`, `ni_rel_stage2/correct ≈ 1e-5`.
- Unit-λ gradient shares vs mse: `gn_contact_rel@1/mse = 0.103`, `gn_ni_rel@1/mse = 0.0035`.

**Chosen λ (target 5–20% gradient share): `lambda_contact = 1.0` (≈10.3%),
`lambda_non_interpenetration_max = 30` (≈10.6%),** with `ni_relative: true`,
`contact_relative: true`, `contact_mode: "mask"`, `i_save: 1000` in the stage-2 config.
§8.1's provisional guesses (0.5 / 50) bracketed both.

Stage 2 launched from the step-8000 checkpoint (chain stopped at that boundary so no
computed steps were discarded); stage-1 chain ended via STOP_CHAIN + scancel, letting gate
job 428 fire on the final checkpoint as a confirmation reading.

**λ_ni bump REVERTED same day — stayed at 30.** After the probe, the training log showed NI
resuming a clear downtrend at λ=30 (7.97e-7 → 7.41e-7 → 7.13e-7 over steps 13k→15k, i.e.
−7.1% and −3.8% per half-block): the probe's flat 9k-vs-14k reading was lumpy convergence,
not a stuck equilibrium, so the intervention criterion (plateaued AND below-band) was no
longer met and the config went back to 30 before any segment ran at 60. Lesson: don't
diagnose a plateau from two probe snapshots when the per-step log disagrees.

**Original bump rationale (superseded):** λ_ni raised 30 → 60 (2026-08-09, effective at the segment-2→3 boundary). Probe on the
step-14000 checkpoint (job 440): contact is converging to its floor by design
(`contact_rel_stage2/correct` 1.85e-4 @7k → 1.48e-4 @9k → 1.18e-4 @14k, share steady at
5.5%), but NI plateaued (5.95e-7 @9k → 7.1e-7 @14k, flat) with a share of only 3.4% at
λ=30 — meeting the pre-registered criterion (share < 5% band floor AND value plateaued).
60 puts NI at ≈6.8% share. Boundary gate at step 13000 (job 438): hand/image **0.469**,
zeroed_hand +5.93% — physics re-accelerated hand-branch growth (0.319 @9k → 0.469 @13k),
confirming §9 CP3a's prediction that physics, not mse, is what develops the branch.

**Post-launch confirmation (2026-08-08 ~10:30):** gate 428 on the final stage-1 checkpoint
read **0.310 READY** (shuffled_hand +6.36%), so 7000's 0.250 was not a threshold graze.
Stage 2 (job 434) resumed cleanly — no mse spike, 15.2 s/step (physics overhead only ~4%),
λ ramp verified (ni_loss/ni_loss_raw ≈ 27 at step 9000 → 30 at 10000). Over the first 1000
steps (first-100 vs last-100 means): mse −2.4%, contact_raw −5.4%, ni_loss_raw −8.8% —
**physics falling 2–4× faster than mse**, satisfying §6 item 5 so far, which the old
teacher never did. Next checks: same trend at the segment boundary (~step 13.6k), and the
CP3a probe on the **EMA** weights at end of stage 2 (the EMA is what ships).

### 8.9 Stage-2 stopping decision — pre-registered criterion (2026-08-13)

Stage 2 state at ~step 32k: hand/image gate flat at ~0.73 since 23k (converged);
`contact_raw` still drifting down a few %/1k; NI at its floor. Meanwhile the sampling-time
a/b (jobs 452/456, step-29k EMA, held-out, 64 samples, `outputs/diagnostics/
ab_guidance_4arm.json`) showed: production guidance (α=10) inert; the fixed-cost greedy
guidance (`sample_guided_v2`) removes 97% of excess penetration; **complete OC-Flow**
(`sample_oc_flow`, discrete adjoint, 4 outer iters) additionally cuts excess contact 31%
(hit@1vox 42%→49%) at flat IoU — i.e. inference-time optimization already closes most of
the physics gap that further training would chase.

**Segment 448 (ends ~05:30 2026-08-13, ~step 37.5k) is the last segment; nothing queued
after it.** Decision rule, registered before seeing the data: job 459 reruns the a/b
(unguided + oc_flow, same seed/data/n as the 29k run) on the final checkpoint. **If
unguided floor-relative `contact_abs` does not improve by ≥10% vs the 29k value
(1.110e-2), training stops and the final checkpoint freezes for distillation.** Gate 460
measures the CP3a ratio on the **EMA** weights (all prior gates were raw). Job 449 gates
raw @32k as an extra trajectory point.

Housekeeping: probe 450 failed as expected (queued for a step-33000 ckpt that never
existed — 445 hung at 32k after its checkpoint save and was cancelled; successor 448
resumed cleanly). The two chronically truncated PNG instances (`MIRACLE_POUNDING_4`,
`Animal_Planet_Foam_2Headed_Dragon_13`) are dropped from `Leap_Hand_train/metadata.csv`
(backup: `metadata.csv.bak_20260813`). If another segment ever hangs right after a
checkpoint+snapshot boundary, suspect the snapshot sampling path.

**Boundary 1 outcome (2026-08-14, job 459, final ckpt = EMA step 37000; 448 ended in a
graceful wall kill at step 37.5k):** unguided floor-relative `contact_abs` =
1.8095e-2 − 1.1964e-2 (paired floor) = **6.13e-3**, vs the 29k baseline 1.110e-2 —
a **45% improvement**, far beyond the ≥10% bar ⇒ **CONTINUE** (the expected freeze did
not happen; the rule is the authority). Corroborating, same run vs 29k: hit1v
0.423→0.519 (floor 0.76), occ_iou_gt 0.497→0.549, pen excess roughly halved, raw CD
0.0654. OC-Flow's remaining edge shrank correspondingly (contact excess 6.1e-3→5.1e-3,
hit1v +2.3pp, pen to floor, fidelity flat) — the teacher is absorbing at training time
what guidance was adding at sampling time. EMA gate on 37k: hand/image 0.672, READY
(job 460). Continuation launched: job 470 = one 24h stage-2 segment resuming from 37k
(submitted `stage2 4 4`, so no self-requeue); boundary packet queued after it: job 471
(2-arm a/b, 64 samples, seed 1337 → `ab_guidance_2arm_final2.json`) and job 472 (EMA
gate). **Iterated criterion, registered 2026-08-14 before seeing 471's data: STOP and
freeze unless 471's unguided floor-relative `contact_abs` < 0.9 × 6.131e-3 =
5.52e-3.** (Floors are per-run but samples/seed are identical, so deltas are paired.)

**Boundary 2 outcome (read 2026-08-18; job 471 on EMA step 42000, segment 470 ended in a
clean wall kill at 42.5k):** unguided floor-relative `contact_abs` = 1.6504e-2 − 1.1964e-2
= **4.54e-3** vs the bar 5.52e-3 (−26% vs 37k's 6.13e-3) ⇒ **CONTINUE**. Every other
metric moved the same way, unguided 37k→42k: hit1v 0.519→0.558, occ_iou 0.549→0.571,
raw CD 0.0654→0.0616, F@0.02 0.472→0.508, EMD −3%, pen excess −5%; NC flat. **OC-Flow
is now fully absorbed**: on 42k the oc_flow arm ties unguided (contact excess 4.68e-3
vs 4.54e-3, hit1v 0.558 = 0.558) at −1.3pp IoU — guidance no longer buys anything on the
current teacher. EMA gate @42k: 0.747, READY (job 472). Rate of improvement is
decelerating (−45% then −26% per ~5k steps). Continuation: job 476 (one 24h segment from
42k, `stage2 5 5`), boundary packet 477 (a/b → `ab_guidance_2arm_final3.json`) + 478
(EMA gate). **Iterated criterion, registered 2026-08-18 before seeing 477's data: STOP
and freeze unless 477's unguided floor-relative `contact_abs` < 0.9 × 4.541e-3 =
4.09e-3.** Successor segment 479 is PRE-QUEUED behind 476 (2026-08-19) so a CONTINUE verdict loses no GPU time; on a STOP verdict `scancel 479` (or touch STOP_CHAIN) — the rule, not the queue, decides. Note the chain went idle Aug 15–18 (no session to relaunch it) — future
continuations should queue the successor segment together with the boundary packet.

**Boundary 3 outcome (2026-08-19, job 477 on EMA step 47000; segment 476 clean wall
kill, successor 479 pre-queued so no idle time):** unguided floor-relative `contact_abs`
= **3.659e-3** vs the bar 4.09e-3 (−19.4% vs 42k's 4.541e-3) => **CONTINUE**. Trend per
5k steps: −45%, −26%, −19% — decelerating but still clearing the −10% bar. Companions,
42k→47k: hit1v 0.558→0.576, occ_iou 0.571→0.585, CD 0.0616→0.0603, F@0.02 0.508→0.526,
EMD flat; OC-Flow again adds nothing (contact excess 3.94e-3 ≥ unguided's). EMA gate
@47k: 0.825, READY (job 478). Segment 479 runs to ~52k (ends ~13:27 Aug 20); boundary-4
packet queued: 480 (a/b → `ab_guidance_2arm_final4.json`) + 481 (gate); successor 482
pre-queued (cancel on STOP). **Iterated criterion, registered 2026-08-19 before 480's
data: STOP and freeze unless 480's unguided floor-relative `contact_abs` <
0.9 × 3.659e-3 = 3.29e-3.**

**Boundary 4 outcome (2026-08-20, job 480 on EMA step 52000):** unguided floor-relative
`contact_abs` = **2.885e-3** vs the bar 3.29e-3 (−21.2% vs 47k) => **CONTINUE**. Trend
per 5k steps: −45, −26, −19, −21% — settled around −20%/segment. Reconstruction guard
(agreed with the user 2026-08-20: a contact gain that costs geometry is a stop-and-review
signal regardless of the rule) passes: 47k→52k IoU 0.585→0.593, CD 0.0603→0.0593, F@0.02
0.526→0.546, hit1v 0.576→0.601, EMD −2.4%, pen excess 5.5e-5→3.7e-5. **OC-Flow is now
counterproductive** (guided contact excess 4.18e-3 > unguided 2.88e-3) — retire it for
this teacher at deployment. EMA gate @52k: 0.888 READY (job 481; the gate is a regression
alarm only, not an objective — see the same discussion). Segment 482 runs 52k→~57k (ends
~13:22 Aug 21); packet 483 (a/b → `ab_guidance_2arm_final5.json`) + 484 (gate);
successor 485 pre-queued (cancel on STOP). **Iterated criterion, registered 2026-08-20
before 483's data: STOP and freeze unless 483's unguided floor-relative `contact_abs`
< 0.9 × 2.885e-3 = 2.60e-3.**

**TEACHER FROZEN (2026-08-20, user decision).** After boundary 4 (CONTINUE by the
rule, −21%/segment and not yet converged) the user froze the teacher anyway to focus
effort on the distillation contribution: **frozen teacher = EMA step 52000**
(`outputs/teacher_v2_stage2_physics/denoiser_teacher_v2_FROZEN.pt` symlink;
provenance in `FROZEN_TEACHER.txt`; STOP_CHAIN sentinel set; segment 482 and packet
483/484/485 cancelled — ckpts 53k/54k on disk are UNEVALUATED leftovers, not the
teacher). Frozen-teacher credentials: contact excess 2.885e-3, hit1v 0.601, IoU 0.593,
CD 0.0593, F@0.02 0.546 (boundary 4); EMA gate 0.888. Deployment refresh queued: job
486 re-converts 52000 into the inference pipeline + regenerates the dex meshes
(`meshes_results_marching_cubes_teacher_v2_52k/`), job 487 the canonical ICP eval
(`summary_teacher_v2_52k_dex_total_total_icp.json`). The §8.9 iterated-rule ledger
CLOSES here. Next phase: distillation + the ICRA deployment study — see ICRA_PLAN.md.

### 8.10 Sampling-time guidance campaign — see EVAL_GUIDANCE.md (2026-08-13)

The guidance implementation (fixed greedy + complete OC-Flow), the four-arm a/b
evaluation (physics + CD/NC/F/EMD, raw and ICP-aligned, full mix + YCB-only + old-teacher
comparison), the HTML galleries, all result JSONs, in-flight jobs, and the deployment/
distillation queue are documented in **EVAL_GUIDANCE.md** — read it together with §8.9's
stopping rule when picking up this project. Headline: guidance-as-is is inert; the fixed
cost eliminates sampling-time penetration; complete OC-Flow additionally cuts excess
contact 31% at flat IoU; unguided remains the most geometrically faithful.

---

## 9. Operational runbook — what to do next, in order

Written 2026-08-06 ~13:45, while stage-1 segment 1 (job 417) is running. Each checkpoint below
says when to act, what to look at, and what "good" looks like. A session executing this needs
no other context than this file.

### CP1 — warm-start check (as soon as `outputs/teacher_v2_stage1_cond/log.txt` exists)

```bash
head -5 outputs/teacher_v2_stage1_cond/log.txt
```
- **Good:** first `mse` in the 0.15–0.25 range (the old lineage started at 0.177 from the same
  checkpoint; the held-out mix differs slightly, so allow slack).
- **Bad:** mse > 0.4 or NaN → the warm start is broken. Immediately
  `touch outputs/teacher_v2_stage1_cond/STOP_CHAIN` and `scancel 417 418`, then debug
  (first suspects: wrong ckpt path in `tools/train_teacher_v2.sbatch`, or finding-I re-zeroing
  regressed).
- Also note `Speed: X steps/h` from `slurm-teacher_v2_stage1-417.out`. 72 h buys `72·X` steps;
  the old lineage needed ~44k to reach mse ≈ 0.102. If `72·X < ~35k`, consider the lr 5e-5
  trial (§5) or fp16 — but do not restart a healthy run for this without weighing lost hours.

### CP2 — probe B/D results (job 411; output also lands in
`outputs/diagnostics/physics_probe_*.json`)

```bash
grep -A30 "PROBE B" slurm-physics_probe_abd-411.out | grep -vE "warn|autocast"
grep -A15 "PROBE D" slurm-physics_probe_abd-411.out | grep -vE "warn|autocast"
```
Use probe B's `gn_ni/mse` and `gn_contact/mse` ratios (for the **correct** x0 variant) to pick
stage-2 λ: scale λ so the physics gradient is roughly 5–20% of the mse gradient. Remember two
scale changes stack: fix 4 shrank the terms ~8.9× (measured `E[(1-t)²]·0.9 = 0.113`), and the
floor-relative variants (§8.1) shrink them further. Record the chosen λ in this file.

### CP3-0 — how long should stage 1 actually run? (READ THIS FIRST)

**"72 hours" was a scheduling constraint, not a training-duration decision**, and this document
briefly conflated the two — calling 72 h "the budget" and quoting progress as a percentage of
it. That framing is wrong and has been removed. `gpu-h200` caps a *submission* at 24 h, so a
chain of 3 is 72 h; when it ends, `sbatch tools/train_teacher_v2.sbatch stage1 1 3` resumes
from the latest checkpoint and gives another 72 h. **The chain is one unit of compute, not an
endpoint.**

Duration should be set by **the CP3a gate alone**, not by the clock and **not by mse**.

§5's "run until mse plateaus" is a poor criterion here and should not be used:

- **mse is ~99% independent of the hand branch.** Probe C measured that zeroing the hand
  entirely changes mse by only ~1% — it is dominated by the pretrained image pathway. Using it
  to judge when the *hand branch* is ready means reading a number that is almost entirely about
  something else.
- **Stage 2 keeps optimising mse anyway.** It adds physics, it does not replace mse. Any
  remaining mse progress happens after the switch regardless, so waiting for convergence buys
  nothing and costs the physics losses steps they could have used to shape the model.
- The two-stage split exists for exactly one reason: physics is meaningless against a branch
  that contributes nothing (‖W‖ = 0 and +0.00% effect at init). The gate measures precisely
  that condition. Once it is satisfied, the reason for the split has expired.

The one genuine counter-consideration is cost, not convergence: physics adds 2–3 `ss_dec`
passes per step, so switching early slows every remaining step. **This has not been measured —
benchmark it before stage 2** (extend `tools/bench_tf32.py`). It argues against switching while
the gate is barely above zero; it does not argue for waiting on mse.

Reference, for choosing a target. The old lineage hit mse ≈ 0.102 at step 44k and ran 55.5k
total, at batch 4 → 16. teacher_v2 runs batch 32 at ~245 steps/h (~15,000 useful steps per
72 h chain after the i_save=1000 boundary loss):

| target | steps | wall clock | chains |
|---|---|---|---|
| match old lineage's **samples seen** (~504k) | ~15,700 | ~64 h | 1 |
| middle | 30,000 | ~122 h | 2 |
| match old lineage's **gradient updates** (55.5k) | 55,500 | ~227 h | ~3.5 |

One chain buys roughly the old lineage's entire *data exposure* but ~1/3 of its *gradient
updates*. Which matters more is open; raising lr to 5e-5 is a bet that fewer, larger updates
suffice. **Decide this explicitly rather than letting the chain's end decide it.**

### CP3a — the stage-1 → stage-2 gate (run on each new checkpoint, ~6 min)

**This replaces §5's "run until mse plateaus", which cannot fire.** At the measured throughput
the 72 h budget buys ~17k steps and the old lineage needed ~44k to plateau, so "plateau" would
never trigger and the stage boundary would be set by the clock rather than by evidence.

The measurement that actually gates physics is whether the hand branch carries signal yet.
`cross_attn_hand.to_out` is zero-initialised, so at step 0 the branch contributes *exactly*
nothing (verified: `max|out(hand) − out(no hand)| = 0.000e+00`). Physics applied then is scored
against a hand the network structurally cannot see — the same impossible objective probe D
measured for CFG-dropped samples, but for 100% of the batch instead of 10%. Probe C measures
exactly when that stops being true.

```bash
sbatch tools/check_hand_gate.sbatch            # newest stage-1 EMA checkpoint, held-out split
```

**Verdicts are a ratio to the image pathway**, measured in the same probe run, not an absolute
mse percentage:

| gate | hand / image | action |
|---|---|---|
| **< 0.10** NOT READY | hand barely registers | keep training stage 1 |
| **0.10–0.25** EMERGING | live and growing | keep training; see CP3-0 |
| **> 0.25** READY | hand is a meaningful fraction of the dominant pathway | go to CP4, then stage 2 |

The first version of this gate used absolute thresholds (5% / 15%) picked only as "clearly above
the old teacher's 2%". That was arbitrary. Running the image ablation (job 429, step 4000) gave a
real denominator:

| variant | mse | vs baseline |
|---|---|---|
| baseline | 0.150441 | — |
| shuffled_hand | 0.154224 | +2.52% |
| zeroed_hand | 0.152573 | +1.42% |
| **zeroed_image** | 0.179431 | **+19.27%** |
| zeroed_maskhand | 0.150445 | **+0.00%** |
| zeroed_all | 0.181682 | +20.77% |

Three things fall out of it:

1. **The hand is worth 7.4% of what the image is worth.** That is not feeble — it is a
   secondary pathway, which is correct: the image determines *what the object is*, the hand
   only constrains *how it is held*. `zeroed_all` ≈ image + hand, so the contributions are
   near-additive, a useful sanity check.
2. **The 2-D hand-mask branch contributes exactly nothing** (`zeroed_maskhand = +0.00%`),
   despite fix 1b giving it a 2-D APE and its weights growing to ‖W‖ ≈ 1.0. Almost certainly
   redundant against the far richer 3-D `x0_hand`. **Fix 1b appears to have bought nothing.**
3. **teacher_v2 @ step 4000 already exceeds the old teacher @ step 54000** on both hand
   metrics (+2.52%/+1.42% vs +1.99%/+1.03%), at ~1/26th the branch weight norm. Indicative
   rather than exact — the old teacher's probe ran on Leap_Hand only, the gate on the
   3-dataset held-out mix — but the baselines are close (0.152 vs 0.150).

**The tension this exposes.** mse rewards hand usage at only ~1.4%, so stage 1 — which trains
on mse alone — has very little incentive to develop the hand branch. The physics losses are
precisely the terms that *would* strongly reward it. Extrapolating the current rate
(+1.48pp per 2000 steps) puts an absolute 15% at ~step 20,000, beyond one chain. That is why
the threshold is now a ratio and why 0.25 (≈5% absolute at present) is the target rather than
15%: waiting longer asks mse to do something it is poorly suited for.

**Gate on the RAW weights, not the EMA — this bit us on the first run.** `ema_rate = 0.9999`
means the EMA is still **82% initial weights at step 2000** (`0.9999^2000 = 0.819`), and
`cross_attn_hand.to_out` is zero-initialised (finding I), so the EMA's copy of the hand branch
is ~10× weaker than the live one early in training. The first gate run (job 425, on the EMA
checkpoint) returned a flat **+0.00%** — while the live weights had already grown the branch
from exactly 0 to `‖W‖ = 1.21`. It was measuring EMA lag, not the model.

| checkpoint | `cross_attn_hand.to_out` ‖W‖ | vs main `cross_attn` |
|---|---|---|
| OLD teacher, step 54k (converged) | **31.99** | **96.2%** |
| teacher_v2 raw, step 2k | 1.21 | 3.6% |
| teacher_v2 EMA, step 2k | 0.12 | 0.4% |

The live weights are also the ones that will receive stage-2's physics gradients, so they are
what the gate should read. `--weights raw` is now the default; check `--weights ema` separately
at the **end** of stage 1, since the EMA is what actually gets deployed.

**That table is also the clearest statement of finding A yet.** The old teacher's hand branch
was not underdeveloped — at ‖W‖ = 31.99 it was **as large as the main cross-attention** — and
shuffling the hand still moved its output only 1.99%. A fully-grown branch doing nothing is
exactly what permutation-invariance predicts: without positional embedding, attention over the
4096 hand tokens can only produce a bag-summary that is nearly identical for every hand. Weight
norm is therefore an "is it learning" signal, **not a target to match** — with fix 1 a much
smaller branch should achieve a much larger functional effect.

**First real gate reading (job 426, step 2000, raw weights):**

| | old teacher (54k, converged) | teacher_v2 (2k, raw) |
|---|---|---|
| `hand.to_out` ‖W‖ | 31.99 | 1.21 |
| mse/shuffled_hand | +1.99% | **+1.04%** |
| mse/zeroed_hand | +1.03% | +0.64% |

Verdict `NOT READY (+1.04%)`, which is the expected answer at step 2000 of ~17,700. The
encouraging part is the ratio: half the functional effect from 1/26th the weight norm, i.e.
~14x more effect per unit of weight than the old teacher reached after 54k steps. Treat that as
directional (effect-vs-norm is not linear, and the models are at very different stages), but it
is the first *in-training* evidence that fix 1 works, and the branch is demonstrably alive
(+1.04%) rather than inert (+0.00% on the EMA).

Re-gate at ~6k and ~12k steps; the threshold question only becomes meaningful there.

Notes:
- Read-only and safe to run while training continues; it only reads checkpoints and writes
  under `outputs/diagnostics/`.
- It writes to its own `handgate_*.json`. `diagnose_physics_losses.py` writes one JSON per
  (run, ckpt) and **a probe subset overwrites the whole file** — that is how job 411's A/B/D
  destroyed job 410's probe-C results. The gate avoids repeating that.
- Nothing exists to gate on before step 2000 (`i_save = 2000`).
- The verdict logic was validated by replaying the old teacher's real probe-C numbers through
  it: it returns `NOT READY (+1.99%)`, which is the correct answer for that model.

### CP3 — daily glance while stage 1 runs (~3 days)

```bash
squeue -u gcaddeo                      # a teacher_v2_stage1 job RUNNING or PENDING
tail -3 outputs/teacher_v2_stage1_cond/log.txt
```
- mse should fall steadily (reference: 0.177 → 0.160 in the first 10.5k steps for the old run).
- If no job is running or pending and mse has not plateaued: a chain link failed — check the
  newest `slurm-teacher_v2_stage1-*.out` and resubmit
  `sbatch tools/train_teacher_v2.sbatch stage1 <next_segment> 3`.
- The 3-segment chain ends after ~72 h total. If mse is still falling and more budget is
  wanted, submit another chain: `sbatch tools/train_teacher_v2.sbatch stage1 1 3` (it resumes
  from the latest checkpoint automatically; segment numbering restarts but behaviour is
  identical).

### CP4 — stage-1 exit checks (when the CP3a gate reads READY — see CP3-0)

1. The CP3a gate must read **READY** (`sbatch tools/check_hand_gate.sbatch`). That is the
   same check as §6 item 3: fix 1 landed iff `mse/shuffled_hand` is clearly worse than
   `mse/baseline`. If the gate never leaves NOT READY, stop — the model is still ignoring the
   hand, and stage 2 would be pointless.
2. Re-run probe A with the **margined** NI formulation (the §8.1 floors were measured
   un-margined) to set `ni_margin`'s floor and confirm `contact_relative`/`ni_relative` make
   sense. This needs a small probe edit to use the margin — see §8.1 note.

### CP5 — launch stage 2

1. Edit `configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage2.json`:
   λ from CP2, plus `"contact_relative": true, "ni_relative": true` (assuming CP4-2 confirms).
   Leave `lambda_non_interpenetration_warmup` alone — the job rewrites it automatically via
   `tools/set_stage2_warmup.py`.
2. `sbatch tools/train_teacher_v2.sbatch stage2 1 3`
3. §6 item 5 is the success criterion: `ni_loss_raw` and `contact_raw` must fall **faster than**
   mse. If they merely co-move again, the physics terms are still not doing work.

### Standing notes
- Job 418 (stage-1 segment 2) was submitted before the `python -u` fix: its stdout will also
  lag by buffering. Trust `log.txt` and `tb_logs`, not the slurm file.
- `outputs/teacher_v2_stage1_cond/STOP_CHAIN` stops the chain; delete it before resubmitting.
- The pre-change tree lives in `TRELLIS_backup_20260806_pre_teacher_v2` (§8.0).

---

## 7. Files touched

| file | state |
|---|---|
| `tools/diagnose_physics_losses.py` | new; **run** — probe C (job 410), probes A/B/D (job 411) |
| `tools/diagnose_physics_losses.sbatch` | new; run |
| `tools/verify_fixes.py` / `.sbatch` | **new**; run, all 5 checks PASS (job 416) |
| `tools/make_heldout_split.py` | **new**; run, split built under `datasets_split/` |
| `tools/set_stage2_warmup.py` | **new**; runs automatically on the stage-2 cold start |
| `tools/train_teacher_v2.sbatch` | **new**; the chained 24h-segment training job |
| `configs/generation/..._sdf_conditioned_stage1.json` | **new** — physics off |
| `configs/generation/..._sdf_conditioned_stage2.json` | **new** — physics on; λ still to be set from probe B |
| `trellis/models/sparse_structure_flow.py` | **modified** — fixes 1a, 1b, finding G, finding I |
| `trellis/trainers/flow_matching/flow_matching.py` | **modified** — fixes 2, 3, 4, 5, 6 (mode switch), physics skipped when λ = 0 |
| `trellis/trainers/flow_matching/mixins/image_conditioned.py` | **modified** — fix 7 |
| `trellis/datasets/components.py` | **modified** — finding H |
| `TEACHER_RETRAIN.md` | this file |

The pre-change state of every one of these is preserved under
`/projects/gcaddeo/train_flow_conditioned/TRELLIS_backup_20260806_pre_teacher_v2` (§8.0).
