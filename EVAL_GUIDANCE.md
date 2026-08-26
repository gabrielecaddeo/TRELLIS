# Sampling-time physics guidance: implementation, a/b evaluation, and open state

**Written 2026-08-13; §7 is the LIVE results ledger (updated through 2026-08-24).**
Self-contained handoff for the guidance/evaluation campaign that followed the
teacher_v2 stage-2 training (see TEACHER_RETRAIN.md §8.8–8.10 for the training
side). §§1–6 are the 2026-08-13 state (historical but still accurate on code/
gotchas); §3's job table and §4's queue are SUPERSEDED by ICRA_PLAN.md (rewritten
2026-08-24: merged four-axis framing + multi-view phases) and by §7's running
entries. A new session: read ICRA_PLAN.md first, then §7 here.

---

## 1. What exists now (code, all in THIS repo — the train repo)

### Samplers — `trellis/pipelines/samplers/flow_euler.py`
| method | what it is |
|---|---|
| `sample_velocity` | FIXED (was broken): had a sign error (physics gradient *ascent* — pushed objects INTO the hand) and decoded the noisy x_t instead of x̂0. Now descent + x̂0 (DPS-style), honors `alpha_vel` (old hardcoded 5000 override removed; default flagged UNCALIBRATED), `guidance_skip=5`, `alpha_vel=0` = exact plain Euler. Only reachable via the *unconditioned* pipeline — production never used it. |
| `sample_velocity_conditioned` | The production guidance (matches the inference repo's operative version): x̂0-decode ✓, correct signs ✓, CFG-aware ✓. Aligned: `guidance_skip` param (was hardcoded 30 here vs operative 5), α default 10. Body hardcodes λ_inter=500, λ_contact=50; `beta`/`delta` params are dead code; contact term has NO floor. |
| `_physics_energy_per_sample` | The FIXED energy shared by the two new methods: NI normalized by hand-interior mass (extent-sensitive, inherently violation-triggered), contact hinged at the measured decoder floor (`contact_floor=0.011`, full-set value — **should be per-dataset**, YCB's true floor is 0.0269), in-hand contact voxels excluded, band-limited. |
| `sample_guided_v2` | Improved greedy: fixed energy + per-sample relative strength `rho*(1-t)^p*‖v‖` (default rho=0.2, p=1), skip 5. rho=0 or floor-satisfied samples ⇒ exact plain Euler. |
| `sample_oc_flow` | **Complete OC-Flow** (discrete adjoint): per-step controls u_i optimized over the whole trajectory vs the terminal cost, `n_outer=4` iterations, true VJPs through the model (`a_i = a_{i+1} − dt·(∂v/∂x)ᵀa_{i+1}`), per-sample normalized updates, trust region ‖u_i‖≤0.3‖v_i‖, early exit at floor. `use_model_jacobian=False` = FlowGrad straight-through. ~2·n_outer× greedy cost. Based on github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control (user's reference; `run_velocity` is its greedy approximation). |
| `FlowEulerGuidanceIntervalSampler.sample_velocity_conditioned` wrapper | FIXED: passed steps/rescale_t positionally into the base's `neg_cond`/`steps` slots (steps=25 arrived as neg_cond). Now keyword-threaded. |
| `sample_velocity_conditioned_oc2` / `_oc` | Pre-existing prototypes, NOT used by the a/b. oc2 has good pieces (trust region, per-sample mixing) but also a `target_g` amplifier that scales *small* gradients UP (anti-violation-trigger). Kept as-is. |

### Model — `trellis/models/sparse_structure_flow.py`
`SparseStructureFlowModelConditioned` gained `use_hand_pe: bool = True`. False ⇒ skips
fixes 1a/1b in forward AND skips registering `mask_hand_pos_emb`, so **pre-fix
checkpoints (the old teacher) load strict=True and run through the exact forward they
were trained with**. Default True: teacher_v2 unaffected (its configs don't set it).

### Harness & tools (`tools/`)
- `ab_eval_guidance.py` + `.sbatch` — the a/b evaluation. Arms: `unguided`
  (FlowEulerGuidanceIntervalSampler.sample, deployment params: steps 25, cfg 5.0,
  interval [0.5,1], rescale_t 3.0), `guided_asis` (α=10), `guided_v2`, `oc_flow`, plus
  the decoded **GT latent as the floor reference**. Same noise/cond per sample across
  arms. Physics metrics (pen_frac/pen_depth/contact_abs/contact_hit1v, occ_iou_gt =
  exact voxel-IoU, occ_frac) all reported floor-relative; mesh metrics (CD/NC/F@0.02/EMD,
  formulas copied from the inference repo's `eval_meshes_paired_emd_voxel.py`) in the RAW
  frame; saves ALL decoded SDFs + input images per sample; writes `.partial` JSON per
  batch (wall-kill-proof). `--no_hand_pe` for old-teacher faithful mode. `--arms` subsets.
- `mesh_metrics.py` — CD/NC/F/EMD lib + `sdf_to_mesh` (marching cubes, matches
  inference_dex.py's convention: level=0.0, ZYX transpose, half-voxel shift).
- `recompute_mesh_metrics_icp.py` + `.sbatch` — POST-HOC ICP-aligned CD/NC/F/EMD from the
  saved SDF blobs, importing `best_icp_align`/`normalize_mesh` from the inference repo's
  eval file. **Runs on the `cpu` partition** (login node OOM-kills it; `--wrap` submission
  got instantly cancelled — use the sbatch file).
- `visualize_ab_sdfs.py` — HTML gallery from saved blobs: input image + one rotatable 3D
  panel per arm (object blue, hand gray 0.35, contact voxels red). CPU/login-node safe.
  View: `cd outputs/diagnostics && python3 -m http.server 8877`, forward port in VS Code.
- `make_ycb_test_split.py` — YCB-only symlink splits from the held-out test splits:
  `datasets_split/Leap_Hand_test_ycb` (27) + `Hands_test_ycb` (32; Hands IS the YCB set).
  Hands_Google has no true YCB (numeric-looking names are product-name false positives).
- `recompute_cd_icp.py` — small CD-ladder reconciliation (raw / icp / norm+icp).

### Other fixes
- `trellis/trainers/flow_matching/mixins/image_conditioned.py`: `_load_dinov2()` loads
  the hub-cached repo with `source='local'` (no network). torch.hub otherwise re-resolves
  GitHub's HEAD on EVERY call and a transient RemoteDisconnected killed job 457 on a node
  that had just run two jobs fine. Applies to training resumes too.
- `datasets_split/Leap_Hand_train/metadata.csv`: dropped 2 chronically truncated-PNG
  instances (`MIRACLE_POUNDING_4`, `Animal_Planet_Foam_2Headed_Dragon_13`); backup
  `metadata.csv.bak_20260813`.

---

## 2. Results so far (all under `outputs/diagnostics/`)

### New teacher (EMA step 29000), full held-out mix, 64 samples — `ab_guidance_4arm.json` (job 456)
Floor-relative physics (floor = GT latent through the same decoder):

| | unguided | guided_asis | guided_v2 | oc_flow |
|---|---|---|---|---|
| pen_frac −floor | 1.31e-4 | 1.36e-4 | **4.2e-6** | **−4.5e-6** |
| contact_abs −floor | 1.110e-2 | 1.049e-2 | 1.092e-2 | **7.67e-3** |
| contact_hit1v (abs; floor 0.76) | 0.423 | 0.436 | 0.438 | **0.493** |
| occ_iou_gt | 0.497 | 0.500 | 0.499 | 0.501 |

**Verdicts:** production guidance (α=10) is inert. The fixed cost alone (guided_v2)
eliminates excess penetration (31×, to noise). Complete OC-Flow additionally cuts excess
contact 31% at flat IoU — the outer loop demonstrably optimizes (E −73% over 4 iters).

### YCB-only (same teacher), 56 samples — `ab_guidance_4arm_ycb.json` (job 458)
Penetration conclusion identical (v2/oc at/below floor). BUT the YCB decoder contact
floor is 2.4× higher (0.0269 vs 0.0113; GT hit1v only 53%) and unguided samples score
BELOW it ⇒ contact guidance has nothing legitimate to push on YCB; OC's contact gain
shrinks to +3pp. Lesson: `contact_floor` must be per-dataset if guidance ships.

### ICP-aligned mesh metrics — `ab_guidance_4arm{,_ycb}_icp.json` (job 462)
Pose absorbs ~⅓ of raw CD (YCB: 0.080→0.054 unguided). After alignment the arms nearly
tie; strong guidance costs ~1pp NC and ~3–8% EMD (YCB), F@0.02 slightly better under v2.
Unguided = most faithful geometry everywhere. **Raw-vs-benchmark note:** the historical
"old teacher CD 0.03" was on the DexYCB captures (`dex-dataset-total-total`, different
dataset entirely) through the inference pipeline — NOT comparable to any number here;
the only honest old-vs-new CD is via that pipeline (deployment step). Also verified: no
`sdf+0.01`-style level offset ever existed in the inference repo's meshing (git history:
`level=0.0` always).

### Old vs new teacher, same YCB samples/seeds — jobs 463+464, DONE 2026-08-13
`ab_guidance_4arm_ycb_oldteacher{,_icp}.json` vs `ab_guidance_4arm_ycb{,_icp}.json`.
Old teacher = `outputs/flow_conditioned_all_losses_resume_32k_resume3_LEAP/ckpts/
denoiser_ema0.9999_step0054000.pt` with `--no_hand_pe` (faithful pre-fix forward).
Pairing verified: `gt_floor` blocks identical to the last digit. n=56.

| | old unguided | **new unguided** | old oc_flow | new oc_flow |
|---|---|---|---|---|
| pen_frac −floor | 3.39e-4 | 1.26e-4 | 4.7e-5 | ≈0 |
| contact_abs −floor | 4.26e-2 | **−5.6e-3 (below)** | 3.15e-2 | −5.4e-3 |
| contact_hit1v (floor 0.532) | 0.171 | 0.426 | 0.313 | 0.455 |
| occ_iou_gt | 0.310 | 0.457 | 0.324 | 0.444 |
| CD raw / ICP | 0.140 / 0.0754 | **0.081 / 0.0542** | 0.132 / 0.0772 | 0.082 / 0.0568 |
| NC ICP | 0.812 | 0.852 | 0.793 | 0.832 |
| F@0.02 ICP | 0.434 | 0.530 | 0.437 | 0.536 |
| EMD ICP | 0.0872 | 0.0699 | 0.0957 | 0.0760 |

**Verdicts.** (i) Physics: the old teacher carries 4.26e-2 excess contact distance and
2.7× the new teacher's excess penetration; the new teacher's *unguided* samples sit
below the YCB contact floor. Achievable hit@1vox capture: old 32%, new 80% (of the
0.532 floor). (ii) Could guidance have patched it? Only partially: OC-Flow on the old
teacher kills penetration (→ floor) but closes just ~26% of the contact-distance excess
and ~56% of the hit1v gap — old+OC-Flow (0.313) still loses to new *unguided* (0.426).
(iii) Fidelity is untouchable by guidance: IoU 0.31 vs 0.46, ICP CD 0.075 vs 0.054,
EMD 0.087 vs 0.070 — the retrain bought ~28% ICP-aligned CD and ~32% IoU that no
sampling-time method recovers. **Retraining was necessary, not merely convenient.**

### Galleries
`ab_guidance_4arm_sdfs_html/` and `ab_guidance_4arm_ycb_sdfs_html/` (`index.html`;
serve dir on :8877). SDF blobs (+input images) in the matching `*_sdfs/` dirs.

---

## 3. In-flight jobs & the pending decision (state at 2026-08-13 ~13:00)

| job | what | state |
|---|---|---|
| 448 | stage-2 LAST training segment (from 32k) | running; wall ~00:25 Aug 14 at ~step 37.6k; NOTHING queued after |
| 459 | a/b (unguided+oc_flow) on FINAL ckpt, paired vs 29k run | queued afterany:448 |
| 460 | CP3a gate on **EMA** weights of final ckpt | queued afterany:448 |
| 463 | old-teacher YCB a/b | COMPLETED 14:11 Aug 13 |
| 464 | ICP mesh metrics on 463's SDFs (`tools/recompute_mesh_metrics_icp_oldteacher.sbatch`) | COMPLETED 14:21 Aug 13 → `ab_guidance_4arm_ycb_oldteacher_icp.json` |
| 465 | teacher_v2 deployment parity (train vs inference repo) | COMPLETED 21:55 Aug 13 — **PARITY PASS, bit-exact** (EMA step 37000) |
| 466/467 | dex benchmark, first attempt | FAILED 00:20 Aug 14 — ported sampler returns grad-attached latent, `.numpy()` refused; fixed in inference commit `f4fd2f3` (detach + no_grad decode) |
| 468 | dex-YCB benchmark inference, teacher_v2 FINAL ckpt (994 instances, resumable) | queued 00:25 Aug 14 → `meshes_results_marching_cubes_teacher_v2/` |
| 469 | dex-YCB paired mesh eval (historical flags) | queued afterany:468 → `summary_teacher_v2_dex_total_total.json` |

**FREEZE + next phase (2026-08-20):** the user froze the teacher at **EMA 52000**
(overriding the rule's CONTINUE; see TEACHER_RETRAIN.md §8.9 "TEACHER FROZEN").
Jobs 486/487 regenerate the dex meshes + canonical ICP eval on the frozen ckpt.
The §4 queue below is superseded by **ICRA_PLAN.md** (deployment-study framing:
distillation, capacity×steps×guidance Pareto, real-world latency). Distillation
machinery audited & fixed 2026-08-20: `trellis/trainers/flow_matching/distillation.py`
had the ENTIRE pre-fix physics block (non-estimator x0 formula, no CFG-drop gating,
w.sum() normalisation, no margin, shell-inclusive hand mask, no GT floors) — now a
faithful port of the stage-2 block; new config
`configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill_teacherv2.json`
(frozen-teacher paths, λ 30/1.0 relative+margined, fp32 student). NOT yet smoke-tested
on GPU; do a short run before a real launch. Launch itself: user approval required.

**480/481 outcome — boundary 4, CONTINUE (2026-08-20):** contact excess **2.88e-3** on
EMA 52000 (bar 3.29e-3, −21%); reconstruction guard passes (all geometry metrics up);
**OC-Flow now counterproductive on this teacher** (4.18e-3 > unguided). Segment 482 →
~57k; packet 483/484 + pre-queued 485; iterated bar for 483: < 2.60e-3. Detail:
TEACHER_RETRAIN.md §8.9 "Boundary 4 outcome".

**477/478 outcome — boundary 3, CONTINUE (2026-08-19):** contact excess **3.66e-3** on
EMA 47000 (bar 4.09e-3, −19%); OC-Flow still fully absorbed; EMA gate 0.825 READY.
Segment 479 → ~52k; packet 480/481 + pre-queued successor 482 behind it; iterated bar
for 480: < 3.29e-3. Detail: TEACHER_RETRAIN.md §8.9 "Boundary 3 outcome".

**471/472 outcome — boundary 2, CONTINUE again (read 2026-08-18):** unguided
floor-relative contact_abs on EMA 42000 = **4.54e-3** (bar 5.52e-3); OC-Flow now
ties unguided (guidance fully absorbed); EMA gate 0.747 READY. Segment 476 running
from 42k, boundary packet 477/478 queued; iterated bar for 477: < 4.09e-3. Detail:
TEACHER_RETRAIN.md §8.9 "Boundary 2 outcome".

**459 outcome — STOPPING RULE SAYS CONTINUE (2026-08-14).** Unguided floor-relative
contact_abs on the step-37000 EMA: **6.13e-3** vs 1.110e-2 at 29k (−45%, bar was −10%).
Stage-2 continues: job 470 (one segment from 37k), boundary packet 471/472 queued after;
iterated stop bar for 471: < 5.52e-3. Details + corroborating metrics in
TEACHER_RETRAIN.md §8.9 "Boundary 1 outcome". Full JSON: `ab_guidance_2arm_final.json`.

**468/469 outcome (dex benchmark, first full run):** 468 COMPLETED — 989/994 meshes in
3h57 (5 instances have no precomputed latent npz in the dataset; the historical run had
a similar 992/996 deficit). 469 ran the WRONG script variant
(`eval_meshes_paired_emd_voxel_dex.py`): the historical summary's keys (CD_cm2,
CD_sqrt_mm, Fscore5/10, ReconstructionRate) come from `..._dex_cm.py`. Its
`summary_teacher_v2_dex_total_total.json` is NOT comparable — superseded by job 474
(`tools/dex_eval_cm_teacher_v2.sbatch`; 473 was its no-ICP draft, cancelled): _cm
script on BOTH mesh dirs — teacher_v2 and the old teacher's
(`meshes_results_marching_cubes_ablation_dex`) — with **ICP always on**
(`--icp --icp_pca`; user directive 2026-08-14: never compare mesh sets without ICP,
and never against a historical no-ICP number) → `summary_{ours_ablation,teacher_v2}
_dex_total_total_icp.json`. **Resolution (473's no-ICP outputs, kept as diagnostics):** under byte-identical
no-ICP flags the two teachers TIE — old CD 0.1962/0.1631 (mean/median), teacher_v2
**0.1941/0.1593**, NC 0.56 both. The feared 4× regression was a harness mismatch:
the historical 0.0449/0.0181 (NC 0.81) is NOT reproducible with these flags, so the
historical run was almost certainly ICP-aligned (and its exact script generation
differs — its summary keys match no current script's output). Operative old-vs-new
numbers come from **job 475** (474 was cancelled mid-run when the user supplied the
exact historical invocation): both dirs through `..._dex_cm.py` with `--cd_squared
--normalize bbox_-1_1 --pt2tri --icp --icp_pca --icp_random_restarts 10 --jobs 24
--emd --voxel_iou --voxel_res 64 --voxel_fill --voxel_fixed_bounds --emd_points 512`
→ `summary_{ours_ablation,teacher_v2}_dex_total_total_icp.json`. With these the
old-teacher side should now genuinely reproduce ≈0.0449/0.0181.

**460 outcome (EMA gate, final ckpt step 37000):** hand +15.70% (shuffled) / image
+23.37% → ratio **0.672, GATE READY**; ≈ raw readings (0.718@32k) minus the ~5% EMA
init residue. JSON: `handgate_teacher_v2_stage2_physics_denoiser_ema0.9999_step0037000.json`.

**448 outcome:** graceful wall kill (SIGUSR1 at 23h55, exit 138 — SLURM says FAILED but
this is the normal chain end). Log reached step 37500; **final checkpoint = step 37000**
(i_save 1000), i.e. exactly the EMA the parity test validated and the ckpt jobs 459/460
and the dex run auto-detect. The §8.9 stopping rule applies on this checkpoint.

**Pre-registered stopping rule (TEACHER_RETRAIN.md §8.9):** if job 459's unguided
floor-relative `contact_abs` does not improve ≥10% vs the 29k value (**1.110e-2**),
stage-2 training STOPS and the final checkpoint freezes for distillation. Gate context:
stage-2 raw-weight hand/image ratio 0.250@7k → 0.729@28k → 0.718@32k (saturated).
Training-side λ: ni 30 (relative, margined), contact 1.0 (relative), both floors from
§8.1; i_save=1000.

### 3.1 How to read each pending result

- **448 (training)**: confirm clean end at the 24h wall — last line of
  `outputs/teacher_v2_stage2_physics/log.txt` ≈ step 37.5k, matching `denoiser_*` ckpts;
  a much earlier stop or a Traceback in `slurm-teacher_v2_stage2-448.out` = crash, then
  the final ckpt is whatever saved last (rule still applies, on that ckpt).
- **459 (stopping rule)**: output `outputs/diagnostics/ab_guidance_2arm_final.json`
  (if only `.partial` exists, the job died at the wall — the partial aggregates are
  usable). Compute: `results['unguided']['contact_abs']['mean'] −
  results['gt_floor']['contact_abs']['mean']`. **STOP unless < 0.999e-2** (10% better
  than 1.110e-2). Also read the `oc_flow` arm: it is the final-checkpoint guidance
  picture that feeds the deployment decision. SDFs land in `..._2arm_final_sdfs/` —
  the visualizer works on them.
- **460 (EMA gate)**: verdict in `slurm-hand_gate-<id>.out` (the job id SLURM assigned
  to 460), JSON `outputs/diagnostics/handgate_teacher_v2_stage2_physics_denoiser_ema*.json`.
  Expect hand/image ≈ the raw readings (0.71–0.73); EMA residue of the init weights is
  only ~5% by step 37k. A markedly lower ratio means the EMA still lags — note it, gate
  again after freezing, but it does not block the stopping decision (which is about the
  raw-weights training dynamics).
- **463 (old teacher)**: `outputs/diagnostics/ab_guidance_4arm_ycb_oldteacher.json` +
  `..._sdfs/`. Read against `ab_guidance_4arm_ycb.json` (new teacher, identical
  samples/seeds): (i) its unguided floor-relative pen/contact vs the new teacher's —
  the headline old-vs-new physics comparison; (ii) whether guidance arms rescue it —
  quantifies how much of the improvement genuinely required retraining vs could have
  been patched at sampling time. The ICP-aligned mesh metrics are ALREADY QUEUED as
  job 464 (afterany:463) → `ab_guidance_4arm_ycb_oldteacher_icp.json`; compare against
  `ab_guidance_4arm_ycb_icp.json` (new teacher, same samples).

---

## 4. Next steps, in order
1. Read the boundary packet (459/460) → apply the stopping rule mechanically → freeze
   (expected) or continue. Record in TEACHER_RETRAIN.md.
2. ~~Read old-teacher a/b (463); run the ICP recompute on its SDFs; write the old-vs-new
   comparison (same-harness, same-samples — the clean one).~~ DONE 2026-08-13 — see §2
   "Old vs new teacher"; retraining was necessary, guidance-on-old closes <⅓ of the gap.
3. Finalization: probe A/B on the final EMA ckpt; final numbers into TEACHER_RETRAIN.md.
4. **Deployment to the inference repo** — PORT DONE 2026-08-13, benchmark queued.
   Inference repo commits: `6e9ca9a` (WIP snapshot of the uncommitted ablation work,
   as prescribed below) then `ad9e0c9` (the port). Ported: model fixes 1a/1b/G with
   `use_hand_pe` (pre-fix denoiser jsons in `25e0d31.../ckpts/` got
   `use_hand_pe: false` so old ckpts strict-load and run their original forward),
   the train repo's full `flow_euler.py` (operative guidance math verified
   line-identical; adds fixed v2 + OC-Flow; no hardcoded save paths), pipeline
   call-site unpacks cond/neg_cond. New pipeline dir `teacher_v2_stage2/`
   (denoiser name `ckpts/denoiser_teacher_v2_final`, decoders symlinked);
   converter + parity + dex runner in `teacher_v2_port/`. **Parity job 465: PASS,
   bit-exact** (velocities max|Δ|=0 at t∈{1.0,0.55,0.1}; 25-step guided rollout
   latent and SDF max|Δ|=0; decoder fingerprints equal).

   **DEX-YCB BENCHMARK RESULT (job 475, 2026-08-14) — the honest same-pipeline
   old-vs-new.** teacher_v2 EMA step 37000 (job 468 meshes, 989/994) vs the old
   teacher's meshes, both through `eval_meshes_paired_emd_voxel_dex_cm.py` with the
   canonical flags `--cd_squared --normalize bbox_-1_1 --pt2tri --icp --icp_pca
   --icp_random_restarts 10 --jobs 24 --emd --voxel_iou --voxel_res 64 --voxel_fill
   --voxel_fixed_bounds --emd_points 512`. Harness validated: the old-teacher rerun
   reproduces the historical json (CD 0.0458/0.0181 vs 0.0449/0.0181; NC 0.806 vs
   0.806; F-curve identical to 3 decimals).

   | ICP-aligned, bbox[-1,1]³ | old teacher (n=992) | **teacher_v2 37k (n=989)** |
   |---|---|---|
   | CD² mean / median | 0.0458 / 0.0181 | **0.0349 / 0.0119** (−24% / −34%) |
   | NC mean / median | 0.806 / 0.835 | **0.828 / 0.866** |
   | F@0.02 mean | 0.130 | **0.160** (+23%) |
   | F@0.04 / 0.06 / 0.08 / 0.10 mean | 0.339 / 0.497 / 0.613 / 0.702 | **0.394 / 0.553 / 0.670 / 0.755** |

   Diagnostic aside (jobs 469/473, no ICP): both teachers tie at raw CD ≈0.195 — the
   raw-frame number is dominated by canonical-pose error and says nothing about
   shape; the historical 0.0449 was ICP'd all along. Outputs:
   `summary_{ours_ablation,teacher_v2}_dex_total_total_icp.{json,csv}`. The 5
   unevaluated instances lack a precomputed latent npz in the dataset (dataset
   gap, not a model failure). Later checkpoints (42k, and whatever freezes) should
   rerun 468→475 for the final table.
   Original plan (kept for reference): port
   `sparse_structure_flow.py` fixes (that repo predates fixes 1a/1b/G — teacher_v2 ckpts
   will NOT load/run correctly there as-is) + the fixed/OC samplers; convert final EMA
   ckpt via `from_pt_to_safetensors.py`; new pipeline dir (pattern:
   `25e0d31.../pipeline_conditioned.json`, from_pretrained hardcodes that json name);
   **strict parity test** (same latent+cond → same SDF in both repos) before trusting;
   then `inference_dex.py` + `eval_meshes_paired_emd_voxel.py` on dex-dataset for the
   true old-vs-new benchmark. NOTE: inference repo working tree has ~70 uncommitted lines
   (flow_euler.py, sparse_structure_flow.py) — commit/stash before porting.
5. Distillation — goal, existing trainer/config, stale fields to fix, and the
   guided-distillation design options are in **§5 below**.
6. If guidance ships: per-dataset `contact_floor`; consider retiring `guided_asis`.

## 5. Distillation: goal and machinery (verified against the code, 2026-08-13)

**Goal:** compress the 24-block teacher into an **8-block student** (same 1024 channels,
same `SparseStructureFlowModelConditioned` class and conditioning) — ~3× fewer
transformer blocks for correspondingly faster sampling, while inheriting the hand
conditioning and physics behavior. This is the reason teacher_v2 exists at all.

**Machinery that already exists:**
- Trainer `ImageConditionedFlowMatchingCFGDistillationTrainerConditioned`
  (`trellis/trainers/flow_matching/distillation.py`): student is the config's
  `denoiser`; the teacher is loaded frozen from `teacher_config_path` +
  `teacher_ckpt_path` and used as an **additional velocity target** (online
  distillation on training data: `distill_loss_weight` (1.0) on ‖v_student − v_teacher‖
  plus `target_loss_weight` (0.25) on the ground-truth flow target, plus
  `use_physics_losses: true`).
- Config `configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill.json`.
  **Stale fields that MUST be updated before launching:** `teacher_config_path` points
  at the OLD teacher's config, `teacher_ckpt_path` is empty (→ set to
  `outputs/teacher_v2_stage2_physics/config.json` + the frozen final EMA ckpt), and the
  physics λ are the OLD-normalization values (200 / 0.1) — re-derive per the §8.1/§8.8
  scale analysis (stage-2 used 30 / 1.0 with `ni_relative`/`contact_relative: true`;
  student λ should be re-probed or start from those).
- Prior student runs exist under `outputs/flow_conditioned_distilled*` (8-block
  variants, distilled from the OLD teacher — superseded). Known incident: the fp16
  distill run NaN'd at step 48407 (`flow_conditioned_distilled_all/nan_debug_autograd.log`);
  teacher_v2 practice is fp32 + TF32.

**Open design choice (from the a/b results):** the trainer distills *velocities on
noised data*, not sampled outputs — so "distill from OC-Flow-guided outputs" is not what
it does today. Two ways to bake guided physics into the student: (a) generate a synthetic
dataset by OC-Flow-guided sampling from the teacher and add it to training; or (b) add
the guidance control term to the teacher's velocity target during distillation
(`v_teacher + u` where u = the fixed-energy gradient step — cheap, computable online with
one decoder pass, elegant fit to the existing trainer). Neither is implemented; decide
after the teacher freezes.

## 6. Cluster gotchas learned this campaign
- Login node OOM-kills anything sizable (ICP recompute included) — use `cpu` partition
  via a proper sbatch FILE (`--wrap` + `--export=NONE` got insta-cancelled, no reason).
- a/b runs with oc_flow: ~5h for 64 samples on 1 GPU (12h limit is safe); `.partial`
  JSON protects against wall kills. TF32 is enabled inside the harness.
- Job 445 hung once right after the step-32k checkpoint+snapshot (zero output 1.5h,
  clean save on disk) — cancelled, successor resumed losslessly. If it recurs at a
  save boundary, suspect the snapshot sampling path.
- Background watchers die with the Claude session; SLURM `--dependency` chains and
  queued jobs survive anything. Prefer dependencies for must-run steps.

## 7. ICRA deployment study — frozen-teacher results (2026-08-20/21)

### 7.1 Teacher rows of the capacity × steps × guidance Pareto table (ICRA_PLAN item 4)

Frozen teacher (EMA 52000), held-out mix, n=64, seed 1337 — all runs paired with
job 480 (gt_floor blocks identical; contact floor 0.011964). `ab_eval_guidance.py`
already had `--steps`; no harness change was needed. **Pitfall handled:**
`guidance_skip` is a step *count* (guidance only on `step_i >= skip`), so the
default 5 would have disabled guided_v2 entirely at 4 steps — the sweep scales it
to keep the skipped trajectory fraction ≈20%: skip 5@25, 2@8, 1@4 (oc_flow has no
skip; it optimizes the whole trajectory). Sources: `ab_guidance_2arm_final4.json`
(job 480, 25-step unguided/oc_flow), `ab_guidance_teacher52k_steps25_v2.json`
(491), `..._steps8.json` (489), `..._steps4.json` (490).

| steps | arm | contact_abs −floor | hit1v | occ_iou | CD | NC | F@0.02 | EMD |
|---|---|---|---|---|---|---|---|---|
| 25 | **unguided** | **2.88e-3** | 0.601 | 0.593 | 0.0593 | 0.836 | 0.546 | 0.0782 |
| 25 | guided_v2 | 8.31e-3 | 0.515 | 0.577 | 0.0613 | 0.830 | 0.528 | 0.0822 |
| 25 | oc_flow | 4.18e-3 | 0.574 | 0.578 | 0.0608 | 0.831 | 0.525 | 0.0824 |
| 8 | **unguided** | **2.63e-3** | 0.621 | 0.593 | 0.0593 | 0.837 | 0.546 | 0.0794 |
| 8 | guided_v2 | 4.41e-2 | 0.126 | 0.507 | 0.0746 | 0.805 | 0.426 | 0.103 |
| 8 | oc_flow | 5.21e-2 | 0.162 | 0.523 | 0.0719 | 0.817 | 0.451 | 0.102 |
| 4 | **unguided** | **2.76e-3** | 0.612 | 0.573 | 0.0604 | 0.827 | — | 0.0809 |
| 4 | guided_v2 | 4.93e-2 | 0.139 | 0.503 | 0.0739 | 0.809 | — | 0.101 |
| 4 | oc_flow | 1.43e-1 | 0.049 | 0.401 | 0.106 | 0.770 | — | 0.150 |

(pen_frac excess is ≈0 (±5e-5) for every arm at every step count — penetration is
fully trained-in; contact is the discriminating axis.)

**Verdicts.**
1. **The teacher is essentially step-count-invariant unguided down to 4 steps**:
   contact excess, hit1v, IoU, CD all within noise of the 25-step values (IoU dips
   0.593→0.573 and CD +2% only at 4 steps). 6× fewer steps ≈ free — a major
   deployment result on its own.
2. **Guidance never re-enters on the steps axis — it gets *worse* as steps
   shrink**: harmful at 25 steps (v2 8.3e-3, oc 4.2e-3 vs unguided 2.9e-3),
   catastrophic at 8 (≈17×) and 4 (v2 18×; oc 50×, hit1v 0.60→0.05, IoU −0.19).
   Mechanistically consistent with absorption + integration: each per-step
   correction rides a larger dt and a noisier x̂0, and there is no legitimate
   contact violation left to fix, so the injected control is pure error. The
   guidance-re-entry hypothesis now rests entirely on the **capacity axis (the
   distilled student)**.

### 7.2 Frozen-teacher dex benchmark — the paper's old-vs-new table (jobs 486/487, 2026-08-21)

Job 486 converted the FROZEN EMA 52000 into the inference repo's `teacher_v2_stage2/`
pipeline and regenerated the dex meshes (`meshes_results_marching_cubes_teacher_v2_52k/`,
989/994 — same 5-instance dataset gap as every run); job 487 ran the canonical ICP
eval (`--cd_squared --normalize bbox_-1_1 --pt2tri --icp --icp_pca
--icp_random_restarts 10 --jobs 24 --emd --voxel_iou --voxel_res 64 --voxel_fill
--voxel_fixed_bounds --emd_points 512`). Both COMPLETED 02:06 Aug 21. Output:
`summary_teacher_v2_52k_dex_total_total_icp.{json,csv}` (inference repo root),
compared against job 475's old-teacher and v2@37k summaries (same harness/flags).

| ICP-aligned, bbox[-1,1]³ | old teacher (n=992) | v2 @37k (n=989) | **v2 @52k FROZEN (n=989)** |
|---|---|---|---|
| CD² mean / median | 0.0458 / 0.0181 | 0.0349 / 0.0119 | **0.0335 / 0.0111** (−27% / −39% vs old) |
| NC mean / median | 0.806 / 0.835 | 0.828 / 0.866 | **0.830 / 0.866** |
| F@0.02 mean | 0.130 | 0.160 | **0.168** (+30% vs old) |
| F@0.04 / 0.06 / 0.08 / 0.10 mean | 0.339 / 0.497 / 0.613 / 0.702 | 0.394 / 0.553 / 0.670 / 0.755 | **0.418 / 0.580 / 0.694 / 0.775** |

The 37k→52k training bought a further −4% CD² mean / −7% median and +5% F@0.02 on
real captures — modest but uniformly positive, consistent with the held-out a/b
trend (§8.9 boundary 4). This is the final old-vs-new row set for the paper.

**Steps axis on REAL captures (jobs 494–497, 2026-08-21) — confirms §7.1.** Same
frozen pipeline, same seed 42, `DEX_STEPS` env override in
`teacher_v2_port/inference_dex_teacher_v2.py` (unset ⇒ identical to the parity-tested
run); sbatch pair `tools/dex_teacher_v2_52k_steps.sbatch <N>` (GPU, deliberately
skips the converter — the generic one would re-convert the 54k leftover) →
`tools/dex_eval_cm_teacher_v2_52k_steps.sbatch <N>` (CPU, canonical ICP flags).
Outputs `summary_teacher_v2_52k_s{8,4}_dex_total_total_icp.json`. n=989, 0 failed.

| ICP-aligned | old teacher (25) | **v2@52k 25 steps** | **v2@52k 8 steps** | v2@52k 4 steps |
|---|---|---|---|---|
| CD² mean / median | 0.0458 / 0.0181 | 0.0335 / 0.0111 | **0.0336 / 0.0112** | 0.0357 / 0.0118 |
| NC mean / median | 0.806 / 0.835 | 0.830 / 0.866 | **0.831 / 0.866** | 0.822 / 0.857 |
| F@0.02 / 0.04 / 0.06 | 0.130 / 0.339 / 0.497 | 0.168 / 0.418 / 0.580 | **0.170 / 0.418 / 0.580** | 0.163 / 0.405 / 0.566 |
| F@0.08 / 0.10 | 0.613 / 0.702 | 0.694 / 0.775 | 0.693 / 0.773 | 0.678 / 0.760 |
| inference wall (989 inst.) | — | 3h55 | 1h23 | 0h48 |

**8 steps is identical to 25 on real captures to the third decimal** (3× fewer NFEs,
2.8× wall). 4 steps costs +6.6% CD² mean / +6% median and −1pp NC/F — a small haircut
that still beats the old teacher at 25 steps by −22% CD². Deployment default for the
paper's latency demo: **8 steps unguided** (4 if the latency budget demands it).

### 7.3 Distillation smoke test (job 488, 2026-08-20) — PASS

The FIXED trainer (`distillation.py`) ran 400 steps on 1 GPU (config
`..._distill_teacherv2_smoke.json` = real config with max_steps 400, i_log 50,
i_sample 200, i_save 300, exercising snapshot+save paths): teacher strict-loads
from `denoiser_teacher_v2_FROZEN.pt` (0 missing/unexpected keys);
`ni_floor`/`contact_floor` in every log line, contact_floor ≈0.0111 = the known
decoder floor; **no NaN**; distill_mse 1.05→0.22 and target_mse 1.19→0.36
(monotone; a from-scratch 8-block student starts ≈1.0 by construction and is
already entering teacher territory); ni_loss ≈0 under the 10k warmup,
contact_raw 0.023→0.005. ~4.9 s/step @ batch 8 on 1 GPU. Launch-ready sbatch:
`tools/train_distill_teacherv2.sbatch` (3×24h self-chaining segments, 2 GPUs,
fresh `outputs/distill_teacherv2`) — **NOT launched; user approval required.**

### 7.4 Full distillation — LAUNCHED 2026-08-21 (user approval), `outputs/distill_teacherv2/`

Chain `tools/train_distill_teacherv2.sbatch`: segment 1 = job 492 (09:56 Aug 21 →
graceful SIGUSR1 wall end at 23h55, exit 138 — sacct shows FAILED, that is the normal
chain end), segment 2 = 493 (started ~09:52 Aug 22), segment 3 = 498 pre-queued.
Segment 1 reached **step 16000** (728 steps/h, 2 GPUs, batch 16; ckpts every 1000,
EMA 0.9999). **No NaN in 16.5k logged steps.** Loss trajectory (bin means):

| steps | distill_mse | target_mse | ni_loss (λ=30 after 10k) | contact_loss (λ=1) |
|---|---|---|---|---|
| 1–500 | 0.342 | 0.483 | 0 | 7.2e-3 |
| 2k–4k | 0.115 | 0.257 | 5e-5 | 7.0e-4 |
| 8k–10k | 0.088 | 0.228 | 7e-5 | 3.0e-4 |
| 14k–16k | 0.076 | 0.217 | 4e-5 | 2.0e-4 |
| 16k–16.5k | **0.0735** | **0.214** | 4e-5 | 1.9e-4 |

target_mse 0.21 is already inside the teacher's own range (0.16–0.19 at 52k, batch
16 — see `outputs/teacher_v2_stage2_physics/log.txt`); distill_mse still falling ~3%
per 2k steps; physics terms at floor-relative noise and shrinking (the student inherits
the teacher's physics through the velocity target, as designed). First student
boundary eval (ICRA_PLAN item 5: a/b harness, unguided+guided_v2+oc_flow at 25/8/4
steps, `--teacher_dir outputs/distill_teacherv2 --ckpt denoiser_ema0.9999_step00XX000.pt`)
is worth running on the 16k EMA now and on the segment-2 end (~32k).

### 7.5 First student boundary eval — 16k EMA (jobs 499–501, 2026-08-22): GUIDANCE RE-ENTERS

`outputs/distill_teacherv2/ckpts/denoiser_ema0.9999_step0016000.pt` through the a/b
harness at 25/8/4 steps (skips 5/2/1), arms unguided/guided_v2/oc_flow, n=64.
Outputs `ab_guidance_student16k_steps{25,8,4}.json`.

**Pairing caveat (found here, fixed for later runs):** the harness DataLoader shuffles
from torch's *global* RNG, whose state at iterator time depends on how many draws model
construction consumed — the 8-block student therefore evaluated a DIFFERENT 64-sample
subset than the 24-block teacher (gt_floor contact 0.01592 vs 0.01196). Arm-vs-arm
comparisons *within* a model are exactly paired and valid; teacher-vs-student absolute
numbers below are NOT. New flag `--data_seed` re-seeds right before the iterator; every
cross-model run from now on passes `--data_seed 1337` (legacy runs: default None).

| steps | arm | contact −floor | hit1v | occ_iou | CD | F@0.02 | pen −floor |
|---|---|---|---|---|---|---|---|
| 25 | unguided | 4.50e-2 | 0.149 | 0.268 | 0.207 | 0.121 | 1.4e-3 |
| 25 | **guided_v2** | **9.51e-3 (−79%)** | **0.395** | **0.290** | **0.185** | **0.164** | 2.1e-4 |
| 25 | oc_flow | 1.66e-2 (−63%) | 0.325 | 0.297 | 0.185 | 0.165 | 1.3e-4 |
| 8 | unguided | 4.64e-2 | 0.145 | 0.266 | 0.209 | 0.121 | 1.4e-3 |
| 8 | guided_v2 | 2.48e-2 (−47%) | 0.237 | 0.277 | 0.193 | 0.146 | 3.2e-4 |
| 8 | oc_flow | 5.25e-2 (+13%) | 0.151 | 0.277 | 0.192 | 0.145 | 1.4e-5 |
| 4 | unguided | 4.92e-2 | 0.137 | 0.261 | 0.213 | 0.119 | 1.5e-3 |
| 4 | guided_v2 | 3.13e-2 (−36%) | 0.210 | 0.274 | 0.202 | 0.138 | 5.6e-4 |
| 4 | oc_flow | 9.69e-2 (+97%) | 0.057 | 0.248 | 0.212 | 0.126 | 3.6e-5 |

**Verdicts.** (i) The capacity axis does what the steps axis did not: on this weak
student guidance HELPS — guided_v2 at 25 steps cuts excess contact 79%, nearly triples
hit1v, removes 85% of excess penetration, AND improves geometry (IoU +0.02, CD −10%,
F@0.02 +36%). This is the re-entry the absorption curve predicted. (ii) Greedy v2 beats
complete OC-Flow on the student at every step count; OC-Flow turns harmful at ≤8
steps (same mechanism as on the teacher: large-dt controls on a noisy x̂0). (iii) The
student at 16k is still far from the teacher (IoU ~0.27 vs ~0.59, CD 0.21 vs 0.06,
unpaired) — BUT the 16k EMA carries a 0.9999^16000 = **20% residue of the random init**
(EMA time constant 10k steps), so this checkpoint understates the student; target_mse
(0.21) is already near the teacher's. Disentangling runs queued below.

**Queued 2026-08-23 02:40 (all `--data_seed 1337`, paired across models):**
- 502/503/504: teacher rows re-measured paired (25/8/4, 3 arms) →
  `ab_guidance_teacher52k_ds_steps{25,8,4}.json`
- 505: student 16k **raw** weights (`denoiser_step0016000.pt`, 25 steps,
  unguided+guided_v2) → `ab_guidance_student16kraw_ds_steps25.json` — EMA-lag check
- 506/507/508 (afterany:493): student post-segment-2 `latest_ema` (~32k, residue 4%),
  25/8/4, 3 arms → `ab_guidance_student_seg2_ds_steps{25,8,4}.json`
Segment 2 progress: step 27.5k, no NaN, distill_mse 0.0646 / target_mse 0.205 (26–28k),
still falling; segment 3 = job 498 pre-queued.

### 7.6 CORRECTION to §7.5 — the 16k "re-entry" was an EMA artifact (jobs 502–505, 2026-08-23)

All four runs `--data_seed 1337` ⇒ gt_floor contact 0.012123 identical across teacher
and student: **pairing now verified.** (This subset is easier than the legacy one —
teacher IoU 0.675 vs 0.593 — so §7.1/§7.5 numbers are not comparable to these; the
`_ds_` files are the paired set from here on.)

| model | steps | arm | contact −floor | hit1v | occ_iou | CD | NC | F@0.02 | EMD |
|---|---|---|---|---|---|---|---|---|---|
| teacher 52k | 25 | **unguided** | **3.81e-3** | 0.613 | 0.675 | 0.0484 | 0.872 | 0.630 | 0.0707 |
| teacher 52k | 25 | guided_v2 | 8.51e-3 | 0.495 | 0.659 | 0.0505 | 0.866 | 0.603 | 0.0740 |
| teacher 52k | 25 | oc_flow | 5.65e-3 | 0.553 | 0.666 | 0.0494 | 0.868 | 0.612 | 0.0724 |
| teacher 52k | 8 | **unguided** | **3.31e-3** | 0.639 | 0.679 | 0.0480 | 0.873 | 0.639 | 0.0708 |
| teacher 52k | 8 | guided_v2 | 4.65e-2 | 0.121 | 0.601 | 0.0614 | 0.846 | 0.521 | 0.0910 |
| teacher 52k | 8 | oc_flow | 5.10e-2 | 0.134 | 0.608 | 0.0601 | 0.849 | 0.529 | 0.0861 |
| teacher 52k | 4 | **unguided** | **3.10e-3** | 0.629 | 0.649 | 0.0514 | 0.861 | 0.590 | 0.0744 |
| teacher 52k | 4 | guided_v2 | 5.18e-2 | 0.108 | 0.591 | 0.0627 | 0.845 | 0.495 | 0.0893 |
| teacher 52k | 4 | oc_flow | 1.35e-1 | 0.036 | 0.513 | 0.0879 | 0.817 | 0.407 | 0.1211 |
| student 16k **raw** | 25 | unguided | 7.21e-3 | 0.491 | 0.436 | 0.1134 | 0.759 | 0.322 | 0.1372 |
| student 16k **raw** | 25 | guided_v2 | 8.07e-3 | 0.464 | 0.434 | 0.1150 | 0.758 | 0.325 | 0.1431 |

**Verdicts (supersede §7.5's interpretation).**
1. **The 16k EMA was init-contaminated, as feared**: the raw 16k weights score
   contact excess 7.2e-3 / hit1v 0.49 / IoU 0.44 / CD 0.113 — roughly 6× / 3× / 1.6× /
   2× better than the 16k EMA (different subset, but the gap dwarfs any subset
   effect). The 20% init residue (0.9999^16000) was the dominant defect.
2. **On the properly-weighted 16k student, guidance does NOT help at 25 steps**:
   guided_v2 7.2e-3 → 8.1e-3 (slightly worse), IoU/CD/F flat, only penetration excess
   moves (5e-5 → −1.5e-5, i.e. to floor — penetration is the one term guidance still
   fixes on every model). So §7.5's "re-entry on the capacity axis" is NOT supported
   once the EMA artifact is removed; what §7.5 measured is re-entry on the
   *training-progress* axis (guidance helps an undertrained/contaminated model — the
   same regime as teacher ckpt 29k), which is consistent with the absorption thesis
   but is not a capacity result. Whether the 8-block student is "absorbed" at 8/4
   steps is answered by 506–508 (32k EMA, 4% residue, 25/8/4, paired).
3. Teacher step-invariance and guidance harm reproduce on the paired subset
   (unguided 3.8/3.3/3.1e-3 at 25/8/4; guidance harmful everywhere, OC-Flow worst
   at low steps).
4. Paired gap at 16k: student IoU 0.436 vs teacher 0.675, CD 0.113 vs 0.048 — large;
   distillation continues (segment 2 ended at step 32000, segment 3 = job 498).

### 7.7 Student 32k EMA, paired (jobs 506–508, 2026-08-23) + chain complete

Distillation chain FINISHED 2026-08-24: segments 492/493/498 (3×24h, ~730 steps/h),
final checkpoint **step 48000** (raw + EMA; EMA init residue 0.9999^48000 = 0.8%).
No NaN in 48.5k logged steps; distill_mse 0.0539 / target_mse 0.194 at the end, still
falling slowly (−4% per 6k steps). No successor queued (MAX_SEGMENTS reached).

32k EMA (residue 4%), paired vs §7.6's teacher rows (same `--data_seed 1337` subset):

| model | steps | arm | contact −floor | hit1v | occ_iou | CD | F@0.02 |
|---|---|---|---|---|---|---|---|
| student 32k | 25 | unguided | 7.13e-3 | 0.476 | 0.491 | 0.0895 | 0.348 |
| student 32k | 25 | guided_v2 | 7.95e-3 | 0.466 | 0.486 | 0.0900 | 0.352 |
| student 32k | 25 | **oc_flow** | **6.33e-3 (−11%)** | **0.508** | 0.494 | 0.0887 | 0.362 |
| student 32k | 8 | unguided | **6.60e-3** | 0.494 | 0.487 | 0.0927 | 0.342 |
| student 32k | 8 | guided_v2 | 3.26e-2 | 0.165 | 0.461 | 0.0993 | 0.305 |
| student 32k | 8 | oc_flow | 5.84e-2 | 0.123 | 0.447 | 0.1037 | 0.295 |
| student 32k | 4 | unguided | 8.70e-3 | 0.476 | 0.478 | 0.0967 | 0.329 |
| student 32k | 4 | guided_v2 | 3.57e-2 | 0.183 | 0.465 | 0.1006 | 0.304 |
| student 32k | 4 | oc_flow | 1.09e-1 | 0.070 | 0.408 | 0.1181 | 0.266 |

**Verdicts.**
1. **Marginal, real re-entry at 25 steps — via OC-Flow only**: −11% contact excess,
   +3pp hit1v, geometry flat-to-slightly-better. Greedy v2 does not help. On the
   teacher OC-Flow is harmful at 25 steps (+48%), so the sign flips with capacity —
   the re-entry exists but is far smaller than the absorption curve's cross-model
   trend suggested (the student's physics losses absorbed most of it during
   distillation, as designed).
2. **At 8/4 steps guidance stays harmful on the student too** (v2 5×, oc 9× contact
   excess at 8) — the large-dt mechanism dominates at low NFE regardless of capacity.
   Deployment: unguided at 8 steps for the student as well (student is also
   step-invariant: CD 0.0895/0.0927/0.0967 at 25/8/4).
3. 16k-raw → 32k-EMA progress (paired): IoU 0.436→0.491, CD 0.113→0.0895, F@0.02
   0.322→0.348; contact excess flat at ~7e-3 (≈1.9× the teacher's 3.8e-3).
4. Remaining paired gap to the teacher at 8 steps: IoU 0.487 vs 0.679, CD 0.093 vs
   0.048, contact excess 2×. Jobs 509–511 queued: same eval on the FINAL 48k EMA.

### 7.8 Student 48k (final chain ckpt), latency table, copy-init ablation start (jobs 509-511/515/514, 2026-08-24)

**48k EMA paired evals** (`ab_guidance_student48k_ds_steps{25,8,4}.json`): the 32k->48k
slope is steep — undertraining confirmed as the dominant gap driver:
| student, unguided | 32k | 48k | teacher |
|---|---|---|---|
| contact excess @8 | 6.60e-3 | **3.41e-3** | 3.31e-3 |
| hit1v @8 | 0.494 | 0.597 | 0.639 |
| IoU @8 | 0.487 | 0.553 | 0.679 |
| CD @8 | 0.0927 | 0.0774 | 0.0480 |
| F@0.02 @8 | 0.342 | 0.436 | 0.639 |

**The student's PHYSICS gap is closed at 48k** (contact excess == teacher within noise,
at every step count); the GEOMETRY gap (IoU/CD/F) remains but is shrinking fast —
extension to ~82k (jobs 512/513) was the right call. **Absorption reproduces on the
student**: the 32k OC-Flow re-entry (−11%) is GONE at 48k (4.04e-3 vs unguided
3.88e-3 at 25 steps) — physics guidance is a substitute for training progress,
transiently useful and then absorbed, on BOTH models. Low-NFE guidance harm unchanged.

**Latency, H200 batch 1 (`latency_h200.json`)**: the denoiser is ~100% of the budget
(cond encode 26 ms, decode 10 ms, marching cubes 3 ms). CFG doubles forwards.
| flow sampling | 25 steps | 8 steps | 4 steps |
|---|---|---|---|
| teacher (757M) | 9550 ms | 3253 ms | 1734 ms |
| student 8x2 (220M) | 3141 ms | **1072 ms** | 572 ms |

=> Real-time (~1 Hz or better) REQUIRES the student: teacher@8 is 3.3 s; student@8 is
1.07 s and student@4 is 0.57 s. This hardens the distillation motivation. (All fp32+
TF32; bf16 would cut further — untested.)

**Copy-init ablation (job 514, running)**: init built and strict-loaded (300 tensors,
0 unexpected); first-step distill_mse 0.60 vs 1.05 from scratch — inherited features
confirmed doing work. Output: `outputs/distill_s8mlp4_copyinit/`.

### 7.9 Multi-view phase — P0 warp validation PASSED (2026-08-24)

Dataset decision (user): dex lacks usable multi-view groups; the synthetic sets are
24 views per grasp (`data_pose_norm/<inst>/<inst>_fNNN_meta.json`, per-view GT SDFs
in `sdfs/`). Held-out `Leap_Hand_test` (319) + `Hands_test` (32) give UNCONTAMINATED
multi-view groups — same held-out status as all §7 evals.

`tools/multiview_warp.py`: per-view grids are similarity transforms of a canonical
grasp frame; the metas' convention (found by composition sweep, NOT the naive
reading): **x_view = s_aug * R_fixed^T @ x_canon + t_aug**, voxel-CENTER sampling,
SDF values rescale by s_dst/s_src. GT->GT warp across view pairs: object IoU
0.94-0.97, hand 0.91-0.95, band |sdf diff| ~0.003 = 1/10 voxel (discretization
noise). Foundation for P1 fusion + P2 consistency guidance is solid.

### 7.10 bf16 latency + P1 fusion smoke (jobs 517/518, 2026-08-24)

**bf16 (autocast, weights fp32) halves everything** (`latency_h200_bf16.json`):
teacher 25/8/4 = 4.65/1.59/0.85 s (was 9.55/3.25/1.73); student = 1.52/0.52/0.28 s.
Student@8 ≈ 2 Hz, student@4 ≈ 3.6 Hz. Ranking unchanged; bf16 QUALITY impact not
yet verified — needs an a/b before the paper quotes bf16 numbers.

**P1 fusion smoke (student 48k @8 steps, n=2 groups — indicative only):** fusion is
dramatic. Floor-relative contact excess: single 4.4e-3 → K4-mean 3.6e-4 → K8-mean
BELOW floor. Geometry: IoU 0.703 → 0.858 (K4 median), CD 0.045 → 0.024, NC 0.856 →
0.945, F@0.02 0.627 → 0.863 — **the fused student beats the single-view TEACHER**
(IoU 0.679, CD 0.048 paired) at a fraction of the compute. Surprise: plain
mean/median ≫ vismean — the visibility weighting HURTS (warped vis masks noisy,
few voters per voxel); drop or rework it. Full runs queued: 519-522 =
{student48k, teacher52k} × {8, 25} steps, 48 held-out groups each →
`mv_fusion_{student48k,teacher52k}_s{8,25}.json`. Harness:
`tools/multiview_fusion_eval.py` (+ `force_view` hook in datasets/components.py).

### 7.11 P1 fusion at scale (jobs 519–522, n=48, 2026-08-24) + P2 smoke (523)

`mv_fusion_{student48k,teacher52k}_s{8,25}.json`, 48 held-out groups, K-view
median fusion (the winning arm; vismean confirmed harmful, retired):

| @8 steps unguided | single | median_K4 | median_K8 |
|---|---|---|---|
| student48k: c_ex / IoU / CD / F.02 | 4.1e-3 / 0.477 / 0.0756 / 0.445 | 1.0e-3 / 0.548 / 0.0596 / 0.524 | **2.4e-4 / 0.574 / 0.0556 / 0.562** |
| teacher52k: same | 3.9e-3 / 0.607 / 0.0501 / 0.606 | 5.2e-4 / 0.670 / 0.0406 / 0.693 | **7.5e-5 / 0.713 / 0.0358 / 0.742** |

(25-step rows within noise of 8-step for both models — step-invariance again.)
**Verdicts.** (i) Fusion is monotone in K for both models: K8 median removes
~95% of excess contact and buys +0.10 IoU / −26–29% CD / +22–26% F@0.02. (ii)
Smoke's "fused student beats single teacher" does NOT hold at n=48 (0.574 vs
0.607 IoU) — close at ~3× lower latency; the 82k / copy-init student may close
it. (iii) Teacher K8-median (IoU 0.716, CD 0.0355) is the best quality measured
in this campaign — the offline/refinement row.

**P2 smoke (student@25, n=2, `mv_p2_smoke.json`): consistency guidance improves
the SINGLE view** — contact excess 4.8e-3 → 2.0e-3 (−58%), IoU 0.695 → 0.727,
NC 0.856 → 0.892: each view absorbs the others' information *during sampling*.
Guided-then-fused ≈ plain-fused in this tiny sample. Full runs queued: 524
(student) / 525 (teacher) → `mv_p2_{student48k,teacher52k}_s25.json`; the teacher
run is the measurement-vs-prior absorption test.

### 7.12 L40S partition validated for evals (job 526, 2026-08-24)

`gpu-l40s` (4 idle nodes, QOS: 2 concurrent jobs, SEPARATE from the 4-job/96-CPU
h200 quota) runs the full eval stack correctly: the 2-group fusion smoke
reproduced H200 numbers (IoU 0.7032 vs 0.7029 single; 0.8082 vs 0.8024 K8-median
— normal cross-arch drift) in ~1 min. Usage: submit any existing eval sbatch with
`sbatch --partition=gpu-l40s ...` (CLI overrides the #SBATCH partition; same
conda env works — sm_86 kernels run on Ada). Default routing from now on: EVALS
→ gpu-l40s, TRAINING → gpu-h200. NOT validated for training runs.

### 7.13 P2 full results + bf16 parity + full-set teacher fusion (jobs 524/525/527/529, 2026-08-24)

**The measurement-vs-prior contrast is MEASURED (the paper's scientific core).**
On the SAME fully-trained teacher where physics guidance is harmful (§7.6:
guided_v2 IoU 0.675→0.659, contact excess 2.2×), cross-view consistency guidance
HELPS: single-view IoU 0.607→0.625, CD 0.0502→0.0474 (−6%), F@0.02 0.603→0.617
(`mv_p2_teacher52k_s25.json`, n=48, paired noise). On the student the same effect
is 3× larger (IoU 0.482→0.543, CD −10%, F@0.02 +15% — `mv_p2_student48k_s25.json`).
**Prior-based guidance gets absorbed by training; measurement-based guidance does
not — and the weaker the model, the more the measurements are worth.**

**Guidance and fusion do NOT stack**: median-fusing the consistency-guided views
is worse than fusing unguided ones (teacher IoU 0.677 vs 0.716; student 0.570 vs
0.578) — mid-sampling consensus correlates the views' errors, destroying the
independence that fusion exploits. Deployment rule: guide when the output is one
view; fuse independent views when K are available; never both.

**bf16 = fp32 quality** (`ab_guidance_teacher52k_ds_steps8_bf16.json`, paired
n=64): every metric within noise (IoU 0.682 vs 0.679, CD 0.0474 vs 0.0480).
The §7.10 bf16 latency table is quotable: student@8 = 0.52 s (~2 Hz) is the
deployment mode at zero quality cost.

**Full-test-set teacher fusion @8 steps (n=351, `mv_fusion_teacher52k_s8_full.json`)
— paper-grade, frozen-teacher-final:**
| arm | contact −floor | IoU | CD | NC | F@0.02 | EMD |
|---|---|---|---|---|---|---|
| single | 2.16e-3 | 0.605 | 0.0586 | 0.849 | 0.568 | 0.0783 |
| median_K2 | −3.6e-4 | 0.634 | 0.0580 | 0.864 | 0.599 | 0.0825 |
| median_K4 | −1.7e-3 | 0.685 | 0.0474 | 0.882 | 0.669 | 0.0738 |
| median_K8 | **−2.5e-3** | **0.716** | **0.0427** | **0.894** | **0.716** | 0.0710 |
Contact excess is BELOW the decoder floor from K=2 on. n=48 subset numbers (§7.11)
confirmed within ~0.01 IoU. 25-step full-set run = job 528 (L40S, in flight).

### 7.14 Exact hand-visibility fusion weighting (job 530, 2026-08-24) — median still wins

User insight tested: hand occlusion is EXACTLY computable (the hand SDF is GT
conditioning), so `vishand` = soft directional weight exp(−α·hand voxels between
voxel and camera), floor 0.25, only counting hand on the CAMERA side (hand behind
the object correctly contributes zero). Student@8, n=48, paired
(`mv_vishand_student48k_s8.json`):
mean_K8 IoU 0.543 / CD 0.0762 < **vishand_K8 0.552 / 0.0712** < **median_K8
0.573 / 0.0559**. Verdict: exact occlusion weighting beats the plain mean but
NOT the robust vote — occluded views' generative completions carry real signal,
and the majority vote uses it while rejecting outliers. **Median stays the
paper's fusion arm**; ablation row mean < vishand < median is itself presentable.
**Hybrid tested too (job 531, user proposal, `mv_hybrid_student48k_s8.json`)**:
smooth gate exp(−(std/τ)², τ=1 voxel) blending vishand-weighted mean (agreement
regions) into median (disagreement regions) — TIES the median exactly (IoU 0.572
vs 0.573, CD 0.0562 vs 0.0558): where views agree, mean≈median so the gate adds
nothing; where they disagree, the hybrid IS the median. Ablation ladder final:
**mean < vishand < hybrid ≈ median** — the median's crown is tested, not assumed.
Hand-crafted fusion branch CLOSED; only the learned aggregator could beat it.

### 7.15 Overnight 2026-08-25: full-set 25-step table, copy-init pulling ahead

**Full-set teacher fusion @25 (job 528, n=351, `mv_fusion_teacher52k_s25_full.json`)**:
single IoU 0.603 / CD 0.0590 → median_K8 **0.719 / 0.0425** — matches the 8-step
full-set table (§7.13) to ~0.003 IoU: step-invariance holds at n=351 and the
teacher fusion tables are COMPLETE (both step counts, full held-out set).

**Copy-init 8×4 ablation is far ahead of from-scratch at equal steps** (segment 1
ended at 16k, graceful wall; segment 2 = job 516 running → ~32k):
| distill_mse / target_mse | @500 | @4k | @8k | @16k |
|---|---|---|---|---|
| copy-init 8×4 | 0.124 / 0.264 | 0.069 / 0.212 | 0.061 / 0.203 | **0.052 / 0.193** |
| from-scratch 8×2 | 0.342 / 0.483 | — | 0.088 / 0.228 | 0.076 / 0.217 |

Copy-init at 16k ≈ from-scratch at **48–65k** in loss terms (the 8×2 needed 48k+
to reach 0.054/0.195) — a ~3× training-compute win, still falling. Confound to
keep honest: 8×4 has ~1.5× the 8×2's parameters, so init and capacity effects are
mixed; the paired a/b (jobs 532/533, raw 16k ckpt) gives the quality-side read.
Extension (8×2) segment 4 ended at 65k (distill 0.0493/target 0.188, slope
flattening); segment 5 (513) → ~82k tomorrow ~10:00. Queued: 532/533 = copy-init
16k raw @8/25; 534 = student 65k EMA @8 (all paired, `--data_seed 1337`).

### 7.16 FINAL STUDENT DECISION (jobs 543–548, 2026-08-26): Student A @81k

Paired a/b, both finals, 25/8/4 steps (`ab_guidance_student{A81k,B32k}_ds_*`):

| @8 steps unguided | contact −floor | hit1v | IoU | CD | NC | F@0.02 |
|---|---|---|---|---|---|---|
| teacher 52k | 3.31e-3 | 0.639 | 0.679 | 0.0480 | 0.873 | 0.639 |
| **A 81k (8×2, 5 seg)** | **1.70e-3** | **0.666** | **0.608** | 0.0673 | 0.845 | **0.516** |
| B 32k (8×4 copy-init, 2 seg) | 5.80e-3 | 0.555 | 0.576 | **0.0655** | 0.837 | 0.470 |

**Decision: Student A @81k EMA (`outputs/distill_teacherv2/ckpts/
denoiser_ema0.9999_step0081000.pt`) is the paper's student.** It beats B on
IoU/F@0.02/hit1v and by 3.4× on contact excess — and its PHYSICS now BEATS THE
TEACHER (1.70e-3 vs 3.31e-3) at half the parameters and ~⅓ the latency. It is
step-invariant 25→8 (IoU 0.612/0.608) with mild degradation at 4 (0.583, contact
6.8e-3). Guidance fully absorbed on A (oc_flow 4.1e-3 > unguided 2.0e-3 @25); B
is still partially unabsorbed (oc_flow HELPS it: 4.98e-3 < 5.86e-3 — the
undertrained-regime signature, consistent with the absorption law).

**Copy-init ablation verdict (honest both ways):** at EQUAL compute (2 segments)
copy-init wins decisively — B@32k vs A@32k: IoU 0.576 vs 0.491, CD 0.0655 vs
0.0895, contact 5.8e-3 vs 7.1e-3. Copy-init ≈ 2–3× faster convergence. But A,
run 2.5× longer, ended higher — and 48k→65k→81k kept paying (IoU 0.553→0.585→
0.608, contact 3.4e-3→2.6e-3→1.7e-3, slope NOT yet flat). Open options (user):
extend A past 81k (quality still climbing), and/or extend B (2.5× compute
headroom, likely passes A eventually, but 1.5× params ⇒ ~1.4× inference latency
— worse for the real-time story). B's chain can resume:
`sbatch tools/train_distill_teacherv2_s8mlp4.sbatch 3 5`.

### 7.17 THESIS RESULT: fused student A@81k beats the single-view teacher (job 600, 2026-08-26)

Paired 48-group fusion @8 steps (`mv_fusion_studentA81k_s8.json`), same harness
as §7.11 (compare within-harness rows, not against §7.16's a/b subset):

| @8 steps | contact −floor | IoU | CD | F@0.02 |
|---|---|---|---|---|
| teacher single (§7.11) | 3.9e-3 | 0.607 | 0.0501 | 0.606 |
| **A@81k median_K8** | **−4.2e-4 (below floor)** | **0.642** | **0.0466** | **0.646** |
| teacher median_K8 (§7.11) | 7.5e-5 | 0.713 | 0.0358 | 0.742 |

**The paper's thesis experiment now passes**: the fused real-time student beats
the single-view teacher on every metric — at ~⅓ the per-view latency, with K
views batched in one forward. (Teacher+fusion remains the offline quality
ceiling.) Full-set n=351 version = job 599, running.

### 7.18 Visual (silhouette) loss: idea, validation, implementation, fine-tune (2026-08-26)

**User proposal**: the conditioning image maps to the grid such that each pixel
corresponds to a voxel column along the -z view axis — so the 2D masks are exact
3D supervision: (1) object-mask pixels ⇒ the column MUST contain object
("presence"); (2) pixels with neither object nor hand mask ⇒ the column is
PROVABLY empty, since only the hand could occlude ("space carving"); (3) hand
pixels ⇒ no constraint.

**Validation** (`tools/validate_mask_column_correspondence.py`, jobs 614/615,
84 held-out views; orientation found by 8-variant sweep: mask[63−y, x] ↔ grid
(x,y)):
- Raw: IoU mean 0.54 / p90 0.86, presence violations 10.5%, carving 1.5%.
- Diagnosis: BIMODAL — clearly-visible objects align at IoU 0.90–0.95
  (pixel-grade: the dataset construction and the paper's assumption are
  CORRECT); the low tail is tiny/hand-buried objects (dice, marbles) whose
  visible mask is ~empty — no signal, not misalignment.
- With per-view similarity calibration (area-ratio scale + centroid shift vs the
  GT-SDF z-projection) + 1px erosion: presence violations 10.5% → **2.0%**
  (p90 5.2%). Carving artifact learned: erode the OBJECT mask → false carving
  violations; the loss instead DILATES (object ∪ hand) before declaring empty.

**Implementation** (`_add_visual_losses` in
`trellis/trainers/flow_matching/distillation.py`, config-gated):
presence = hinge on soft-min SDF along columns of the eroded calibrated object
mask (margin 1 voxel); carving = relu(−sdf) in columns outside the dilated
mask union; per-sample calibration fitted against the GT projection (GT decode
already available in the physics block); views with <30 visible-object pixels
get zero weight; same time-weighting + CFG-drop gating as physics. Config:
`..._distill_teacherv2_visual_ft.json` (λ_presence = λ_carving = 2.0, λ_ni
fixed 30 no-warmup).

**Fine-tune arm = job 616** (`tools/train_distill_visual_ft.sbatch`): 1×24h,
2 GPUs, fresh `outputs/distill_visual_ft/`, warm-started from A@81k EMA.
Readout (tomorrow): triple comparison A@81k vs A+visual-ft vs A-extension
(~114k, jobs 610/612 running) — separates "visual loss helped" from "more
training helped" at comparable extra compute. Prediction: carving cuts
floating-geometry outliers (CD/EMD tails), partially replicating fusion's
benefit at single-view latency. If confirmed, the mask terms also become
measurement-guidance candidates at inference (masks are observations — the
absorption law §7.13 says measurement terms keep helping).

### 7.19 P4 recursive student: APPROVED (user decision 2026-08-26), queued after current tests

The streaming demo has two tiers: (i) ring-buffer median — NO training, per
frame: student forward (0.52 s) → warp to canonical → per-voxel median over the
last K SDFs (~1 MB each); statistically correct per §7.13 (fuse INDEPENDENT
reconstructions; never couple during generation). (ii) **P4 learned recursion —
user-approved, to run after the current tests/trainings**. Recipe (agreed):
- Architecture: warped prior reconstruction → frozen VAE encoder → 16³×8 latent
  → NEW zero-init input layer added to the token embedding (the input_layer_x0h
  pattern). Zero-init ⇒ step-0 behavior == student A exactly; warm-start from
  the best student ckpt. NO base-model re-pretraining (precedent: hand
  conditioning itself was added to the vision-pretrained base the same way).
- Data: simulated streaming from the 24-view grasps — prior = warped fusion of
  OTHER views of the same grasp; curriculum corrupted-GT → precomputed frozen-
  student reconstructions (one-time offline pass, ~2-3 GPU-days, subsample views
  to halve); prior dropout ~30% (preserves single-view mode; anti-copy);
  pose-jitter on the warp for rig robustness. Copy-shortcut is further blocked
  by construction: the prior comes from other views, wrong exactly where the
  current image is informative.
- Losses unchanged (distill + physics + visual if §7.18 validates).
- Budget: ~4-6 GPU-days total. Expected: beats post-hoc median at equal K
  (learned integration), upgrades the demo to "model integrates observations
  over the grasp".
