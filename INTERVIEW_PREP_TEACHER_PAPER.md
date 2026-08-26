# Interview prep — the published hand-conditioned reconstruction work

Prepared 2026-08-26 for a technical-journal interview about the PUBLISHED paper
only (no distillation, no follow-up losses, no multi-view work — do not mention
ongoing campaigns beyond a generic "we are extending this toward real-time use").
Written from the codebase and campaign documentation; **cross-check any number
or claim you quote against the paper PDF — where they disagree, the paper wins.**

---

## 1. The one-paragraph story (open with this)

We reconstruct the full 3D shape of an object while a robot hand is holding it,
from a single RGB image. The twist: the hand — normally the *problem*, because
it occludes much of the object — becomes part of the *solution*. A robot knows
its own hand exactly (joint angles → full hand geometry), so we feed the hand's
3D shape, its contact points, and its image silhouette into a generative 3D
model as conditioning, and we train that model with physics-aware losses: the
reconstructed object may not interpenetrate the hand, and it must actually touch
the hand where the grasp says it does. The result is a reconstruction that is
not only visually plausible but *physically consistent with the grasp* — which
is what a robot needs if it wants to re-grasp, hand over, or manipulate the
object.

## 2. Why a generative model at all (the core insight)

Single-view, in-hand reconstruction is radically ill-posed: the camera sees one
side, and the hand hides a large part of even that side. No regression can
"compute" the back of the object — it must be *imagined* consistently with
priors. That is exactly what a generative model provides: a learned prior over
object shapes, steered by whatever evidence exists. Our evidence is unusually
rich for a robotics setting: the image (appearance), plus the hand (geometry we
know *exactly* because the robot proprioceives its own joints). The design
question of the paper is: how do you inject that knowledge into a generative
model so it is actually used? Answer: as conditioning channels AND as training
losses — belt and braces.

## 3. The backbone: flow matching in a 3D latent space

**Representation.** Objects (and the hand) live as Signed Distance Fields on a
64³ grid over a normalized cube: every voxel stores the distance to the surface,
negative inside. Why SDF and not occupancy or meshes: (i) the surface is
recovered at sub-voxel precision by marching cubes at level 0; (ii) *distances
are physically meaningful* — penetration depth and contact distance, the
quantities our physics losses need, are literally readable off the grid; (iii)
SDFs are smooth, so gradients behave.

**Latent compression.** A 3D-convolutional VAE compresses the 64³ SDF into a
16³ × 8-channel latent (a factor ~256). The generative model works there. Why:
a transformer over 64³ = 262k tokens is intractable; over 16³ = 4,096 tokens it
is comfortable. The VAE is trained first and then frozen — the generative model
never sees raw voxels.

**Flow matching (know this cold — it will be asked).** We use rectified flow /
flow matching, the modern, simpler cousin of diffusion:
- Define a *straight-line* path between data and noise:
  x_t = (1 − t)·x₀ + t·ε, with x₀ the clean latent, ε ~ N(0, I), t ∈ [0, 1].
- The model is trained to predict the *velocity* of that path,
  v = ε − x₀, given (x_t, t, conditioning). The loss is a plain MSE:
  L = E‖v̂(x_t, t, c) − (ε − x₀)‖². That's the whole training objective for
  the generative part — no ELBOs, no noise schedules to tune.
- From a velocity estimate you get a clean-sample estimate *in closed form*:
  x̂₀ = x_t − t·v̂. This little identity is load-bearing: it is what lets us
  evaluate physics on the *decoded geometry* at any point of training or
  sampling, because at every noise level we can ask the model "what do you
  currently believe the object is?" and decode that belief.
- Sampling is numerical integration of an ODE: start from pure noise at t=1,
  take Euler steps along the predicted velocity down to t=0. We use 25 steps
  with a *time rescaling* that concentrates steps near t=1. Insight: in flow
  models the coarse structural decisions (where is the object, how big, what
  topology) happen at high noise; the low-noise end only refines details. So
  you spend your step budget where decisions are made. The same reasoning sets
  the training-time distribution of t (a logit-normal biased toward high
  noise): train hardest where the problem is hardest.
- Why flow matching over classic diffusion, if asked: same expressive power,
  simpler objective, straighter sampling trajectories (fewer steps for the same
  quality), and a cleaner x̂₀ estimator — which our physics machinery relies on.

**Classifier-free guidance (CFG).** During training, conditioning is randomly
dropped ~10% of the time, so the same network learns both the conditional and
unconditional velocity. At sampling, we extrapolate: v = v_uncond +
s·(v_cond − v_uncond) with s = 5. One refinement worth mentioning: we apply CFG
only on the high-noise *interval* of the trajectory (roughly the first half).
Insight: guidance is for steering *decisions*, and decisions happen at high
noise; at low noise strong CFG mostly amplifies artifacts — and skipping it
there also halves the compute for that part of the trajectory.

## 4. The hand conditioning (the paper's heart)

Three channels, each carrying a different kind of knowledge, each injected
where it is most natural:

1. **The hand's own SDF, encoded with the *same* VAE** into the same latent
   space as the object. Insight: representation alignment — the transformer
   compares hand-geometry tokens and object-geometry tokens in the same
   coordinate language, so "am I inside the hand?" is an easy relation to
   learn. The hand latent enters both through the input (fused with the noisy
   object latent) and through dedicated cross-attention in every block.
2. **A contact/touch volume**: the grasp's contact points rasterized into the
   grid (plus a distance-to-contact field), encoded by a small 3D CNN and fused
   at the input. Insight: contacts are the strongest single cue about the
   hidden side of the object — where the fingertips are, the surface *must* be.
   This is proprioceptive/tactile information, image-free.
3. **The 2D hand mask**, tokenized with its own positional embedding and its
   own cross-attention stream. Insight: this tells the model *which image
   pixels to distrust* — appearance under the hand mask belongs to the hand,
   not the object.
- The image itself is encoded by a frozen DINOv2 ViT (self-supervised
  features): robust, semantic, and not fooled by synthetic-vs-real texture
  differences — the main reason sim-trained conditioning transfers to real
  captures.

If asked "why so many channels instead of just the image": each channel is
information the robot gets *for free* (its own kinematics, its own touch
sensors), and each covers a blind spot of the others. The image sees the
visible side; the contacts constrain the hidden side; the hand geometry defines
the free space; the mask arbitrates the boundary.

## 5. The physics losses (the second contribution)

During training, at each step we take the model's current belief x̂₀ = x_t −
t·v̂, decode it through the *frozen, differentiable* VAE decoder into an actual
64³ SDF, and impose two penalties on that geometry:

- **Non-interpenetration**: the object's SDF may not be negative (inside)
  where the hand's interior is. Solid objects do not overlap — a purely
  physical fact the data alone teaches too slowly.
- **Contact**: at the annotated contact voxels, the object's |SDF| should be
  ~0 — the surface must pass through the fingertips. A stable grasp *implies*
  contact; a reconstruction floating a millimeter off the fingers is wrong
  even if it looks right.

Both are weighted toward the low-noise end of training (where x̂₀ is a
meaningful shape rather than a blur), and both flow gradients through the
decoder back into the latent-space model. Insight to articulate: **we constrain
the geometry, not the latent.** The latent space is an opaque learned code;
physics lives in metric space. Putting the frozen decoder inside the training
loop is what connects the two — it's the differentiable bridge between "what
the generative model says" and "what Newton says".

At inference time, the same energies can optionally *guide* sampling: at each
Euler step, decode x̂₀, take the gradient of the physics energy with respect to
the current state, and nudge the velocity accordingly (a DPS-style correction).
This is a test-time knob — training-time physics does the heavy lifting.

## 6. Data: why synthetic, and how

Training data is synthetic: thousands of grasps of household objects (YCB and
large object collections) in simulated robot hands, each rendered from 24
viewpoints. Per view the pipeline produces exactly the supervision this method
needs and that no real dataset provides at scale: the object's GT SDF, the
hand's SDF, contact annotations, pixel-perfect hand/object masks, and the VAE
latents — all in a pose-normalized grid aligned to the camera. Insight: this
task *needs* simulation, because ground-truth signed distance fields and
contact labels are essentially uncapturable in the real world; and the sim2real
gap is absorbed mostly by conditioning on DINOv2 features (semantics, not
pixels) and on geometry (hand pose), which transfers exactly.

Evaluation is on **real captures**: a DexYCB-derived benchmark (~1000 in-hand
frames of YCB objects, real cameras, real hands-with-objects), reconstructing
from one view per instance and comparing meshes against the known YCB models.

## 7. Metrics — and the one honest subtlety

Chamfer distance (squared), Normal Consistency, F-score at increasing
thresholds, and Earth Mover's Distance, all computed after **ICP alignment** in
a normalized frame. Why ICP: a single view fixes shape much better than it
fixes global pose; without alignment, the metric is dominated by a rigid pose
offset and says nothing about reconstruction quality. If pressed for numbers,
use the paper's published table (verify against the PDF); our independently
reproduced values for the published model on the real benchmark are CD²
0.046/0.018 (mean/median), NC 0.81, F@0.02 0.13 rising to F@0.10 0.70 — quote
the paper's own figures in print.

## 8. Model card (if asked for specs)

- Diffusion-transformer (DiT-style): 24 blocks, width 1024, 16 heads, MLP
  ratio 4 (~0.75B parameters), operating on 4,096 latent tokens (16³).
- Per block: adaptive layer-norm modulation on the timestep, self-attention,
  and three cross-attention streams (image tokens / hand-mask tokens /
  hand-latent tokens).
- Frozen components: DINOv2 ViT-L image encoder; 3D-conv VAE encoder/decoder.
- Inference: 25 Euler steps, CFG 5.0 on the high-noise interval, one image +
  hand state in, one mesh out via marching cubes.

## 9. Insights worth saying out loud (the quotable lines)

- "The hand is both the problem and the answer: it hides the object, but a
  robot knows its own hand perfectly — geometry, pose, and touch — and that
  knowledge constrains the hidden shape more than another camera would."
- "We don't ask the network to be a physicist; we decode its belief into
  actual geometry at every training step and penalize physical nonsense there.
  The differentiable decoder is the bridge between the learned latent space
  and metric space."
- "Generative models hallucinate — that's their job; the unseen side of the
  object must be invented. Our contribution is making them hallucinate
  *responsibly*: consistent with contact, free of interpenetration."
- "Everything the model is conditioned on is information a robot already has
  for free. No extra sensors, no scanning motion — one glance."

## 10. Likely questions, suggested answers

- *Why not classic multi-view / photogrammetry?* One glance is the setting:
  manipulation shouldn't pause for a scanning trajectory. And classic methods
  reconstruct the visible; the grasped object is mostly not visible.
- *Why not NeRF/Gaussian splatting?* Those are per-scene optimizations needing
  many views; we need a *prior-driven*, feed-forward prediction from one view,
  in a representation (SDF) where physics is expressible.
- *How does it handle unseen objects?* The prior generalizes at the category/
  shape level (trained across thousands of shapes); evaluation is on real
  captures with the standard split conventions — check the PDF's exact wording
  before claiming "never-seen categories".
- *Failure modes?* (Be honest.) Thin structures at 64³ resolution; objects
  almost fully enclosed by the hand (little evidence, prior takes over);
  global pose ambiguity from one view (why metrics are ICP-aligned); sim2real
  texture gaps on unusual materials.
- *Is it real-time?* The published system is not; it runs in seconds per
  object. Say: "making this real-time without losing the physics is exactly
  the direction we're pushing now" — and leave it there.
- *What's the training cost?* Order of a few GPU-weeks on modern accelerators,
  dominated by the flow model; the VAE and image encoder are pretrained and
  frozen.
- *Why flow matching instead of diffusion?* Simpler loss (one MSE), straighter
  trajectories → fewer sampling steps, and a clean closed-form current-shape
  estimate that the physics losses and guidance are built on.

## 11. Do NOT bring up (scope guard)

Distillation/students, the retrained teacher, guidance-absorption findings,
multi-view fusion, consistency guidance, silhouette losses, latency numbers on
specific GPUs, or any result from the ongoing campaign. If the interviewer
probes future work: "faster models, exploiting the video stream during
manipulation, and tighter physics — ongoing work." Nothing more specific.
