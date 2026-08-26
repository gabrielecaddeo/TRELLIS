"""P4 CPU self-test (run on a compute node; login OOMs).
1. zero-init prior branch is an exact no-op; keep=0 gates it off.
2. trellis/utils/mv_warp_np == tools/multiview_warp on a real pair.
3. Prior dataset yields prior_sdf/prior_keep with sane stats.
Delete after the P4 smoke passes if it has no further use.
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trellis.models.sparse_structure_flow import SparseStructureFlowModelConditioned
m = SparseStructureFlowModelConditioned(
    resolution=16, in_channels=8, out_channels=8, model_channels=64,
    cond_channels=64, num_blocks=1, num_heads=4, mlp_ratio=2, patch_size=1,
    pe_mode="ape", qk_rms_norm=True, use_touch=True, use_encoding_hand=True,
    use_weighted_attention=True, use_prior=True)
assert m.input_layer_prior.weight.abs().max().item() == 0
assert m.input_layer_prior.bias.abs().max().item() == 0
B = 2
x = torch.randn(B, 8, 16, 16, 16); t = torch.tensor([100.0, 500.0])
# cond/cond_mask mimic DINOv2 ViT-L/14 @518: 1369 patch tokens + 5 (cls+registers)
cond = {"cond": torch.randn(B, 1374, 64), "cond_mask": torch.randn(B, 1374, 64),
        "mask_hand": torch.rand(B, 37, 37), "mask_obj": torch.rand(B, 1369, 1),
        "x0_hand": torch.randn(B, 8, 16, 16, 16), "touch": torch.randn(B, 2, 64, 64, 64)}
with torch.no_grad():
    y0 = m(x, t, dict(cond))
    cond2 = dict(cond); cond2["x0_prior"] = torch.randn(B, 8, 16, 16, 16); cond2["prior_keep"] = torch.ones(B)
    y1 = m(x, t, cond2)
assert torch.equal(y0, y1), "zero-init prior branch is NOT a no-op"
print("1. zero-init no-op OK")
torch.nn.init.normal_(m.input_layer_prior.weight, std=0.02)
with torch.no_grad():
    y2 = m(x, t, dict(cond2))
    cond3 = dict(cond2); cond3["prior_keep"] = torch.zeros(B)
    y3 = m(x, t, cond3)
assert (y2 - y0).abs().max().item() > 0, "prior inert with nonzero weights"
assert torch.equal(y3, y0), "keep=0 does not gate the prior off"
print("1b. gating OK")

from multiview_warp import load_view, warp_sdf as warp_a
from trellis.utils.mv_warp_np import warp_sdf as warp_b
inst_dir = "datasets_split/Hands_test/data_pose_norm"
inst = sorted(os.listdir(inst_dir))[0]
d = os.path.join(inst_dir, inst)
m0, s0 = load_view(d, inst, 0); m5, _ = load_view(d, inst, 5)
wa = warp_a(s0["object"], m0, m5); wb = warp_b(s0["object"], m0, m5)
assert np.array_equal(wa, wb), "mv_warp_np diverges from tools/multiview_warp"
print("2. warp copy OK")

from trellis import datasets
np.random.seed(0)
ds = datasets.ImageConditionedSparseStructureLatentSDFConditionedPrior(
    "datasets_split/Hands_test",
    latent_model="vae_final_all_resume_2_0300000", min_aesthetic_score=4.5,
    image_size=518,
    pretrained_ss_dec="microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
    ss_dec_path="/projects/gcaddeo/train_flow/TRELLIS/outputs/vae_final_all_resume_2",
    ss_dec_ckpt="step0300000",
    prior_source="gt_corrupt", prior_k_max=7, prior_dropout=0.3)
keeps, fracs = [], []
for i in range(12):
    it = ds[i % len(ds)]
    assert it["prior_sdf"].shape == (1, 64, 64, 64), it["prior_sdf"].shape
    keeps.append(float(it["prior_keep"]))
    if it["prior_keep"] > 0:
        fracs.append(float((it["prior_sdf"] < 0).float().mean()))
print(f"3. dataset OK: keep_rate={np.mean(keeps):.2f} (expect ~0.7), "
      f"inside_frac kept priors mean={np.mean(fracs):.4f} (expect ~0.02-0.04)")
assert 0 < np.mean(keeps) < 1 and all(0.001 < f < 0.3 for f in fracs)
print("ALL P4 CPU SELF-TESTS PASSED")
