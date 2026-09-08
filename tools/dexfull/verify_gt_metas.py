"""GT->GT warp through the materialized GT-pose metas of dex-full-groups (harness meta path)."""
import json, os, sys, numpy as np
sys.path.insert(0, "tools")
from multiview_warp import warp_sdf
from hand_pose_registration import estimate_similarity, warp_sdf_affine
G = "/projects/gcaddeo/inference/TRELLIS/dex-full-groups"
import pandas as pd
m = pd.read_csv(f"{G}/metadata.csv"); ok = m[m.cond_rendered].sha256.tolist()
rng = np.random.default_rng(1); sel = [ok[i] for i in rng.choice(len(ok), 30, replace=False)] + ok[:4]
def load(g, v, p): return np.load(f"{G}/data_pose_norm/{g}/sdfs/{g}_f{v:03d}__{p}.npy").astype(np.float32)
def meta(g, v): return json.load(open(f"{G}/data_pose_norm/{g}/{g}_f{v:03d}_meta.json"))
def iou(a, b): A = a < 0; B = b < 0; return (A & B).sum() / max(1, (A | B).sum())
rows = []
for g in sel:
    mr = meta(g, 0); assert not os.path.islink(f"{G}/data_pose_norm/{g}/{g}_f000_meta.json"), "meta still a symlink"
    assert "pose_export" in mr, "not a GT meta"
    oo, ho = load(g, 0, "object"), load(g, 0, "hand")
    for v in range(1, 8):
        mv = meta(g, v); o, h = load(g, v, "object"), load(g, v, "hand")
        wo = warp_sdf(o, mv, mr); wh = warp_sdf(h, mv, mr)
        band = np.abs(wo - oo)[np.abs(oo) < 0.1].mean()
        est = estimate_similarity(h, ho)
        ri = iou(warp_sdf_affine(o, est[0], est[1], est[2]), oo) if est else np.nan
        rows.append((iou(wo, oo), iou(wh, ho), band, ri, mv["pose"]["fit_residual"]))
a = np.array(rows)
print(f"{len(rows)} pairs over {len(sel)} groups")
for n, c in zip(["meta-GT obj IoU", "meta-GT hand IoU", "obj band|dsdf|", "registration obj IoU", "norm fit resid"], a.T):
    q = np.nanpercentile(c, [5, 50, 95]); print(f"  {n:22s} mean {np.nanmean(c):.4f}  p5/50/95 {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}")
print("  frac meta-GT obj IoU < 0.8:", (a[:, 0] < 0.8).mean())
assert np.median(a[:, 0]) > 0.9, "GT metas do not reproduce the extrinsics warp"
print("VERIFY OK")
