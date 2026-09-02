import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiview_warp import load_view
from hand_pose_registration import estimate_similarity, gt_relative, _sample, _grid_pts, D

inst_dir = "datasets_split/Leap_Hand_test/data_pose_norm"
inst = sorted(os.listdir(inst_dir))[0]
d = os.path.join(inst_dir, inst)
m_i, s_i = load_view(d, inst, 3); m_j, s_j = load_view(d, inst, 0)
h_i, h_j = s_i["hand"], s_j["hand"]
print("hand sdf ranges:", h_i.min(), h_i.max(), "| frac_neg", (h_i<0).mean())

a_g, R_g, t_g = gt_relative(m_i, m_j)
a_e, R_e, t_e, diag = estimate_similarity(h_i, h_j)
print(f"GT  a={a_g:.4f}  EST a={a_e:.4f}  (ratio {a_e/a_g:.4f})")

surf = _grid_pts(np.abs(h_j) < 0.05)
rng = np.random.default_rng(0); surf = surf[rng.choice(len(surf), 4000, replace=False)]
def cost(a, R, t):
    return np.mean(np.abs(_sample(h_i, (a*(R@surf.T)).T + t, cval=0.05*a) / a))
print(f"cost @GT  = {cost(a_g, R_g, t_g):.5f}")
print(f"cost @EST = {cost(a_e, R_e, t_e):.5f}")
# 1-D scale sweep around GT rotation/translation, adjusting t to keep centroid fixed
in_j = h_j < 0; c_j = _grid_pts(in_j).mean(0)
c_i_gt = a_g*(R_g@c_j) + t_g
for f in [0.94, 0.97, 1.0, 1.03, 1.06]:
    aa = a_g * f
    tt = c_i_gt - aa*(R_g@c_j)
    print(f"  scale x{f:.2f}: cost {cost(aa, R_g, tt):.5f}")
