import json, sys, os, glob, numpy as np, yaml
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
sys.path.insert(0, "tools")
from hand_pose_registration import estimate_similarity, warp_sdf_affine
R = "/projects/gcaddeo/inference/TRELLIS/dex-full"
b = json.load(open(f"{R}/benchmark_groups.json")); CAMS = b["__meta__"]["cams"]; groups = b["groups"]
ext_dates = sorted(os.path.basename(p).split("_")[1] for p in glob.glob(f"{R}/calibration/extrinsics_*"))
_ext_cache = {}
def ext_for(date):
    d = [e for e in ext_dates if e <= date][-1]
    if d not in _ext_cache:
        y = yaml.load(open(glob.glob(f"{R}/calibration/extrinsics_{d}_*/extrinsics.yml")[0]), Loader=yaml.UnsafeLoader)
        _ext_cache[d] = {k: np.array(v, dtype=np.float64).reshape(3, 4) for k, v in y["extrinsics"].items()}
    return d, _ext_cache[d]
def verts(path):
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith("v ")])
def load(I, p): return np.load(f"{R}/data_pose_norm/{I}/sdfs/{I}_f000__{p}.npy").astype(np.float32)
def iou(a, b): A = a < 0; B = b < 0; return (A & B).sum() / max(1, (A | B).sum())
dx = 2 / 64
def sample(sdf, pts):
    return map_coordinates(sdf, ((pts + 1) / dx - 0.5).T, order=1, mode="nearest")
def fit_norm(I):
    """grid = s * x_m + t (no rotation): least squares |sdf(s v + t)| over mesh vertices of both parts."""
    V = {p: verts(f"{R}/renders_cond/{I}/{p}.obj") for p in ("hand", "object")}
    S = {p: load(I, p) for p in ("hand", "object")}
    vm = np.concatenate(list(V.values()))
    vg = np.concatenate([np.load(f"{R}/data_pose_norm/{I}/idxs/{I}_f000__{p}.npy") for p in ("hand", "object")])
    vg = -1 + (vg + 0.5) * dx
    s0 = ((vg.max(0) - vg.min(0) + dx) / (vm.max(0) - vm.min(0))).mean(); t0 = (vg.min(0) - dx/2) - s0 * vm.min(0)
    def resid(p):
        s, t = p[0], p[1:]
        return np.concatenate([np.clip(sample(S[k], s * V[k] + t), -0.1, 0.1) for k in V])
    sol = least_squares(resid, np.r_[s0, t0], loss="soft_l1", f_scale=0.02)
    return sol.x[0], sol.x[1:], float(np.abs(sol.fun).mean())
def M_grid_to_cam(I, flip):
    s, t, _ = fit_norm(I); A = np.eye(4); A[:3, :3] = np.eye(3) / s; A[:3, 3] = -t / s
    Fm = np.eye(4); Fm[:3, :3] = np.diag([1., -1., -1.]) if flip else np.eye(3)
    return Fm @ A
def M_cam_to_world(ext, serial, inv):
    M = np.eye(4); M[:3, :] = ext[serial]
    return np.linalg.inv(M) if inv else M
def gt_similarity(ext, I_src, cam_src, I_dst, cam_dst, flip, inv):
    """(a, R, t) with x_src = a R x_dst + t between the two instances' grids."""
    T_sd = np.linalg.inv(M_grid_to_cam(I_dst, flip)) @ np.linalg.inv(M_cam_to_world(ext, cam_dst, inv)) \
           @ M_cam_to_world(ext, cam_src, inv) @ M_grid_to_cam(I_src, flip)
    T_ds = np.linalg.inv(T_sd); A = T_ds[:3, :3]; a = np.cbrt(np.linalg.det(A))
    return a, A / a, T_ds[:3, 3]
def rot_angle(Rm): return np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))

keys = sorted(groups)
k = keys[0]; d, ext = ext_for(k.split("|")[1]); insts = groups[k]
print("group", k, "extrinsics", d, "det R:", [round(np.linalg.det(ext[c][:, :3]), 3) for c in CAMS[:3]])
for I in insts[:2]:
    s, t, c = fit_norm(I); print("  norm fit", I[-30:], "s %.4f t %s resid %.4f" % (s, np.round(t, 4), c))
ref = insts[0]; oo, ho = load(ref, "object"), load(ref, "hand")
best = {}
for v in (1, 3, 5):
    I = insts[v]; o, h = load(I, "object"), load(I, "hand")
    a_e, R_e, t_e, _ = estimate_similarity(h, ho)
    print(f"view {v}: registration a={a_e:.3f} rot={rot_angle(R_e):.1f}  objIoU {iou(warp_sdf_affine(o, a_e, R_e, t_e), oo):.3f}")
    for flip in (True, False):
        for inv in (False, True):
            a, Rm, t = gt_similarity(ext, I, CAMS[v], ref, CAMS[0], flip, inv)
            io = iou(warp_sdf_affine(o, a, Rm, t), oo)
            print(f"   flip={flip} inv={inv}: a={a:.3f} dRot vs reg {rot_angle(Rm.T @ R_e):.2f}deg objIoU {io:.3f} handIoU {iou(warp_sdf_affine(h, a, Rm, t), ho):.3f}")
            best[(flip, inv)] = best.get((flip, inv), 0) + io
conv = max(best, key=best.get); print("CONVENTION flip,inv =", conv)
if "--sweep" in sys.argv:
    rng = np.random.default_rng(0); sel = [keys[i] for i in rng.choice(len(keys), 40, replace=False)]
    rows = []
    for k in sel:
        d, ext = ext_for(k.split("|")[1]); insts = groups[k]; ref = insts[0]
        if not all(os.path.exists(f"{R}/renders_cond/{I}/hand.obj") for I in insts): continue
        oo, ho = load(ref, "object"), load(ref, "hand")
        for v in range(1, 8):
            I = insts[v]; o, h = load(I, "object"), load(I, "hand")
            a, Rm, t = gt_similarity(ext, I, CAMS[v], ref, CAMS[0], *conv)
            est = estimate_similarity(h, ho)
            if est is None: continue
            a_e, R_e, t_e, _ = est
            rows.append((iou(warp_sdf_affine(o, a, Rm, t), oo), iou(warp_sdf_affine(o, a_e, R_e, t_e), oo),
                         iou(warp_sdf_affine(h, a, Rm, t), ho), rot_angle(Rm.T @ R_e), np.linalg.norm(t - t_e) / dx, abs(a_e / a - 1) * 100))
    arr = np.array(rows)
    print(f"sweep {len(rows)} pairs")
    for name, col in zip(["GT-ext objIoU", "reg objIoU", "GT-ext handIoU", "rot err deg", "trans err vox", "scale err %"], arr.T):
        q = np.percentile(col, [5, 50, 95]); print(f"  {name:15s} mean {col.mean():.3f}  p5/50/95 {q[0]:.3f} {q[1]:.3f} {q[2]:.3f}")
    print("  frac GT-ext objIoU<0.8:", (arr[:, 0] < 0.8).mean(), " reg<0.8:", (arr[:, 1] < 0.8).mean())
