"""Validation for the proposed silhouette/space-carving loss (user idea,
2026-08-26): does each conditioning-image pixel correspond to a voxel column
along the grid's -z view axis, exactly enough to derive 3D supervision?

For held-out instances/views, using the DATASET'S OWN mask transform
(get_instance -> pack['mask_obj'/'mask_hand'] at 518^2, downsampled to 64^2):
  proj[x,y]      = any_z( GT object SDF < 0 )        (occupancy projection)
  mask64         = dataset mask, resized to 64^2
Orientation between (x,y) grid axes and (row,col) image axes is unknown a
priori -> swept over the 8 transpose/flip variants on the first views and then
fixed (must be consistent).

Reported per view and aggregate:
  iou              IoU(proj, mask_obj64)
  presence_viol    P(column has NO object | object pixel)      -> loss claim 1
  carving_viol     P(column HAS object | empty-empty pixel)    -> loss claim 2
  hand_pix_frac    fraction of pixels that are hand (no-constraint zone)
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from easydict import EasyDict as edict
from trellis import datasets

ORIENTS = [(t, fr, fc) for t in (0, 1) for fr in (0, 1) for fc in (0, 1)]

def orient(a, t, fr, fc):
    if t: a = a.T
    if fr: a = a[::-1, :]
    if fc: a = a[:, ::-1]
    return a

def to64(mask_t):
    img = Image.fromarray((mask_t.numpy() * 255).astype(np.uint8))
    return np.asarray(img.resize((64, 64), Image.BILINEAR)).astype(np.float32) / 255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--config", default="configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill_teacherv2.json")
    ap.add_argument("--num_instances", type=int, default=24)
    ap.add_argument("--views", default="0,5,11,17")
    ap.add_argument("--fit_correction", action="store_true",
                    help="Additionally fit a per-view 2D similarity (scale in "
                         "[0.8,1.3], integer shifts +-8) aligning the mask to the "
                         "projection, and report corrected metrics — tests whether "
                         "the misalignment is a fixable scale/shift error.")
    ap.add_argument("--erode", type=int, default=0,
                    help="Erode both masks by N pixels (at 64^2) before violation "
                         "metrics — separates boundary aliasing from real misalignment.")
    ap.add_argument("--output", default="outputs/diagnostics/mask_column_validation.json")
    args = ap.parse_args()

    cfg = edict(json.load(open(args.config)))
    ds = getattr(datasets, cfg.dataset.name)(args.data_dir, **cfg.dataset.args)
    ds.inference = True
    views = [int(v) for v in args.views.split(",")]

    per, orient_votes = [], {}
    chosen = None
    for i in range(min(args.num_instances, len(ds.instances))):
        root, inst = ds.instances[i]
        for v in views:
            try:
                ds.force_view = v
                pack = ds.get_instance(root, inst)
            except Exception as e:
                print(f"[skip] {inst} v{v}: {type(e).__name__}: {e}"); continue
            finally:
                ds.force_view = None
            sdf = np.load(os.path.join(root, "data_pose_norm", inst, "sdfs",
                                       f"{inst}_f{v:03d}__object.npy"))
            proj = (sdf < 0).any(axis=2).astype(np.float32)       # [x,y]
            mobj = to64(pack["mask_obj"]); mhand = to64(pack["mask_hand"])

            if chosen is None:
                # vote over the first batch of views
                best = max(ORIENTS, key=lambda o: (np.logical_and(orient(proj, *o) > .5, mobj > .5).sum()
                                                   / max(np.logical_or(orient(proj, *o) > .5, mobj > .5).sum(), 1)))
                orient_votes[best] = orient_votes.get(best, 0) + 1
                if sum(orient_votes.values()) >= 12:
                    chosen = max(orient_votes, key=orient_votes.get)
                    print(f"orientation votes: {orient_votes} -> chosen {chosen}")
                continue

            p = orient(proj, *chosen)
            if args.fit_correction:
                from scipy.ndimage import zoom as ndzoom

                def transform(m, sc, dr, dc):
                    mz = ndzoom(m, sc, order=1)
                    can = np.zeros((96, 96), np.float32)
                    h, w = mz.shape
                    r0, c0 = (96 - h) // 2, (96 - w) // 2
                    rs, cs = max(r0, 0), max(c0, 0)
                    can[rs:rs + min(h, 96), cs:cs + min(w, 96)] = mz[:min(h, 96), :min(w, 96)]
                    r, c = 16 + dr, 16 + dc
                    return can[r:r + 64, c:c + 64]

                best_iou, best_prm = -1.0, (1.0, 0, 0)
                for sc in np.linspace(0.8, 1.3, 11):
                    for dr in range(-8, 9, 2):
                        for dc in range(-8, 9, 2):
                            m2 = transform(mobj, sc, dr, dc)
                            iou_t = (np.logical_and(p > .5, m2 > .5).sum()
                                     / max(np.logical_or(p > .5, m2 > .5).sum(), 1))
                            if iou_t > best_iou:
                                best_iou, best_prm = iou_t, (sc, dr, dc)
                mobj = transform(mobj, *best_prm)
                mhand = transform(mhand, *best_prm)
            if args.erode:
                from scipy.ndimage import binary_erosion
                mobj = binary_erosion(mobj > .5, iterations=args.erode).astype(np.float32)
            po, mo, mh = p > .5, mobj > .5, mhand > .5
            iou = np.logical_and(po, mo).sum() / max(np.logical_or(po, mo).sum(), 1)
            obj_pix = mo & ~mh
            empty_pix = ~mo & ~mh
            presence_viol = float((~po & obj_pix).sum() / max(obj_pix.sum(), 1))
            carving_viol = float((po & empty_pix).sum() / max(empty_pix.sum(), 1))
            per.append(dict(inst=inst, view=v, iou=float(iou),
                            presence_viol=presence_viol, carving_viol=carving_viol,
                            hand_pix_frac=float(mh.mean())))
    agg = {k: (float(np.mean([r[k] for r in per])), float(np.percentile([r[k] for r in per], 90)))
           for k in ("iou", "presence_viol", "carving_viol", "hand_pix_frac")}
    print(f"\nn views = {len(per)}   (mean, p90)")
    for k, v in agg.items():
        print(f"  {k:<15} {v[0]:.4f}  {v[1]:.4f}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({"orientation": chosen, "aggregate": agg, "per_view": per}, open(args.output, "w"), indent=2)
    print(f"wrote {args.output}")

if __name__ == "__main__":
    main()
