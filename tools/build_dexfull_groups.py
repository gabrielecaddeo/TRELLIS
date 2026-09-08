"""Build a train-repo-convention dataset root for the dex-full REAL-CAPTURE
multi-view export (EVAL_GUIDANCE.md §7.27).

dex-full stores one instance per (object, sequence, camera, frame) with a
single frame f000 (000.png, 000_mask1.png, 000_mask2.png, no transforms.json,
no metadata.csv); the 8 simultaneous cameras of a grasp are 8 separate
instances tied together by view_groups.json. The train-repo dataset class
(and therefore streaming_eval / multiview_fusion_eval) wants ONE instance with
views 0..7 inside it. This script materializes that layout with SYMLINKS
(no data copied): one group -> one instance, view v = camera CAMS[v]
(fixed global camera order, sorted serials).

Group instance name: <ycb_object>__<subject>__<seq_time>__f<frame>.

metadata.csv marks ss_latent_<latent> True only when all 16 latent files of
the group exist -> rerun after latent encoding finishes (idempotent).

Usage:
  python tools/build_dexfull_groups.py --groups dex-full/benchmark_groups.json \
      --out /projects/gcaddeo/inference/TRELLIS/dex-full-groups
"""
import os, json, argparse, csv

LATENT = "vae_final_all_resume_2_0300000"


def link(src, dst):
    if os.path.lexists(dst):
        return
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/projects/gcaddeo/inference/TRELLIS/dex-full")
    ap.add_argument("--groups", required=True, help="benchmark_groups.json (cams + groups)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    spec = json.load(open(args.groups))
    cams = spec["__meta__"]["cams"]
    groups = spec["groups"]
    keys = sorted(groups)
    if args.limit:
        keys = keys[:args.limit]
    lat_src = os.path.join(args.src, "ss_latents_sdf_pose", LATENT)
    lat_dst = os.path.join(args.out, "ss_latents_sdf_pose", LATENT)
    for d in ("renders_cond", "data_pose_norm", lat_dst):
        os.makedirs(os.path.join(args.out, d) if not os.path.isabs(d) else d, exist_ok=True)

    rows, n_lat, n_missing_render = [], 0, 0
    manifest = {"__meta__": {"src": args.src, "cams": cams, "latent": LATENT,
                             "note": "view v of a group instance == camera cams[v]; "
                                     "all 8 views are SIMULTANEOUS captures (DexYCB 8-cam rig)"},
                "groups": {}}
    for k in keys:
        subj, date, tim, frame = k.split("|")
        insts = groups[k]
        obj = insts[0].split(f"_{date}_")[0].replace("custom_objects_", "")
        G = f"{obj}__{subj}__{tim}__f{frame}"
        rc = os.path.join(args.out, "renders_cond", G)
        dp = os.path.join(args.out, "data_pose_norm", G)
        for sub in ("", "sdfs", "contacts", "idxs"):
            os.makedirs(os.path.join(dp, sub), exist_ok=True)
        os.makedirs(rc, exist_ok=True)
        frames, ok_render, ok_lat = [], True, True
        for v, I in enumerate(insts):
            s_rc = os.path.join(args.src, "renders_cond", I)
            s_dp = os.path.join(args.src, "data_pose_norm", I)
            if not os.path.exists(os.path.join(s_rc, "000.png")):
                ok_render = False
            link(os.path.join(s_rc, "000.png"), os.path.join(rc, f"{v:03d}.png"))
            link(os.path.join(s_rc, "000_mask1.png"), os.path.join(rc, f"{v:03d}_mask_1.png"))
            link(os.path.join(s_rc, "000_mask2.png"), os.path.join(rc, f"{v:03d}_mask_2.png"))
            link(os.path.join(s_rc, "meta.json"), os.path.join(rc, f"{v:03d}_source_meta.json"))
            link(os.path.join(s_dp, f"{I}_f000_meta.json"), os.path.join(dp, f"{G}_f{v:03d}_meta.json"))
            for part in ("object", "hand"):
                link(os.path.join(s_dp, "sdfs", f"{I}_f000__{part}.npy"),
                     os.path.join(dp, "sdfs", f"{G}_f{v:03d}__{part}.npy"))
                link(os.path.join(s_dp, "idxs", f"{I}_f000__{part}.npy"),
                     os.path.join(dp, "idxs", f"{G}_f{v:03d}__{part}.npy"))
                ls = os.path.join(lat_src, f"{I}_0__{part}.npz")
                if not os.path.exists(ls):
                    ok_lat = False
                link(ls, os.path.join(lat_dst, f"{G}_{v}__{part}.npz"))
            for c in ("contact_coords", "dist_to_contact"):
                link(os.path.join(s_dp, "contacts", f"{I}_f000_{c}.npy"),
                     os.path.join(dp, "contacts", f"{G}_f{v:03d}_{c}.npy"))
            frames.append({"file_path": f"{v:03d}.png", "camera": cams[v], "source_instance": I})
        with open(os.path.join(rc, "transforms.json"), "w") as f:
            json.dump({"frames": frames, "note": "real captures; no camera matrices (DexYCB extrinsics "
                       "not in the export) - warps come from hand registration"}, f, indent=1)
        n_lat += ok_lat
        n_missing_render += (not ok_render)
        rows.append({"sha256": G, "aesthetic_score": 10.0, "rendered": True, "voxelized": True,
                     f"ss_latent_{LATENT}": ok_lat, "cond_rendered": ok_render, "local path": ""})
        manifest["groups"][G] = {"key": k, "instances": insts}
    with open(os.path.join(args.out, "metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(manifest, open(os.path.join(args.out, "groups_manifest.json"), "w"), indent=1)
    print(f"{len(rows)} group instances -> {args.out}; with latents: {n_lat}; "
          f"missing renders: {n_missing_render}")


if __name__ == "__main__":
    main()
