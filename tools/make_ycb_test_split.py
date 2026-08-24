"""
Carve a YCB-objects-only subset out of the existing held-out test splits
(datasets_split/{Leap_Hand,Hands}_test), symlinking payload dirs back to the
source the same way tools/make_heldout_split.py does -- no data is copied.

Detection is by instance-name prefix, matching the YCB(-Video) numeric naming
convention (e.g. "005_tomato_soup_can", "custom_objects_002_master_chef_can",
"063-a_marbles"):

    Hands_test        -- ALL 32 rows are YCB objects (the "Hands" dataset is the
                          YCB mesh set rigged with different hand types).
    Leap_Hand_test     -- 30/319 rows carry YCB numeric names (a mix of the base
                          YCB set and the extended YCB-Video objects).
    Hands_Google_test -- no genuine YCB objects; the few numeric-looking matches
                          are false positives from product names (e.g. "Nescafe
                          ... 8_08_oz_23_g ..."), so this root is excluded.

Writes datasets_split/{Leap_Hand,Hands}_test_ycb with a filtered metadata.csv.
"""
import argparse
import os
import re

import pandas as pd

PAYLOAD_DIRS = ["renders_cond", "data_pose_norm", "ss_latents_sdf_pose"]

# NNN_name or NNN-x_name (a/b/c/... variant), optionally prefixed by
# "custom_objects_", anchored so it must be the start of the object name.
YCB_RE = re.compile(r"^(?:custom_objects_)?[0-9]{3}(-[a-z])?_[a-z]")


def link_payload(src_root: str, dst_root: str) -> None:
    os.makedirs(dst_root, exist_ok=True)
    for d in PAYLOAD_DIRS:
        src = os.path.join(src_root, d)
        dst = os.path.join(dst_root, d)
        if not os.path.isdir(src):
            print(f"  WARNING: {src} does not exist, skipping the link")
            continue
        if os.path.islink(dst):
            os.remove(dst)
        if not os.path.exists(dst):
            os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_root", type=str, default="datasets_split")
    ap.add_argument("--sources", type=str, default="Leap_Hand_test,Hands_test",
                    help="Existing *_test split names to filter (comma-separated)")
    args = ap.parse_args()

    split_root = os.path.abspath(args.split_root)
    out_names = []
    for name in args.sources.split(","):
        name = name.strip()
        src_root = os.path.join(split_root, name)
        meta_path = os.path.join(src_root, "metadata.csv")
        assert os.path.exists(meta_path), f"missing {meta_path}"

        df = pd.read_csv(meta_path)
        is_ycb = df["sha256"].astype(str).str.match(YCB_RE)
        df_ycb = df[is_ycb].reset_index(drop=True)

        out_name = f"{name}_ycb"
        out_root = os.path.join(split_root, out_name)
        os.makedirs(out_root, exist_ok=True)
        df_ycb.to_csv(os.path.join(out_root, "metadata.csv"), index=False)
        link_payload(src_root, out_root)

        print(f"{name}: {len(df_ycb)}/{len(df)} rows are YCB -> {out_root}")
        if len(df_ycb) > 0:
            out_names.append(out_name)

    print("\nCombined --data_dir for tools that take comma-separated roots:")
    print(",".join(os.path.join(split_root, n) for n in out_names))


if __name__ == "__main__":
    main()
