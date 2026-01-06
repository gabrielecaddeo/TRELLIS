import argparse
import shutil
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(
        description="Repack custom_objects_* folders into cleaner structure with model.obj/model.obj.mtl."
    )
    ap.add_argument("--in_root", required=True, help="Root directory containing custom_objects_* folders.")
    ap.add_argument("--out_root", required=True, help="Root directory where new folders will be created.")
    ap.add_argument("--obj_name", default="object.obj", help="Name of the original OBJ file (default: object.obj).")
    ap.add_argument("--mtl_name", default="material.mtl", help="Name of the original MTL file (default: material.mtl).")
    ap.add_argument("--png_glob", default="material_0.png",
                    help="PNG filename or glob inside each folder (default: material_0.png).")
    ap.add_argument("--prefix", default="custom_objects_",
                    help="Prefix to strip from folder names (default: custom_objects_).")
    ap.add_argument("--verbose", action="store_true", help="Print extra info.")
    return ap.parse_args()

def compute_new_folder_name(old_name: str, prefix: str) -> str:
    """
    From 'custom_objects_031_spoon_shadowhand' -> '031_spoon'
    From 'custom_objects_032_spoon_brush_allegro' -> '032_spoon_brush'
    i.e. strip prefix and drop last '_' segment.
    """
    if old_name.startswith(prefix):
        stripped = old_name[len(prefix):]
    else:
        stripped = old_name

    parts = stripped.split("_")
    if len(parts) <= 1:
        # nothing to drop, just return stripped
        return stripped
    return "_".join(parts[:-1])

def rewrite_obj_mtllib(src_obj: Path, dst_obj: Path, new_mtl_name: str):
    """
    Copy OBJ and overwrite any 'mtllib' line to point to new_mtl_name.
    """
    lines = src_obj.read_text(errors="ignore").splitlines()
    out_lines = []
    for line in lines:
        if line.lower().startswith("mtllib "):
            out_lines.append(f"mtllib {new_mtl_name}")
        else:
            out_lines.append(line)
    dst_obj.write_text("\n".join(out_lines) + "\n")

def rewrite_mtl_texture_paths(src_mtl: Path, dst_mtl: Path):
    """
    Copy MTL and make sure texture paths are just basenames (no paths).
    Does NOT rename the PNG itself – just trims any leading path.
    """
    keys = ("map_Ka", "map_Kd", "map_Ks", "map_Ns", "map_d",
            "map_Bump", "bump", "disp", "decal", "norm", "map_Pr")

    lines = src_mtl.read_text(errors="ignore").splitlines()
    out_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue

        if not any(stripped.startswith(k + " ") for k in keys):
            out_lines.append(line)
            continue

        parts = stripped.split()
        if len(parts) < 2:
            out_lines.append(line)
            continue

        prefix = " ".join(parts[:-1])
        tex_ref = parts[-1]
        tex_name = Path(tex_ref).name  # only keep filename
        out_lines.append(f"{prefix} {tex_name}")

    dst_mtl.write_text("\n".join(out_lines) + "\n")

def main():
    args = parse_args()
    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not in_root.is_dir():
        print(f"[error] in_root is not a directory: {in_root}")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    folders = sorted([p for p in in_root.iterdir() if p.is_dir()])
    processed = 0

    for sub in folders:
        new_name = compute_new_folder_name(sub.name, args.prefix)
        if args.verbose:
            print(f"[info] {sub.name} -> {new_name}")

        src_obj = sub / args.obj_name
        src_mtl = sub / args.mtl_name

        # Find PNG (default: material_0.png)
        png_candidates = list(sub.glob(args.png_glob))
        if not png_candidates:
            if args.verbose:
                print(f"[skip] {sub}: PNG '{args.png_glob}' not found.")
            continue
        src_png = png_candidates[0]

        if not src_obj.is_file() or not src_mtl.is_file():
            if args.verbose:
                print(f"[skip] {sub}: missing {args.obj_name} or {args.mtl_name}.")
            continue

        dst_folder = out_root / new_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        dst_obj = dst_folder / "model.obj"
        dst_mtl = dst_folder / "model.obj.mtl"
        dst_png = dst_folder / src_png.name  # keep same png name

        # Rewrite OBJ to use new MTL name
        rewrite_obj_mtllib(src_obj, dst_obj, dst_mtl.name)

        # Rewrite MTL texture paths to be local
        rewrite_mtl_texture_paths(src_mtl, dst_mtl)

        # Copy PNG
        shutil.copy2(src_png, dst_png)

        processed += 1
        if args.verbose:
            print(f"[ok] wrote {dst_obj}, {dst_mtl}, {dst_png}")

    print(f"[done] processed {processed} folders")

if __name__ == "__main__":
    main()
