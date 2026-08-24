"""
Qualitative viewer for the a/b guidance evaluation.

Reads the SDF blobs saved by tools/ab_eval_guidance.py (--save_sdf_samples) and
writes one standalone HTML page per sample plus an index. Each page shows:

  - the input conditioning image
  - one interactive 3D panel per method (unguided / guided_asis / guided_v2 /
    oc_flow / gt): the OBJECT zero-level surface (blue) and the HAND zero-level
    surface (semi-transparent gray) from the same 64^3 grid, with the annotated
    contact voxels as red dots. Rotate/zoom each panel independently.

CPU-only; safe on the login node (no model, no GPU).

    python tools/visualize_ab_sdfs.py                       # defaults
    python tools/visualize_ab_sdfs.py --max_samples 16
"""
import os
import sys
import glob
import base64
import argparse
import io

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_metrics import sdf_to_mesh  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from PIL import Image  # noqa: E402

ARM_ORDER = ["unguided", "guided_asis", "guided_v2", "oc_flow", "gt"]


def mesh_trace(mesh, color, opacity, name):
    v, f = mesh.vertices, mesh.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=opacity, name=name, showlegend=False,
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.15),
    )


def contact_trace(touch_grid):
    idx = np.argwhere(touch_grid > 0)  # [N,3] voxel indices
    if len(idx) == 0:
        return None
    xyz = -1.0 + (idx + 0.5) * (2.0 / touch_grid.shape[0])
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(size=2.2, color="red"), name="contacts", showlegend=False,
    )


def image_to_b64(img_chw_uint8):
    arr = np.transpose(img_chw_uint8, (1, 2, 0))
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def sample_page(arms_sdf, sdf_hand, touch, img_b64, title):
    arms = [a for a in ARM_ORDER if a in arms_sdf]
    fig = make_subplots(
        rows=1, cols=len(arms),
        specs=[[{"type": "scene"}] * len(arms)],
        subplot_titles=arms, horizontal_spacing=0.005,
    )
    hand_mesh = sdf_to_mesh(sdf_hand)
    ct = contact_trace(touch)
    for c, arm in enumerate(arms, start=1):
        obj_mesh = sdf_to_mesh(arms_sdf[arm])
        if obj_mesh is not None:
            fig.add_trace(mesh_trace(obj_mesh, "royalblue", 1.0, arm), row=1, col=c)
        if hand_mesh is not None:
            fig.add_trace(mesh_trace(hand_mesh, "lightgray", 0.35, "hand"), row=1, col=c)
        if ct is not None:
            fig.add_trace(ct, row=1, col=c)
    scene = dict(
        xaxis=dict(range=[-1, 1], visible=False),
        yaxis=dict(range=[-1, 1], visible=False),
        zaxis=dict(range=[-1, 1], visible=False),
        aspectmode="cube",
    )
    fig.update_layout(
        height=440, margin=dict(l=0, r=0, t=40, b=0),
        **{f"scene{'' if i == 1 else i}": scene for i in range(1, len(arms) + 1)},
    )
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return f"""<html><head><title>{title}</title></head>
<body style="font-family:sans-serif;margin:12px">
<h3 style="margin:4px 0">{title}</h3>
<div style="display:flex;align-items:flex-start;gap:16px">
  <div><div style="font-size:13px;color:#555">input image</div>
       <img src="data:image/png;base64,{img_b64}" width="300"/></div>
  <div style="flex:1">{plot_html}</div>
</div>
<div style="font-size:13px;color:#555;margin-top:6px">
blue = object zero level &nbsp;|&nbsp; gray = hand zero level &nbsp;|&nbsp;
red dots = annotated contact voxels. Drag to rotate, scroll to zoom (per panel).</div>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdf_dir", type=str,
                    default="outputs/diagnostics/ab_guidance_4arm_sdfs")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    out_dir = args.out_dir or args.sdf_dir + "_html"
    os.makedirs(out_dir, exist_ok=True)

    blobs = sorted(glob.glob(os.path.join(args.sdf_dir, "samples_*.pt")))
    assert blobs, f"no samples_*.pt under {args.sdf_dir}"

    index_rows, n_out = [], 0
    for blob_path in blobs:
        blob = torch.load(blob_path, map_location="cpu", weights_only=True)
        arm_keys = [k for k in blob if k.startswith("sdf_")
                    and k not in ("sdf_hand", "sdf_gt")]
        k = blob["sdf_hand"].shape[0]
        for b in range(k):
            if args.max_samples > 0 and n_out >= args.max_samples:
                break
            arms_sdf = {ak.replace("sdf_", ""): blob[ak][b, 0].numpy() for ak in arm_keys}
            arms_sdf["gt"] = blob["sdf_gt"][b, 0].numpy()
            title = f"sample {n_out:03d}"
            html = sample_page(
                arms_sdf,
                blob["sdf_hand"][b, 0].numpy(),
                blob["touch"][b, 0].numpy(),
                image_to_b64(blob["image"][b].numpy()) if "image" in blob else "",
                title,
            )
            fname = f"sample_{n_out:03d}.html"
            with open(os.path.join(out_dir, fname), "w") as f:
                f.write(html)
            index_rows.append(f'<a href="{fname}">{title}</a>')
            n_out += 1
        if args.max_samples > 0 and n_out >= args.max_samples:
            break

    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write("<html><body style='font-family:sans-serif'><h3>a/b guidance samples</h3>"
                + "<br/>\n".join(index_rows) + "</body></html>")
    print(f"wrote {n_out} sample pages + index.html under {out_dir}")


if __name__ == "__main__":
    main()
