"""Dex-YCB benchmark inference for teacher_v2 (same procedure as inference_dex.py,
different pipeline dir + output dir). One mesh per instance, view 0, seed 42.
Resumable: instances with an existing sample.ply are skipped.
"""
import os

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

import gc
import glob
import traceback

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes

from trellis.pipelines import TrellisImageTo3DPipelineConditioned

REPO = "/projects/gcaddeo/inference/TRELLIS"
# DEX_PIPELINE_DIR overrides the pipeline dir (e.g. student_a_stage2 from
# convert_teacher_v2.py --pipeline_dir). Unset -> teacher, identical to the
# parity-tested runs.
PIPELINE_DIR = os.environ.get("DEX_PIPELINE_DIR") or f"{REPO}/teacher_v2_stage2"
DATA_ROOT = f"{REPO}/dex-dataset-total-total"
OUT_ROOT = os.environ.get("DEX_OUT_ROOT") or f"{REPO}/meshes_results_marching_cubes_teacher_v2"
# DEX_STEPS overrides the sparse-structure sampler's step count (pipeline json
# default: 25). Unset -> {} -> behavior identical to the parity-tested runs.
SAMPLER_PARAMS = (
    {"steps": int(os.environ["DEX_STEPS"])} if os.environ.get("DEX_STEPS") else {}
)

pipeline = TrellisImageTo3DPipelineConditioned.from_pretrained(PIPELINE_DIR)
pipeline.cuda()


def save_mesh(sdf, filename):
    # Same meshing convention as inference_dex.py: level 0.0, ZYX transpose,
    # half-voxel shift (samples at voxel centers).
    dx = 2.0 / 64.0
    origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    vol_zyx = np.transpose(sdf, (2, 1, 0))
    verts_zyx, faces, normals_zyx, _ = marching_cubes(vol_zyx, level=0.0, spacing=(dx, dx, dx))
    verts_xyz = verts_zyx[:, [2, 1, 0]] + origin + 0.5 * dx
    normals_xyz = normals_zyx[:, [2, 1, 0]]
    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, vertex_normals=normals_xyz, process=False)
    mesh.export(filename)


seed = 42
torch.manual_seed(seed)

folders = sorted(glob.glob(f"{DATA_ROOT}/data_pose_norm/*"))
print(f"{len(folders)} instances")

for folder in folders:
    string_name = os.path.basename(folder)
    for view_id in range(1):
        out_dir = f"{OUT_ROOT}/{string_name}/{view_id:02d}"
        if os.path.exists(f"{out_dir}/sample.ply"):
            print(f"Skipping {string_name} view {view_id} (exists)")
            continue
        os.makedirs(out_dir, exist_ok=True)
        try:
            outputs = pipeline.run_velocity(
                root=DATA_ROOT,
                instance_name=string_name,
                view=view_id,
                seed=seed,
                sparse_structure_sampler_params=SAMPLER_PARAMS,
            )
        except Exception:
            print(f"problem: {string_name}")
            traceback.print_exc()
            continue
        save_mesh(outputs.squeeze(0, 1).cpu().numpy(), filename=f"{out_dir}/sample.ply")
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
