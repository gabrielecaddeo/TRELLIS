import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines import TrellisImageTo3DPipelineConditioned
from trellis.utils import render_utils, postprocessing_utils
import torch
import numpy as np
from sys import argv
import os
import gc
from skimage.measure import marching_cubes
import trimesh
# string_name = str(argv[1]) if len(argv) > 1 else "default_instance"
# view_id = int(argv[2]) if len(argv) >= 2 else 0
print(f"Instance name")
# print(f"Running optimization for instance '{string_name}', view {view_id}")
# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipelineConditioned.from_pretrained("/home/user/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96")
pipeline.cuda()

def save_mesh(sdf, filename: str='ciao'):

    dx = 2.0/64.0
    origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)

    # Your sdf is indexed as sdf[i,j,k] with i->x, j->y, k->z.
    # skimage expects volume[z,y,x], so transpose (x,y,z) -> (z,y,x).
    vol_zyx = np.transpose(sdf, (2, 1, 0))

    # Extract 0-level set in ZYX index space (spacing is isotropic here)
    verts_zyx, faces, normals_zyx, _ = marching_cubes(vol_zyx, level=0.0, spacing=(dx, dx, dx))

    # Convert verts back to XYZ coordinate order
    verts_xyz = verts_zyx[:, [2, 1, 0]]
    normals_xyz = normals_zyx[:, [2, 1, 0]]

    # IMPORTANT: your SDF samples live at voxel *centers*.
    # marching_cubes assumes grid points at integer coordinates -> shift by half voxel.
    verts_xyz = verts_xyz + origin + 0.5 * dx

    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, vertex_normals=normals_xyz, process=False)

    mesh.export(filename)

seed = 42
torch.manual_seed(seed)
# Load an image
# Load constraints
# target = torch.tensor(np.load('/home/user/TRELLIS/bunny.npy')).to('cuda')

# --- Choose Guidance Strategy ---
# 1: D-Flow w/ Continuous Adjoint (fast, memory-efficient, adaptive solver)
# 2: D-Flow w/ Discrete Adjoint (stable, fixed-step solver)
# 3: OC-Flow w/ Continuous Adjoint (optimizes control function, most advanced)


# --- Run Pipeline ---
# outputs = pipeline.run_optimization(
#     image=image,
#     noise=noise,
#     constraint_sdf=target,
#     seed=42,
#     sparse_structure_sampler_params=params,
# )
import glob
folders = glob.glob('/home/user/TRELLIS/folders_amodal_2/*')
for folder in folders:
    string_name = os.path.basename(folder)
    for view_id in range(24):
        print("alloc", torch.cuda.memory_allocated()/1e9, "GB",
      "reserved", torch.cuda.memory_reserved()/1e9, "GB")
        if os.path.exists(f"/home/user/TRELLIS/meshes_results_marching_cubes_2/{string_name}/{view_id:02d}/sample.ply"):
            print(f"Skipping {string_name} view {view_id} as it already exists.")
            continue
        os.makedirs(f"/home/user/TRELLIS/meshes_results_marching_cubes_2/{string_name}/{view_id:02d}", exist_ok=True)
        outputs = pipeline.run_velocity(
            root='/home/user/TRELLIS/datasets/Hands',
            instance_name=string_name,
            view=view_id,
            seed=42)
        save_mesh(outputs.squeeze(0,1).cpu().numpy(), filename=f"/home/user/TRELLIS/meshes_results_marching_cubes_2/{string_name}/{view_id:02d}/sample.ply")
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        # view_id = 1
        # string_name = "custom_objects_004_sugar_box_barrett_4"

        # glb = postprocessing_utils.to_glb(
        #     outputs['gaussian'][0],
        #     outputs['mesh'][0],
        #     # Optional parameters
        #     simplify=0.95,          # Ratio of triangles to remove in the simplification process
        #     texture_size=1024,      # Size of the texture used for the GLB
        # )
        # glb.export(f"/home/user/TRELLIS/meshes_results_train_simple/{string_name}/{view_id:02d}/sample.glb")
        # del outputs, glb
        # gc.collect()
        # torch.cuda.empty_cache()
# exit(0)
# outputs = pipeline.run_velocity(
#     image=image,
#     target=target,
#     seed=1,
# )

# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes
# '''
# # Render the outputs
# video = render_utils.render_video(outputs['gaussian'][0])['color']
# imageio.mimsave("sample_gs.mp4", video, fps=30)
# video = render_utils.render_video(outputs['radiance_field'][0])['color']
# imageio.mimsave("sample_rf.mp4", video, fps=30)
# video = render_utils.render_video(outputs['mesh'][0])['normal']
# imageio.mimsave("sample_mesh.mp4", video, fps=30)

# # GLB files can be extracted from the outputs
# glb = postprocessing_utils.to_glb(
#     outputs['gaussian'][0],
#     outputs['mesh'][0],
#     # Optional parameters
#     simplify=0.95,          # Ratio of triangles to remove in the simplification process
#     texture_size=1024,      # Size of the texture used for the GLB
# )
# glb.export("sample.glb")

# # Save Gaussians as PLY files
# outputs['gaussian'][0].save_ply("sample.ply")
# '''


# y0 = variable(grad=True)
# optimizer = torch.optim.lbfgs([y0], lr=0.01)
# for i in range(100):
#     y1 = odeint_adjoint(func, y0, t, method='euler', options={'step_size': 0.1})
#     z = decoder(y1)
#     loss = loss_fn(z, target)
#     optimizer.step()



# def func(t, y):
#     """
#     Example function for ODE solver.
#     """
#     return flow(t, y)
