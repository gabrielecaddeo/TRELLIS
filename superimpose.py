import os
import torch
import trimesh
import numpy as np
import pyvista as pv
import mesh2sdf
import utils3d
import glob
import gc
import pathlib
import networkx as nx
import argparse 
import scipy
# Compute bounding box efficiency of a mesh
def compute_bounding_box_efficiency(mesh, pitch=0.01):
    # Get bounding box volume
    bounds = mesh.bounds
    bbox_volume = np.prod(bounds[1] - bounds[0])
    
    # Estimate actual mesh volume using voxelized occupancy
    voxelized = mesh.voxelized(pitch=pitch)
    filled= voxelized.fill()
    mesh_volume_est = filled.points.shape[0] * pitch**3

    # Efficiency: how much of the bounding box is "occupied"
    efficiency = mesh_volume_est / bbox_volume
    return efficiency

# Check if the mesh is dense enough based on bounding box efficiency
def is_dense_enough(mesh, min_efficiency=0.15, pitch=0.01):
    efficiency = compute_bounding_box_efficiency(mesh, pitch)
    print(efficiency)
    return efficiency >= min_efficiency


def fix_sdf(buggy_sdf, voxel_occupancy_grid):
    """
    Corrects the signs of a Signed Distance Function (SDF) using a ground-truth
    voxel occupancy grid.

    This function preserves the distance magnitude from the original SDF but enforces
    the sign (inside/outside) based on the occupancy grid.

    Args:
        buggy_sdf (np.ndarray): A 3D numpy array representing the SDF with potentially
                                incorrect signs.
        voxel_occupancy_grid (np.ndarray): A 3D boolean or integer numpy array where
                                           True or 1 indicates a point is inside the
                                           object.

    Returns:
        np.ndarray: A new 3D numpy array with the corrected SDF values.
    """
    print("  Correcting SDF signs using ground-truth voxel occupancy...")
    
    # 1. Compute the Unsigned Distance Field (UDF)
    unsigned_distance = np.abs(buggy_sdf)
    
    # 2. Create the sign multiplier field.
    # We want -1 for inside (True/1) and +1 for outside (False/0).
    # The formula 1 - 2*V accomplishes this:
    # If V=1 (inside), M = 1 - 2*1 = -1
    # If V=0 (outside), M = 1 - 2*0 = +1
    sign_multiplier = 1 - 2 * voxel_occupancy_grid.astype(np.float32)
    
    # 3. Construct the corrected SDF
    corrected_sdf = sign_multiplier * unsigned_distance
    
    return corrected_sdf

def is_simple_mesh(mesh, max_faces=50000):
    # Criterion 1: Single component
    # components = trimesh.graph.connected_components(mesh, return_faces=False)
    if not is_dense_enough(mesh):
        print("Mesh has multiple components.")
        return False
    print('1 component')
    # Criterion 2: Watertight
    # if not mesh.is_watertight:
    #     print("Mesh is not watertight.")
    #     return False
    # print('watertight')
    # # Criterion 3: Reasonable size
    # if len(mesh.faces) > max_faces:
    #     print(f"Mesh has too many faces: {len(mesh.faces)} > {max_faces}.")
    #     return False
    # print(f"Mesh has {len(mesh.faces)} faces, which is within the limit of {max_faces}.")
    # # Optional Criterion 4: No non-manifold edges (optional)
    if not mesh.is_winding_consistent:
        print("Mesh has non-manifold edges.")
        return False
    print("Mesh is simple: single component, watertight, and within face limit.")
    return True

def fix_sdf(buggy_sdf, voxel_occupancy_grid):
    """
    Corrects the signs of a Signed Distance Function (SDF) using a ground-truth
    voxel occupancy grid.

    This function preserves the distance magnitude from the original SDF but enforces
    the sign (inside/outside) based on the occupancy grid.

    Args:
        buggy_sdf (np.ndarray): A 3D numpy array representing the SDF with potentially
                                incorrect signs.
        voxel_occupancy_grid (np.ndarray): A 3D boolean or integer numpy array where
                                           True or 1 indicates a point is inside the
                                           object.

    Returns:
        np.ndarray: A new 3D numpy array with the corrected SDF values.
    """
    print("  Correcting SDF signs using ground-truth voxel occupancy...")
    
    # 1. Compute the Unsigned Distance Field (UDF)
    unsigned_distance = np.abs(buggy_sdf)
    
    # 2. Create the sign multiplier field.
    # We want -1 for inside (True/1) and +1 for outside (False/0).
    # The formula 1 - 2*V accomplishes this:
    # If V=1 (inside), M = 1 - 2*1 = -1
    # If V=0 (outside), M = 1 - 2*0 = +1
    sign_multiplier = 1 - 2 * voxel_occupancy_grid.astype(np.float32)
    
    # 3. Construct the corrected SDF
    corrected_sdf = sign_multiplier * unsigned_distance
    
    return corrected_sdf


def check_simple_meshes():
    simple_meshes = []
    #=== Check each instance for simple mesh criteria ===
    for path in instances:
        instance = path.rsplit('/')[-1]
        print(instance)
        ply_mesh_path = os.path.join(root, "destination", instance, "mesh.ply")
        try:
            mesh = trimesh.load(ply_mesh_path, force='mesh')
            scene = trimesh.Scene([mesh])
            scene.show()
            
            if is_simple_mesh(mesh):
                simple_meshes.append(path)
            else:
                print(f"Mesh {instance} is not simple: too complex or not watertight.")
            

            del mesh
            del scene
            gc.collect()

        except Exception as e:
            print(f"Failed to load {path}: {e}")

    print(f"\nFound {len(simple_meshes)} simple meshes out of {len(simple_meshes)}")
    return simple_meshes

def count_objects_in_mesh(mesh):
    """
    Count distinct objects in a mesh by splitting it into connected components.
    """
    # Process the mesh to ensure it's clean
    mesh.process(validate=True)
    
    # Split the mesh into connected components
    fragments = mesh.split(only_watertight=False)
    
    # Filter out small fragments based on face count
    min_face_count = int(len(mesh.faces) / 50)  # Adjust this threshold as needed
    significant_parts = [part for part in fragments if len(part.faces) > min_face_count]
    
    return len(significant_parts), significant_parts



def pre_adjustment_sdf(instance_name, root):

    # Extract the instance name (the UUID) from the voxel filename.
    # instance_name = pathlib.Path(voxel_ply_path).stem
    # instance_name = '00ad7d701436cd5f4a7f6a38b7c95b7d1f41a7a0fbf60a9674d26747209a2f86' #pathlib.Path(voxel_ply_path).stem
    voxel_ply_path = os.path.join(root,'voxels',instance_name+'.ply')
    print(f"Processing instance: {instance_name}")

    # Now, construct the path to the corresponding mesh file.
    ply_mesh_path = os.path.join(root, "destination", instance_name, "mesh.ply")

    if not os.path.exists(ply_mesh_path):
        print(f"  [Warning] Corresponding mesh file not found, skipping: {ply_mesh_path}")
        return 

    print(f"  Voxel path: {voxel_ply_path}")
    print(f"  Mesh path: {ply_mesh_path}")
    
    try:
        # === Load mesh ===
        mesh = trimesh.load(ply_mesh_path, force='mesh')
        print(f"  Mesh bounding box: {mesh.bounding_box.bounds}")
        
        vertices = mesh.vertices
        faces = mesh.faces
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale

        # === Compute SDF on 64³ grid ===
        resolution = 64
        level = 1 / resolution
        sdf, _ = mesh2sdf.compute(vertices, faces, resolution, fix=True, level=level, return_mesh=True)
        
        # === Load binary voxelization ===
        position = utils3d.io.read_ply(voxel_ply_path)[0]
        coords = ((torch.tensor(position, dtype=torch.float32) + 0.5) * resolution).long()
        coords = torch.clamp(coords, 0, resolution - 1)
        ss = torch.zeros(resolution, resolution, resolution, dtype=torch.bool)
        ss[coords[:, 0], coords[:, 1], coords[:, 2]] = True

        # === Data preparation for plotting ===
        threshold = 0.02
        
        # --- Prepare Voxel Shell Surface ---
        voxel_grid_np = ss.numpy().astype(np.uint8)
        grid = pv.ImageData()
        grid.dimensions = voxel_grid_np.shape
        grid.point_data['voxels'] = voxel_grid_np.flatten(order='F')
        voxel_shell_surface = grid.contour([0.5], scalars='voxels')
        
        # --- Prepare SDF Grid for Contouring (used by plot 2) ---
        sdf_pyvista_grid = pv.ImageData()
        sdf_pyvista_grid.dimensions = sdf.shape
        sdf_pyvista_grid.point_data['sdf'] = sdf.flatten(order='F')

        # === Static Plot (for reference) ===
        print("\n--- Displaying Static Plot ---")
        static_plotter = pv.Plotter()
        static_plotter.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.5)
        
        # Using the robust NumPy method for the static plot of the volumetric region
        sdf_near_zero_mask = np.abs(sdf) <= threshold
        sdf_coords = np.argwhere(sdf_near_zero_mask)
        sdf_point_cloud = pv.PolyData(sdf_coords)

        static_plotter.add_points(sdf_point_cloud, color="purple", opacity=0.8, point_size=7, render_points_as_spheres=True, label=f"SDF |x| <= {threshold}")
        static_plotter.add_axes()
        static_plotter.show_grid()
        static_plotter.add_legend()
        static_plotter.show(title=f"Static SDF vs Voxel Shell for {instance_name}", window_size=[1000, 800])
        
        # === PLOT 1: Interactive Volumetric |SDF| Region ===
        # Visualizing the volume of points where |sdf| <= x + threshold
        print("\n--- Displaying Interactive Plot 1: Volumetric |SDF| Region ---")
        p1 = pv.Plotter(window_size=[1000, 800])
        p1.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.3)
        p1.add_text("Voxel Shell + Interactive |SDF| Region", font_size=12)

        def callback_abs_sdf_volume(x_value):
            # This callback now updates a point cloud based on the slider value.
            p1.remove_actor("sdf_point_cloud", render=False) # Remove the old point cloud
            
            # The dynamic threshold is C = x + threshold
            dynamic_threshold = x_value + threshold
            
            if dynamic_threshold >= 0:
                # Use NumPy to find all points where |sdf| <= dynamic_threshold
                mask = np.abs(sdf) <= dynamic_threshold
                coords = np.argwhere(mask)
                
                # Only try to plot if we found some points
                if coords.size > 0:
                    point_cloud = pv.PolyData(coords)
                    p1.add_points(
                        point_cloud, 
                        name="sdf_point_cloud", 
                        color="purple", 
                        opacity=0.8, 
                        point_size=7, 
                        render_points_as_spheres=True
                    )

        # Add the slider widget to the plotter
        p1.add_slider_widget(
            callback=callback_abs_sdf_volume,
            rng=[-0.5, 0.5],
            value=0.0,
            title="x, where |sdf| <= x + threshold",
            style='modern'
        )
        p1.add_axes()
        p1.show_grid()
        p1.show(title=f"Interactive Volumetric |SDF| for {instance_name}")


        # === PLOT 2: Interactive SDF Level Set ===
        # Visualizing the level set where sdf = threshold2
        print("\n--- Displaying Interactive Plot 2: SDF Level Set ---")
        p2 = pv.Plotter(window_size=[1000, 800])
        p2.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.3)
        p2.add_text("Voxel Shell + Interactive SDF Level", font_size=12)

        def callback_single_sdf(threshold2):
            # This callback updates the single isosurface for the original sdf.
            p2.remove_actor("sdf_iso_single", render=False)
            iso = sdf_pyvista_grid.contour([threshold2], scalars='sdf')
            p2.add_mesh(iso, name="sdf_iso_single", color="purple")

        # Add the slider widget to the plotter
        p2.add_slider_widget(
            callback=callback_single_sdf,
            rng=[-0.5, 0.5],
            value=0.0,
            title="threshold2, where sdf <= threshold2",
            style='modern'
        )
        p2.add_axes()
        p2.show_grid()
        p2.show(title=f"Interactive SDF for {instance_name}")

    except Exception as e:
        print(f"  [ERROR] Failed to process {instance_name}: {e}")


import numpy as np
import scipy.ndimage

def shift_sdf_field(sdf: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Shift a 3D SDF field by a given amount (in voxel grid units, e.g. [dx, dy, dz]).
    
    Args:
        sdf: [D, H, W] SDF grid
        shift: [3] array with (dx, dy, dz) in voxel units
    
    Returns:
        Shifted SDF, same shape
    """
    assert sdf.ndim == 3, "SDF must be a 3D array"
    assert shift.shape == (3,), "Shift must be a 3-element array"

    # Create grid of coordinates
    dz, dy, dx = np.meshgrid(
        np.arange(sdf.shape[0]),
        np.arange(sdf.shape[1]),
        np.arange(sdf.shape[2]),
        indexing='ij'
    )  # each of shape [D, H, W]

    # Stack into a 3xDHW array of coordinates
    coords = np.stack([dz - shift[0], dy - shift[1], dx - shift[2]], axis=0).astype(np.float32)

    # Interpolate
    shifted_sdf = scipy.ndimage.map_coordinates(sdf, coords, order=1, mode='nearest')

    return shifted_sdf


def post_adjustment_sdf(instance_name, root):
    # Extract the instance name (the UUID) from the voxel filename.
    # instance_name = pathlib.Path(voxel_ply_path).stem
    # instance_name = '00ad7d701436cd5f4a7f6a38b7c95b7d1f41a7a0fbf60a9674d26747209a2f86' #pathlib.Path(voxel_ply_path).stem
    voxel_ply_path = os.path.join(root,'voxels',instance_name+'.ply')
    print(f"Processing instance: {instance_name}")

    # Now, construct the path to the corresponding mesh file.
    ply_mesh_path = os.path.join(root, "destination", instance_name, "mesh.ply")

    if not os.path.exists(ply_mesh_path):
        print(f"  [Warning] Corresponding mesh file not found, skipping: {ply_mesh_path}")
        return

    print(f"  Voxel path: {voxel_ply_path}")
    print(f"  Mesh path: {ply_mesh_path}")
    
    try:
        # === Load mesh ===
        mesh = trimesh.load(ply_mesh_path, force='mesh')
        print(f"  Mesh bounding box: {mesh.bounding_box.bounds}")
        
        vertices = mesh.vertices
        faces = mesh.faces
        # bbmin = vertices.min(0)
        # bbmax = vertices.max(0)
        # center = (bbmin + bbmax) * 0.5
        # scale = 2.0 / (bbmax - bbmin).max()
        # vertices = (vertices - center) * scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6) * 2


        # === Compute SDF on 64³ grid ===
        resolution = 64
        level = 1 / resolution
        sdf_buggy, meshrec = mesh2sdf.compute(vertices, faces, resolution, fix=False, level=level, return_mesh=True)
        
        # === Load binary voxelization ===
        position = utils3d.io.read_ply(voxel_ply_path)[0]
        coords = ((torch.tensor(position, dtype=torch.float32) + 0.5) * resolution).long()
        # coords = torch.round((torch.tensor(position, dtype=torch.float32) + 0.5) * resolution - 1e-6).to(torch.int64)

        coords = torch.clamp(coords, 0, resolution - 1)
        ss = torch.zeros(resolution, resolution, resolution, dtype=torch.bool)
        ss[coords[:, 0], coords[:, 1], coords[:, 2]] = True


        # === CORRECT THE SDF SIGNS ===
        sdf = fix_sdf(sdf_buggy, ss.numpy())


 
        # sdf = shift(sdf_tensor, shift_in_grid.numpy())
        # === Data preparation for plotting ===
        threshold = 0.01
        
        # --- Prepare Voxel Shell Surface ---
        voxel_grid_np = ss.numpy().astype(np.uint8)
        grid = pv.ImageData()
        grid.dimensions = voxel_grid_np.shape
        grid.point_data['voxels'] = voxel_grid_np.flatten(order='F')
        voxel_shell_surface = grid.contour([0.5], scalars='voxels')
        
        # --- Prepare SDF Grid for Contouring (used by plot 2) ---
        sdf_pyvista_grid = pv.ImageData()
        sdf_pyvista_grid.dimensions = sdf.shape
        sdf_pyvista_grid.point_data['sdf'] = sdf.flatten(order='F')

        # === Static Plot (for reference) ===
        print("\n--- Displaying Static Plot ---")
        static_plotter = pv.Plotter()
        static_plotter.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.5)
        
        sdf_near_zero_mask = np.abs(sdf) <= threshold
        sdf_coords = np.argwhere(sdf_near_zero_mask)
        sdf_point_cloud = pv.PolyData(sdf_coords)

        static_plotter.add_points(sdf_point_cloud, color="purple", opacity=0.8, point_size=7, render_points_as_spheres=True, label=f"SDF |x| <= {threshold}")
        static_plotter.add_axes()
        static_plotter.show_grid()
        static_plotter.add_legend()
        static_plotter.show(title=f"Static SDF vs Voxel Shell for {instance_name}", window_size=[1000, 800])
        
        # === PLOT 1: Interactive Volumetric |SDF| Region ===
        print("\n--- Displaying Interactive Plot 1: Volumetric |SDF| Region ---")
        p1 = pv.Plotter(window_size=[1000, 800])
        p1.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.3)
        p1.add_text("Voxel Shell + Interactive |SDF| Region", font_size=12)

        def callback_abs_sdf_volume(x_value):
            p1.remove_actor("sdf_point_cloud", render=False) 
            dynamic_threshold = x_value + threshold
            
            if dynamic_threshold >= 0:
                mask = np.abs(sdf) <= dynamic_threshold
                coords = np.argwhere(mask)
                if coords.size > 0:
                    point_cloud = pv.PolyData(coords)
                    p1.add_points(point_cloud, name="sdf_point_cloud", color="purple", opacity=0.8, point_size=7, render_points_as_spheres=True)

        p1.add_slider_widget(callback=callback_abs_sdf_volume, rng=[-0.5, 0.5], value=0.0, title="x, where |sdf| <= x + threshold", style='modern')
        p1.add_axes()
        p1.show_grid()
        p1.show(title=f"Interactive Volumetric |SDF| for {instance_name}")

        # === PLOT 2: Interactive SDF Level Set ===
        print("\n--- Displaying Interactive Plot 2: SDF Level Set ---")
        p2 = pv.Plotter(window_size=[1000, 800])
        p2.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.3)
        p2.add_text("Voxel Shell + Interactive SDF Level", font_size=12)

        def callback_single_sdf(threshold2):
            p2.remove_actor("sdf_iso_single", render=False)
            iso = sdf_pyvista_grid.contour([threshold2], scalars='sdf')
            p2.add_mesh(iso, name="sdf_iso_single", color="purple")

        p2.add_slider_widget(callback=callback_single_sdf, rng=[-1.0, 1.0], value=0.0, title="threshold2, where sdf <= threshold2", style='modern')
        p2.add_axes()
        p2.show_grid()
        p2.show(title=f"Interactive SDF for {instance_name}")
        
       # === PLOT 3: Interactive SDF Slice with Binned Contours ===
        print("\n--- Displaying Plot 3: Interactive SDF Slice with Binned Contours ---")
        p3 = pv.Plotter(window_size=[1000, 800])
        p3.set_background('darkgrey') # Set background to see empty slices clearly
        
        # --- Generate Coherent Logarithmic Contour Levels ---
        sdf_min, sdf_max = sdf.min(), sdf.max()
        num_bins = 10
        
        # To create a log scale over a range that includes negative numbers,
        # we map the SDF range to a positive interval, log-space it, then map back.
        delta = sdf_max - sdf_min
        # Start logspace from a small fraction of the total range to avoid log(0)
        epsilon = 1e-4 * delta 
        log_spaced_increments = np.logspace(np.log10(epsilon), np.log10(delta + epsilon), num_bins + 1)
        # Shift these increments to start from sdf_min
        contour_levels = sdf_min + log_spaced_increments - epsilon
        contour_levels[0] = sdf_min # Ensure the first level is exactly the minimum
        contour_levels[-1] = sdf_max # Ensure the last level is exactly the maximum
        
        # Define a callback function for the slider
        def update_slice_y(y_pos):
            # Remove previous actors to prevent overlap
            p3.remove_actor("slice_mesh", render=False)
            p3.remove_actor("slice_edges", render=False)
            
            # Create a 2D slice along the Y-axis at the given position
            sdf_slice = sdf_pyvista_grid.slice(normal='y', origin=(sdf_pyvista_grid.center[0], y_pos, sdf_pyvista_grid.center[2]))

            # If the slice is outside the object, it might be empty.
            if sdf_slice.n_points == 0:
                return # Do nothing, leaving the dark grey background

            # Add the solid-filled regions using n_colors for a binned effect
            p3.add_mesh(
                sdf_slice,
                name="slice_mesh",
                scalars='sdf',
                cmap='viridis',
                n_colors=num_bins, # Use number of bins for discrete colors
                clim=[sdf_min, sdf_max], # Enforce the color limits
                show_scalar_bar=False, 
            )
            
            # Generate and add subtle contour lines that match the bins
            edges = sdf_slice.contour(contour_levels, scalars='sdf')
            p3.add_mesh(edges, name="slice_edges", color='white', line_width=1, opacity=0.4)

        # Manually call the update function once to populate the plotter before adding the scalar bar
        initial_y_pos = sdf_pyvista_grid.center[1]
        update_slice_y(initial_y_pos)
        
        # Now that a mesh exists, add a scalar bar that respects the binned colors
        p3.add_scalar_bar(
            title='SDF Value', 
            n_labels=5, # Approximate number of labels
            color='white'
        )
        
        # Add a slider to control the slice position along the Y-axis
        y_bounds = sdf_pyvista_grid.bounds[2:4]
        p3.add_slider_widget(
            callback=update_slice_y,
            rng=y_bounds,
            value=initial_y_pos,
            title="Y Slice Position",
            style='modern'
        )
        
        p3.add_text("Interactive SDF Slice along Y-axis", font_size=12)
        p3.view_xz() # Set camera to view the XZ plane (correct for a Y-slice)
        p3.show(title=f"Interactive SDF Slice for {instance_name}")

    except Exception as e:
        print(f"  [ERROR] Failed to process {instance_name}: {e}")



def post_adjustment_sdf_mod(instance_name, root, visualize=True, save_sdf=False, sdf_folder = None):
    # Extract the instance name (the UUID) from the voxel filename.
    # instance_name = pathlib.Path(voxel_ply_path).stem
    # instance_name = '00ad7d701436cd5f4a7f6a38b7c,95b7d1f41a7a0fbf60a9674d26747209a2f86' #pathlib.Path(voxel_ply_path).stem
    voxel_ply_path = os.path.join(root,'voxels',instance_name+'.ply')
    print(f"Processing instance: {instance_name}")

    # Now, construct the path to the corresponding mesh file.
    ply_mesh_path = os.path.join(root, 'renders', instance_name, "mesh.ply")

    if not os.path.exists(ply_mesh_path):
        print(f"  [Warning] Corresponding mesh file not found, skipping: {ply_mesh_path}")
        return

    print(f"  Voxel path: {voxel_ply_path}")
    print(f"  Mesh path: {ply_mesh_path}")
    
    try:
        # === Load mesh ===
        mesh = trimesh.load(ply_mesh_path, force='mesh')
        print(f"  Mesh bounding box: {mesh.bounding_box.bounds}")
        
        vertices = mesh.vertices
        faces = mesh.faces

        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6) * 2


        # === Compute SDF on 64³ grid ===
        resolution = 64
        level = 1 / resolution
        sdf_buggy, meshrec = mesh2sdf.compute(vertices, faces, resolution, fix=False, level=level, return_mesh=True)
        
        # === Load binary voxelization ===
        position = utils3d.io.read_ply(voxel_ply_path)[0]
        position = torch.tensor(position, dtype=torch.float32)  # [N, 3]

        # Convert to voxel index space
        index_space = (position + 0.5) * resolution

        # Compute both floor and ceil
        floor_coords = torch.floor(index_space).to(torch.int64)
        ceil_coords  = torch.ceil(index_space).to(torch.int64)

        # Combine both sets
        all_coords = torch.cat([floor_coords, ceil_coords], dim=0)

        # Clamp to grid bounds
        all_coords = torch.clamp(all_coords, 0, resolution - 1)

        # Remove duplicates
        all_coords = torch.unique(all_coords, dim=0)

        # Create voxel occupancy grid
        ss = torch.zeros(resolution, resolution, resolution, dtype=torch.bool)
        ss[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]] = True

        # === CORRECT THE SDF SIGNS ===
        sdf = fix_sdf(sdf_buggy, ss.numpy())

        if save_sdf:
            # Ensure the directory exists
            if os.path.isdir(sdf_folder):
            # Save the SDF to the specified path
                sdf_filename = os.path.join(sdf_folder, instance_name+ ".npy")
            else:
                raise ValueError("sdf_folder must be provided if save_sdf is True")
            np.save(os.path.join(sdf_filename), sdf)
            print(f"  SDF saved to: {sdf_folder}")
        if visualize:
            
            if save_sdf:
                print(f"  SDF loaded from: {sdf_folder}")
                sdf= np.load(os.path.join(sdf_folder, instance_name+'.npy'))

        
            # sdf = shift(sdf_tensor, shift_in_grid.numpy())
            # === Data preparation for plotting ===
            threshold = 0.005
            
            # --- Prepare Voxel Shell Surface ---
            voxel_grid_np = ss.numpy().astype(np.uint8)
            grid = pv.ImageData()
            grid.dimensions = voxel_grid_np.shape
            grid.point_data['voxels'] = voxel_grid_np.flatten(order='F')
            voxel_shell_surface = grid.contour([0.5], scalars='voxels')
            
            # --- Prepare SDF Grid for Contouring (used by plot 2) ---
            sdf_pyvista_grid = pv.ImageData()
            sdf_pyvista_grid.dimensions = sdf.shape
            sdf_pyvista_grid.point_data['sdf'] = sdf.flatten(order='F')

            # === Static Plot (for reference) ===
            print("\n--- Displaying Static Plot ---")
            static_plotter = pv.Plotter()
            static_plotter.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.5)
            
            sdf_near_zero_mask = np.abs(sdf) <= threshold
            sdf_coords = np.argwhere(sdf_near_zero_mask)
            sdf_point_cloud = pv.PolyData(sdf_coords)

            static_plotter.add_points(sdf_point_cloud, color="purple", opacity=0.8, point_size=7, render_points_as_spheres=True, label=f"SDF |x| <= {threshold}")
            static_plotter.add_axes()
            static_plotter.show_grid()
            static_plotter.add_legend()
            static_plotter.show(title=f"Static SDF vs Voxel Shell for {instance_name}", window_size=[1000, 800])
            
            # === PLOT 1: Interactive Volumetric |SDF| Region ===
            print("\n--- Displaying Interactive Plot 1: Volumetric |SDF| Region ---")
            p1 = pv.Plotter(window_size=[1000, 800])
            p1.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.3)
            p1.add_text("Voxel Shell + Interactive |SDF| Region", font_size=12)

            def callback_abs_sdf_volume(x_value):
                p1.remove_actor("sdf_point_cloud", render=False) 
                dynamic_threshold = x_value + threshold
                
                if dynamic_threshold >= 0:
                    mask = np.abs(sdf) <= dynamic_threshold
                    coords = np.argwhere(mask)
                    if coords.size > 0:
                        point_cloud = pv.PolyData(coords)
                        p1.add_points(point_cloud, name="sdf_point_cloud", color="purple", opacity=0.8, point_size=7, render_points_as_spheres=True)

            p1.add_slider_widget(callback=callback_abs_sdf_volume, rng=[-0.5, 0.5], value=0.0, title="x, where |sdf| <= x + threshold", style='modern')
            p1.add_axes()
            p1.show_grid()
            p1.show(title=f"Interactive Volumetric |SDF| for {instance_name}")

            # === PLOT 2: Interactive SDF Level Set ===
            print("\n--- Displaying Interactive Plot 2: SDF Level Set ---")
            p2 = pv.Plotter(window_size=[1000, 800])
            p2.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.3)
            p2.add_text("Voxel Shell + Interactive SDF Level", font_size=12)

            def callback_single_sdf(threshold2):
                p2.remove_actor("sdf_iso_single", render=False)
                iso = sdf_pyvista_grid.contour([threshold2], scalars='sdf')
                p2.add_mesh(iso, name="sdf_iso_single", color="purple")

            p2.add_slider_widget(callback=callback_single_sdf, rng=[-1.0, 1.0], value=0.0, title="threshold2, where sdf <= threshold2", style='modern')
            p2.add_axes()
            p2.show_grid()
            p2.show(title=f"Interactive SDF for {instance_name}")
            
        # === PLOT 3: Interactive SDF Slice with Binned Contours ===
            print("\n--- Displaying Plot 3: Interactive SDF Slice with Binned Contours ---")
            p3 = pv.Plotter(window_size=[1000, 800])
            p3.set_background('darkgrey') # Set background to see empty slices clearly
            
            # --- Generate Coherent Logarithmic Contour Levels ---
            sdf_min, sdf_max = sdf.min(), sdf.max()
            num_bins = 10
            
            # To create a log scale over a range that includes negative numbers,
            # we map the SDF range to a positive interval, log-space it, then map back.
            delta = sdf_max - sdf_min
            # Start logspace from a small fraction of the total range to avoid log(0)
            epsilon = 1e-4 * delta 
            log_spaced_increments = np.logspace(np.log10(epsilon), np.log10(delta + epsilon), num_bins + 1)
            # Shift these increments to start from sdf_min
            contour_levels = sdf_min + log_spaced_increments - epsilon
            contour_levels[0] = sdf_min # Ensure the first level is exactly the minimum
            contour_levels[-1] = sdf_max # Ensure the last level is exactly the maximum
            
            # Define a callback function for the slider
            def update_slice_y(y_pos):
                # Remove previous actors to prevent overlap
                p3.remove_actor("slice_mesh", render=False)
                p3.remove_actor("slice_edges", render=False)
                
                # Create a 2D slice along the Y-axis at the given position
                sdf_slice = sdf_pyvista_grid.slice(normal='y', origin=(sdf_pyvista_grid.center[0], y_pos, sdf_pyvista_grid.center[2]))

                # If the slice is outside the object, it might be empty.
                if sdf_slice.n_points == 0:
                    return # Do nothing, leaving the dark grey background

                # Add the solid-filled regions using n_colors for a binned effect
                p3.add_mesh(
                    sdf_slice,
                    name="slice_mesh",
                    scalars='sdf',
                    cmap='viridis',
                    n_colors=num_bins, # Use number of bins for discrete colors
                    clim=[sdf_min, sdf_max], # Enforce the color limits
                    show_scalar_bar=False, 
                )
                
                # Generate and add subtle contour lines that match the bins
                edges = sdf_slice.contour(contour_levels, scalars='sdf')
                p3.add_mesh(edges, name="slice_edges", color='white', line_width=1, opacity=0.4)

            # Manually call the update function once to populate the plotter before adding the scalar bar
            initial_y_pos = sdf_pyvista_grid.center[1]
            update_slice_y(initial_y_pos)
            
            # Now that a mesh exists, add a scalar bar that respects the binned colors
            p3.add_scalar_bar(
                title='SDF Value', 
                n_labels=5, # Approximate number of labels
                color='white'
            )
            
            # Add a slider to control the slice position along the Y-axis
            y_bounds = sdf_pyvista_grid.bounds[2:4]
            p3.add_slider_widget(
                callback=update_slice_y,
                rng=y_bounds,
                value=initial_y_pos,
                title="Y Slice Position",
                style='modern'
            )
            
            p3.add_text("Interactive SDF Slice along Y-axis", font_size=12)
            p3.view_xz() # Set camera to view the XZ plane (correct for a Y-slice)
            p3.show(title=f"Interactive SDF Slice for {instance_name}")

    except Exception as e:
        print(f"  [ERROR] Failed to process {instance_name}: {e}")



# for num,instance_path in enumerate(instances):
#     if num==0:
#         continue
#     instance = instance_path.rsplit('/')[-1]
#     print(instance)
#     ply_mesh_path = os.path.join(root, "destination", instance, "mesh.ply")
#     voxel_ply_path = os.path.join(root, "voxels", f"{instance}.ply")

#     # === Load mesh ===
#     # import open3d as o3d

#     # # Load original mesh with Open3D
#     # o3d_mesh = o3d.io.read_triangle_mesh(ply_mesh_path)
#     # if not o3d_mesh.has_vertex_normals():
#     #     o3d_mesh.compute_vertex_normals()

#     # # Sample points and reconstruct watertight mesh using Poisson
#     # pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=100000)
#     # watertight_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

#     # # Crop back to bounding box of original mesh
#     # bbox = o3d_mesh.get_axis_aligned_bounding_box()
#     # watertight_o3d = watertight_o3d.crop(bbox)

#     # # Convert to trimesh for mesh2sdf
#     # mesh = trimesh.Trimesh(
#     #     vertices=np.asarray(watertight_o3d.vertices),
#     #     faces=np.asarray(watertight_o3d.triangles)
#     # )
#     mesh = trimesh.load(ply_mesh_path, force='mesh')
#     print(mesh.bounding_box.bounds)
#     vertices = mesh.vertices
#     faces = mesh.faces
#     bbmin = vertices.min(0)
#     bbmax = vertices.max(0)
#     center = (bbmin + bbmax) * 0.5
#     scale = 2.0 / (bbmax - bbmin).max()  # instead of 2.0
#     vertices = (vertices - center) * scale
#     # === Compute SDF on 64³ grid ===
#     level = 1 / resolution
#     sdf, _ = mesh2sdf.compute(vertices, faces, resolution, fix=True, level=level, return_mesh=True)
#     # === Load binary voxelization ===
#     position = utils3d.io.read_ply(voxel_ply_path)[0]
#     coords = ((torch.tensor(position) + 0.5) * resolution - 0.5).long()
#     ss = torch.zeros(resolution, resolution, resolution, dtype=torch.bool)
#     ss[coords[:, 0], coords[:, 1], coords[:, 2]] = True

#     # === Find 0-level SDF values (e.g., near zero) ===
#     threshold = -0.002  # tolerance around zero
#     sdf_near_zero = sdf <= threshold
#     sdf_coords = np.argwhere(sdf_near_zero)
#     def sdf_coords_to_voxel_index_space(sdf_inds, res=64):
#         # Convert mesh2sdf index [i] to voxel grid index in [0, 64)
#         # mesh2sdf maps to [-1, 1], so scale by (res / 2) and shift
#         return ((sdf_inds + 0.5) * (res / 2) - 0.5)
    
    
#     # sdf_coords_xyz = sdf_coords[:, [2, 1, 0]]
#     # === Convert to world coordinates for visualization ===
#     def grid_to_world(inds, res=64):
#         return (inds + 0.5) / res - 0.5

#     def grid_to_world_sdf(inds, res=64):
#         # mesh2sdf maps the box to [-1, 1] → spacing = 2 / res
#         return (inds - 0.) * (2.0 / res) - 1.0

#     voxel_coords = np.argwhere(ss.numpy())
#     voxel_points = voxel_coords.astype(np.float32)
#     sdf_zero_points = sdf_coords.astype(np.float32)  #*2.0 - [resolution/2 +1, resolution/2+1, resolution/2+1] # scale to [-1, 1] range
#     # sdf_zero_points = sdf_zero_points - np.array([1, 1, 0.5]) / resolution
#     # === Create point clouds ===
#     voxel_cloud = pv.PolyData(voxel_points)
#     sdf_zero_cloud = pv.PolyData(sdf_zero_points)

#     # === Plot ===
#     plotter = pv.Plotter()
#     plotter.add_points(voxel_cloud, color="red", point_size=6, render_points_as_spheres=True, label="Voxels")
#     plotter.add_points(sdf_zero_cloud, color="blue", point_size=5, render_points_as_spheres=True, label="SDF ≈ 0")
#     plotter.add_axes()
#     plotter.show_grid()
#     plotter.add_legend()
#     plotter.show(title="SDF vs Binary Voxel Grid (Point Overlay)", window_size=[1000, 800])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Superimpose SDF and voxel grid.")
    parser.add_argument("--resolution", type=int, default=64, help="Resolution of the SDF grid.")
    parser.add_argument("--min_efficiency", type=float, default=0.15, help="Minimum bounding box efficiency for mesh density.")
    parser.add_argument("--pitch", type=float, default=0.01, help="Pitch for bounding box efficiency calculation.")
    parser.add_argument("--root", type=str, default='/home/user/TRELLIS/datasets/ObjaverseXL_sketchfab', help="Root directory for instances.")

    args = parser.parse_args()
    
    resolution = args.resolution
    print(f"Using resolution: {resolution}")
    instances = glob.glob(os.path.join(args.root, "voxels") + "/*.ply")
    for instance in instances:
        instance_name = os.path.basename(instance).split('.')[0]
        print(f"Processing instance: {instance_name}")
        # post_adjustment_sdf('0247cd695a0f52d0b4db04574ff873f6cec4c2bd0f5067fae16ea227e8f411b0', args.root)
        post_adjustment_sdf_mod(instance_name, args.root, save_sdf=True, sdf_folder=os.path.join(args.root,'sdfs'), visualize=False)
        