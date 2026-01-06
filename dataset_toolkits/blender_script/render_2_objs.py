import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def init_render(engine='CYCLES', resolution=512, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True
        
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()

    # Disable all devices first
    for device in prefs.devices:
        device.use = False

    # Enable only the first visible CUDA device
    for device in prefs.devices:
        if device.type == 'CUDA':
            device.use = True
            break
    

def get_view_layer():
    """Return the active view layer or the first scene view layer."""
    scene = bpy.context.scene
    # Prefer the context view_layer if it belongs to the scene
    vl = bpy.context.view_layer
    try:
        if vl and vl.name in scene.view_layers:
            return scene.view_layers[vl.name]
    except Exception:
        pass
    # Fallback: first available
    return scene.view_layers[0]

def init_nodes(
    save_depth: bool = False,
    save_normal: bool = False,
    save_albedo: bool = False,
    save_mist: bool = False,
    save_masks: bool = False,
    num_masks: int = 0,
):
    """
    Set up compositor nodes for depth, normal, albedo, mist and object masks.
    """
    if not any([save_depth, save_normal, save_albedo, save_mist, save_masks]):
        return {}, {}
    outputs = {}
    spec_nodes = {}
    
    bpy.context.scene.use_nodes = True
    scene = bpy.context.scene
    view_layer = get_view_layer()
    # enable passes
    view_layer.use_pass_z = save_depth
    view_layer.use_pass_normal = save_normal
    view_layer.use_pass_diffuse_color = save_albedo
    view_layer.use_pass_mist = save_mist
    view_layer.use_pass_object_index = save_masks
    
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 10 # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)
        
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = map
    
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
    
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
        
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output

    # --- object index masks (one per ID) ---
    if save_masks and num_masks > 0:
        for idx in range(1, num_masks + 1):
            id_node = nodes.new('CompositorNodeIDMask')
            id_node.index = idx

            mask_out = nodes.new('CompositorNodeOutputFile')
            mask_out.base_path = ''
            mask_out.file_slots[0].use_node_format = True
            mask_out.format.file_format = 'PNG'
            mask_out.format.color_mode = 'BW'
            mask_out.format.color_depth = '8'

            links.new(render_layers.outputs['IndexOB'], id_node.inputs['ID value'])
            links.new(id_node.outputs['Alpha'], mask_out.inputs[0])

            outputs[f'mask_{idx}'] = mask_out
        
    return outputs, spec_nodes


def init_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam


def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }


def check_vertex_colors(obj: bpy.types.Object) -> None:
    """Check if the object has vertex colors."""
    if obj.type == 'MESH':
        if len(obj.data.vertex_colors) > 0:
            print(f"{obj.name} has vertex colors!")
        else:
            print(f"{obj.name} does NOT have vertex colors.")


def assign_vertex_color_to_material(obj: bpy.types.Object):
    """Assign the vertex colors to the material for rendering."""
    if obj.type == 'MESH':
        # Create a material if none exists
        if len(obj.data.materials) == 0:
            mat = bpy.data.materials.new(name="VertexColorMaterial")
            obj.data.materials.append(mat)
        else:
            mat = obj.data.materials[0]

        # Enable the use of nodes (Shading system)
        mat.use_nodes = True
        tree = mat.node_tree
        nodes = tree.nodes
        
        # Create a vertex color node
        vertex_color_node = nodes.new(type='ShaderNodeAttribute')
        vertex_color_node.attribute_name = "Col"  # The default name for the vertex color layer
        
        # Create a diffuse BSDF shader
        bsdf = nodes.get("Principled BSDF")
        tree.links.new(vertex_color_node.outputs[0], bsdf.inputs['Base Color'])


def load_object(object_path: str) -> List[bpy.types.Object]:
    """Loads a model with a supported file extension into the scene.

    Returns a list of newly imported mesh objects.
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return []

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    before_objs = set(bpy.context.scene.objects)

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)

    after_objs = set(bpy.context.scene.objects)
    new_objs = [obj for obj in (after_objs - before_objs) if obj.type == "MESH"]
    return new_objs


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
        

def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objs:
        return
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
            

def delete_custom_normals():
    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()


def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['View Layer'].material_override = new_mat


def unhide_all_objects() -> None:
    """Unhides all objects in the scene."""
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        

def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes."""
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objs:
        return
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = mesh_objs[0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
        

def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene."""
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objs:
        return
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene."""
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene to a unit cube centered at origin."""
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix


def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode)

    # Decide how many masks we will write (hand + object)
    num_masks = 2 if arg.save_masks and arg.object_hand and arg.object_obj else 0

    outputs, spec_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist,
        save_masks=arg.save_masks,
        num_masks=num_masks,
    )

    # Load geometry: either two separate objects (preferred) or a single one
    if arg.object_hand and arg.object_obj:
        init_scene()
        print("Loading hand object:", arg.object_hand)
        hand_objs = load_object(arg.object_hand)
        print("Loading grasped object:", arg.object_obj)
        obj_objs = load_object(arg.object_obj)

        # Assign object indices for masks
        for o in hand_objs:
            o.pass_index = 1
        for o in obj_objs:
            o.pass_index = 2

        if arg.split_normal:
            split_mesh_normal()

    else:
        # Fallback: single object as before
        if arg.object is None:
            raise RuntimeError("No object specified. Provide --object or --object_hand/--object_obj.")
        if arg.object.endswith(".blend"):
            delete_invisible_objects()
        else:
            init_scene()
            print(arg.object)
            load_object(arg.object)
            if arg.split_normal:
                split_mesh_normal()

    print('[INFO] Scene initialized.')
    
    # normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    
    # Initialize camera and lighting
    cam = init_camera()
    init_lighting()
    print('[INFO] Camera and lighting initialized.')

    # Override material
    if arg.geo_mode:
        override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }
    views = json.loads(arg.views)

    for i, view in enumerate(views):
        cam.location = (
            view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['pitch'])
        )
        cam.data.lens = 16 / np.tan(view['fov'] / 2)
        
        if arg.save_depth:
            spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
            spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)
        
        # main RGB
        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png')

        # auxiliary outputs (depth, normal, albedo, mist, masks)
        for name, output in outputs.items():
            output.file_slots[0].path = os.path.join(arg.output_folder, f'{i:03d}_{name}')
            
        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        # Fix filenames (Blender appends frame numbers)
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}*.{ext}')
            if not path:
                continue
            path = path[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
            
        # Save camera parameters
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        }
        if arg.save_depth:
            metadata['depth'] = {
                'min': view['radius'] - 0.5 * np.sqrt(3),
                'max': view['radius'] + 0.5 * np.sqrt(3)
            }
        to_export["frames"].append(metadata)
    
    # Save the camera parameters
    with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
        
    if arg.save_mesh:
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')        
        with open("/tmp/blender_debug_log.txt", "a") as f:
            f.write(f"[INFO] Inside mesh to {os.path.join(arg.output_folder, 'mesh.ply')}\n")
            f.write(f"arg.save_mesh: {arg.save_mesh}\n")
    
        # export ply mesh
        bpy.ops.export_mesh.ply(filepath=os.path.join(arg.output_folder, 'mesh.ply'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, default=None,
                        help='Path to single 3D model file to be rendered (legacy mode).')
    parser.add_argument('--object_hand', type=str, default=None,
                        help='Path to hand 3D model file (for mask index 1).')
    parser.add_argument('--object_obj', type=str, default=None,
                        help='Path to grasped object 3D model file (for mask index 2).')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--save_masks', action='store_true', help='Save object index masks (hand & object).')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
