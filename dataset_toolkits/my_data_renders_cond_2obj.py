import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render_cond(output_dir, num_views, instance):
    output_folder = os.path.join(output_dir, 'renders_cond', instance)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
            BLENDER_PATH, '-b',
            '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_2_objs.py'),
            '--',
            '--views', json.dumps(views),
            '--object_hand', os.path.join(dir, 'isaac.obj'),
            '--object_obj', os.path.join(dir, 'object.obj'),
            '--output_folder', output_folder,
            '--resolution', '1024',
            '--save_masks',
        ]

    # if file_path.endswith('.blend'):
    #     args.insert(1, file_path)
    # print('caling')
    call(args, stderr=DEVNULL, stdout=DEVNULL)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'cond_rendered': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_views', type=int, default=24,
                        help='Number of views to render')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)

    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    import glob
    directories = glob.glob('/home/user/blender-4.5.3-linux-x64/blender-4.5.3-linux-x64/google_objs/*')
    for dir in directories:
        # process objects
        
        instance = os.path.basename(dir)
        print(instance)
        func = partial(_render_cond, output_dir='/home/user/merged_google', num_views=24)
        cond_rendered = _render_cond( '/home/user/merged_google', 24, instance)

