import os
import json
import sys
import argparse
import glob
import pandas as pd
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # fallback if tqdm isn't installed

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def _render_cond(file_path, output_dir, num_views, instance):
    """
    One render job. Safe to run in a separate process.
    Returns a dict with 'instance', 'ok', 'msg'.
    """
    try:
        output_folder = os.path.join(output_dir, 'renders_cond', instance)
        os.makedirs(output_folder, exist_ok=True)

        # Build camera {yaw, pitch, radius, fov}
        yaws, pitchs = [], []
        offset = (np.random.rand(), np.random.rand())
        for i in range(num_views):
            y, p = sphere_hammersley_sequence(i, num_views, offset)
            yaws.append(y); pitchs.append(p)

        fov_min, fov_max = 10, 70
        radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
        radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
        k_min = 1 / radius_max**2
        k_max = 1 / radius_min**2
        ks = np.random.uniform(k_min, k_max, (1000000,))
        radius = [1 / np.sqrt(k) for k in ks]
        fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
        views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f}
                 for y, p, r, f in zip(yaws, pitchs, radius, fov)]

        args = [
            BLENDER_PATH, '-b',
            '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
            '--',
            '--views', json.dumps(views),
            '--object', os.path.expanduser(file_path),
            '--output_folder', os.path.expanduser(output_folder),
            '--resolution', '1024',
        ]
        if file_path.endswith('.blend'):
            args.insert(1, file_path)

        ret = call(args, stdout=DEVNULL)
        if ret != 0:
            return {'instance': instance, 'ok': False, 'msg': f'Blender exit code {ret}'}

        tf_path = os.path.join(output_folder, 'transforms.json')
        if os.path.exists(tf_path):
            return {'instance': instance, 'ok': True, 'msg': 'cond_rendered'}
        else:
            return {'instance': instance, 'ok': False, 'msg': 'transforms.json missing'}
    except Exception as e:
        return {'instance': instance, 'ok': False, 'msg': f'Exception: {e}'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_views', type=int, default=24, help='Number of views to render')
    parser.add_argument('--max_workers', type=int, default=1, help='Parallel workers')
    parser.add_argument('--input_root', type=str, default='/home/user/TRELLIS/merged',
                        help='Root containing one subfolder per instance (with merged.obj)')
    parser.add_argument('--output_dir', type=str, default='/home/user/merged_renders_cond',
                        help='Where to write renders_cond/<instance>')
    args = parser.parse_args()

    print('Checking blender...', flush=True)
    _install_blender()

    os.makedirs(os.path.join(args.output_dir, 'renders_cond'), exist_ok=True)

    directories = sorted(glob.glob(os.path.join(args.input_root, '*')))
    jobs = []
    for d in directories:
        instance = os.path.basename(d)
        file_path = os.path.join(d, 'merged.obj')
        if not os.path.isfile(file_path):
            # skip silently; you could log if you want
            continue
        jobs.append((file_path, args.output_dir, args.num_views, instance))

    print(f'Found {len(jobs)} instances to render.')

    results = []
    # Use a process pool so each render is isolated
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_inst = {
            ex.submit(_render_cond, *job): job[3]  # instance name
            for job in jobs
        }
        iterator = as_completed(future_to_inst)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(future_to_inst), desc='Rendering', ncols=100)

        for fut in iterator:
            res = fut.result()
            results.append(res)

    # Optional: write a small CSV summary
    df = pd.DataFrame(results).sort_values('instance')
    summary_csv = os.path.join(args.output_dir, 'cond_rendered_summary.csv')
    df.to_csv(summary_csv, index=False)
    ok = (df['ok'].sum() if not df.empty else 0)
    print(f'Completed: {ok}/{len(results)} successful. Summary: {summary_csv}')
