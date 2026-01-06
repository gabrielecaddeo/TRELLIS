import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis.models as models


torch.set_grad_enabled(False)

def get_sdf(instance, number):
    sdf_hand = torch.tensor(np.load(os.path.join(opt.output_dir, 'data_pose_norm', instance, 'sdfs', f'{instance}_f{number:03d}__hand.npy')), dtype=torch.float32)
    sdf_hand = torch.clamp(sdf_hand, -2, 2)
    sdf_hand = sdf_hand.unsqueeze(0).unsqueeze(0)
    sdf_obj = torch.tensor(np.load(os.path.join(opt.output_dir, 'data_pose_norm', instance, 'sdfs', f'{instance}_f{number:03d}__object.npy')), dtype=torch.float32)
    sdf_obj = torch.clamp(sdf_obj, -2, 2)
    sdf_obj = sdf_obj.unsqueeze(0).unsqueeze(0)
    return sdf_hand, sdf_obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=False, default='/home/user/TRELLIS/datasets/Hands',
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='outputs',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default='0140000',
                        help='Checkpoint to load')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    # opt.enc_model = 'ss_vae_sdf_geom_eikonal_001_normal_1_l1_1_conv3d_16l8_fp16_1node_ABO_HSSD_3D_FUTURE_change_phase0'
    opt = edict(vars(opt))

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model}_{opt.ckpt}'
        cfg = edict(json.load(open(os.path.join('/home/user/TRELLIS',opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join('/home/user/TRELLIS',opt.model_root, opt.enc_model, 'ckpts', f'encoder_ema0.9999_step{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    os.makedirs(os.path.join(opt.output_dir, 'ss_latents_sdf_pose', latent_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
        print(f'Total instances in metadata: {len(metadata)}')

    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
            
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['voxelized'] == True]
        print(len(metadata))
        if f'ss_latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'ss_latent_{latent_name}'] == False]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    
    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    
    # for sha256 in copy.copy(sha256s):
    #     if os.path.exists(os.path.join(opt.output_dir, 'ss_latents', latent_name, f'{sha256}.npz')):
    #         records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
    #         sha256s.remove(sha256)

    # encode latents
    load_queue = Queue(maxsize=64)
    jobs = [(sha, i) for sha in sha256s for i in range(24)]
    print(f"Number of sha256s to process: {len(sha256s)}")
    print(f"Example sha256s: {sha256s[:5]}")
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=16) as saver_executor:
            
            def loader(sha256, sdf_number):
                try:
                    sdf_hand, sdf_obj = get_sdf(sha256, sdf_number)
                    load_queue.put((sha256, sdf_hand, sdf_obj, sdf_number))
                except Exception as e:
                    print(f"Error loading SDF for {sha256} #{sdf_number}: {e}", flush=True)
            loader_executor.map(lambda x: loader(*x), jobs)
            print('loaded')
            def saver(sha256, pack, sdf_number, type):
                save_path = os.path.join(opt.output_dir, 'ss_latents_sdf_pose', latent_name, f'{sha256}_{sdf_number}__{type}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'ss_latent_{latent_name}_{sdf_number}_{type}': True})
                
            for _ in tqdm(range(len(jobs)), desc="Extracting latents"):
                sha256, sdf_hand, sdf_obj, sdf_number = load_queue.get()
                # print(f"Got {sha256} #{sdf_number}", flush=True)
                try:
                    sdf_hand = sdf_hand.cuda().float()
                    sdf_obj = sdf_obj.cuda().float()
                    # print("Moved to CUDA", flush=True)
                    with torch.no_grad():
                        latent_hand = encoder(sdf_hand, sample_posterior=False)
                        latent_obj = encoder(sdf_obj, sample_posterior=False)
                    # print("Encoded latent", flush=True)
                    assert torch.isfinite(latent_hand).all(), f"Non-finite latent {sha256} f{sdf_number:03d}"
                    assert torch.isfinite(latent_obj).all(), f"Non-finite latent {sha256} f{sdf_number:03d}"
                    pack = {'mean': latent_hand[0].cpu().numpy()}
                    saver_executor.submit(saver, sha256, pack, sdf_number, 'hand')
                    pack = {'mean': latent_obj[0].cpu().numpy()}
                    saver_executor.submit(saver, sha256, pack, sdf_number, 'object')
                    # print("Submitted saver", flush=True)
                except Exception as e:
                    print(f"Error processing {sha256} #{sdf_number}: {e}", flush=True)
                
            saver_executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing {e}.")
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'ss_latent_{latent_name}_{opt.rank}_sdf.csv'), index=False)

