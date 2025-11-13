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
    sdf = torch.tensor(np.load(os.path.join(opt.output_dir, 'data', instance, 'sdfs', f'{instance}_f{number:03d}.npy')), dtype=torch.float32)
    sdf = torch.clamp(sdf, -2, 2)
    sdf = sdf.unsqueeze(0).unsqueeze(0)
    return sdf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, default='outputs',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default='step0132000',
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


    latent_name = f'{opt.enc_model}_{opt.ckpt}'
    cfg = edict(json.load(open(os.path.join('/home/user/TRELLIS',opt.model_root, opt.enc_model, 'config.json'), 'r')))
    encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
    decoder = getattr(models, cfg.models.decoder.name)(**cfg.models.decoder.args).cuda()
    ckpt_path = os.path.join('/home/user/TRELLIS',opt.model_root, opt.enc_model, 'ckpts', f'encoder_ema0.9999_step{opt.ckpt}.pt')
    encoder.load_state_dict(torch.load(ckpt_path), strict=False)
    ckpt_path = os.path.join('/home/user/TRELLIS',opt.model_root, opt.enc_model, 'ckpts', f'decoder_ema0.9999_step{opt.ckpt}.pt')
    decoder.load_state_dict(torch.load(ckpt_path), strict=False)
    encoder.eval()
    decoder.eval()
    print(f'Loaded model from {ckpt_path}')

    # filter out objects that are already processed
    # for sha256 in copy.copy(sha256s):
    #     if os.path.exists(os.path.join(opt.output_dir, 'ss_latents', latent_name, f'{sha256}.npz')):
    #         records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
    #         sha256s.remove(sha256)
    # sdf = torch.tensor(np.load(os.path.join('/home/user/TRELLIS', 'datasets/ABO/data_pose/0037f821028abeef50670d1f7cb60629a10d72f8e4195db7e483d6ab26f685fd/sdfs/0037f821028abeef50670d1f7cb60629a10d72f8e4195db7e483d6ab26f685fd_f000.npy')), dtype=torch.float32)
    sdf = torch.tensor(np.load(os.path.join('/home/user/hand_f000.npy')), dtype=torch.float32)
    
    sdf = torch.clamp(sdf, -2, 2)
    sdf = sdf.unsqueeze(0).unsqueeze(0).cuda().float()
    with torch.no_grad():
        latent = encoder(sdf, sample_posterior=False)
        recon = decoder(latent)
    torch.save(recon.detach().cpu(), '/home/user/recon_test_hand.pt')
    

