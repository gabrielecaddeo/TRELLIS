"""Does TF32 perturb this model's output enough to matter? Compare fp32 vs TF32 outputs."""
import json, torch
from trellis import models
cfg = json.load(open('configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage1.json'))
m = getattr(models, cfg['models']['denoiser']['name'])(**cfg['models']['denoiser']['args']).cuda().eval()
sd = torch.load('/projects/gcaddeo/train_flow/TRELLIS/outputs/flow_final_norm_resume330/ckpts/denoiser_ema0.9999_step0350000.pt', map_location='cpu')
m.load_state_dict(sd, strict=False)
B, R = 4, m.resolution
g = torch.Generator(device='cuda').manual_seed(0)
x = torch.randn(B, m.in_channels, R, R, R, device='cuda', generator=g)
t = torch.full((B,), 500.0, device='cuda')
cond = {'cond': torch.randn(B,1374,m.cond_channels,device='cuda',generator=g),
        'cond_mask': torch.randn(B,1374,m.cond_channels,device='cuda',generator=g),
        'mask_hand': torch.rand(B,37,37,device='cuda',generator=g),
        'mask_obj': torch.rand(B,37,37,device='cuda',generator=g),
        'x0_hand': torch.randn(B,m.in_channels,R,R,R,device='cuda',generator=g),
        'touch': torch.rand(B,2,64,64,64,device='cuda',generator=g)}
with torch.no_grad():
    torch.backends.cuda.matmul.allow_tf32 = False
    y32 = m(x, t, dict(cond)).float()
    torch.backends.cuda.matmul.allow_tf32 = True
    ytf = m(x, t, dict(cond)).float()
rel = ((ytf - y32).norm() / y32.norm()).item()
print(f"fp32 output norm      : {y32.norm().item():.5f}")
print(f"TF32 vs fp32 rel diff : {rel:.3e}")
print(f"model prediction std  : {y32.std().item():.5f}")
print(f"=> perturbation is {100*rel:.4f}% of signal; training noise (mse~0.17) dwarfs this"
      if rel < 1e-2 else "=> LARGE - do not adopt TF32")
