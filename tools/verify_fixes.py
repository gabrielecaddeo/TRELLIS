"""
Model-level checks for the TEACHER_RETRAIN.md fixes, before committing GPU-days to a retrain.

    python tools/verify_fixes.py --config configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage1.json

Checks, in order:
  1. the base (pre-conditioning) checkpoint loads, and the only `missing` keys are the
     genuinely new conditioning modules;
  2. `initialize_input_layer_x0h()` really copies input_layer -> input_layer_x0h;
  3. at init the hand branch is a no-op (zero-initialised to_out), so the warm start is
     numerically identical to the base model regardless of what enters the branch;
  4. once the branch is un-zeroed, permuting the hand latent CHANGES the output -- this is
     fix 1. Re-run with --no_hand_pe to see the same check fail, which is the bug.
  5. `forward()` no longer mutates the caller's cond dict, so two calls on one dict agree.
"""
import argparse
import json

import torch

from trellis import models

BASE_CKPT = ('/projects/gcaddeo/train_flow/TRELLIS/outputs/flow_final_norm_resume330/'
             'ckpts/denoiser_ema0.9999_step0350000.pt')

NEW_MODULE_PREFIXES = (
    'input_layer_x0h.', 'mask_hand_embedder.', 'mask_hand_pos_emb',
    'contact_encoder.', 'fuse_x0_contact.',
)
NEW_BLOCK_SUBSTRINGS = ('norm_hand.', 'cross_attn_mask_hand.', 'cross_attn_hand.')


def is_new(key: str) -> bool:
    return key.startswith(NEW_MODULE_PREFIXES) or any(s in key for s in NEW_BLOCK_SUBSTRINGS)


def make_cond(model, B, device, dtype, generator):
    res = model.resolution
    def rnd(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    return {
        'cond': rnd(B, 1374, model.cond_channels),
        'cond_mask': rnd(B, 1374, model.cond_channels),
        'mask_hand': torch.rand(B, 37, 37, device=device, dtype=dtype, generator=generator),
        'mask_obj': torch.rand(B, 37, 37, device=device, dtype=dtype, generator=generator),
        'x0_hand': rnd(B, model.in_channels, res, res, res),
        'touch': torch.rand(B, 2, 64, 64, 64, device=device, dtype=dtype, generator=generator),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',
                    default='configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage1.json')
    ap.add_argument('--ckpt', default=BASE_CKPT)
    ap.add_argument('--no_hand_pe', action='store_true',
                    help='disable the fix-1 positional embedding, to show check 4 failing')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = json.load(open(args.config))
    margs = cfg['models']['denoiser']['args']
    model = getattr(models, cfg['models']['denoiser']['name'])(**margs).to(device)
    model.eval()
    if args.no_hand_pe:
        model.pe_mode_backup, model.pe_mode = model.pe_mode, 'none-for-test'

    ok = True

    # ---- 1. warm start ----
    state = torch.load(args.ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    stale = [k for k in missing if not is_new(k)]
    print(f'[1] loaded {len(state)} tensors: {len(missing)} missing, {len(unexpected)} unexpected')
    print(f'    missing keys not accounted for by the new conditioning modules: {stale}')
    print(f'    unexpected keys: {unexpected}')
    ok &= not stale and not unexpected
    assert 'mask_hand_pos_emb' in missing or not hasattr(model, 'mask_hand_pos_emb'), \
        'mask_hand_pos_emb should be reported missing by the base checkpoint'

    # ---- 2. input_layer_x0h init ----
    model.initialize_input_layer_x0h()
    same = torch.equal(model.input_layer.weight, model.input_layer_x0h.weight) and \
           torch.equal(model.input_layer.bias, model.input_layer_x0h.bias)
    print(f'[2] initialize_input_layer_x0h copies input_layer: {same}')
    ok &= same

    B = 2
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(B, model.in_channels, model.resolution, model.resolution, model.resolution,
                    device=device, generator=g)
    t = torch.full((B,), 500.0, device=device)
    cond = make_cond(model, B, device, torch.float32, g)
    cond_zero = dict(cond)
    cond_zero['x0_hand'] = torch.zeros_like(cond['x0_hand'])
    cond_zero['touch'] = torch.zeros_like(cond['touch'])

    with torch.no_grad():
        # ---- 3. the hand branch is a no-op at init ----
        # This is what makes the warm start exact: whatever enters the hand branch, the
        # zero-initialised to_out means it contributes nothing to the residual stream, so
        # step 0 reproduces the base model.
        y0 = model(x, t, cond)
        y1 = model(x, t, cond_zero)
        d_init = (y0 - y1).abs().max().item()
        print(f'[3] max|out(hand) - out(no hand)| at init = {d_init:.3e}  '
              f'(want ~0: zero-init to_out makes the warm start exact)')
        ok &= d_init < 1e-5

        # ---- 5. forward does not mutate the caller's cond dict ----
        shapes_before = {k: tuple(v.shape) for k, v in cond.items()}
        ya = model(x, t, cond)
        yb = model(x, t, cond)
        shapes_after = {k: tuple(v.shape) for k, v in cond.items()}
        idem = torch.allclose(ya, yb) and shapes_before == shapes_after
        print(f'[5] two forwards on the same cond dict agree, shapes unchanged: {idem}')
        ok &= idem

    del model, state
    torch.cuda.empty_cache()

    # ---- 4. permutation sensitivity of the hand branch ----
    # Built separately with use_touch=False: the contact encoder consumes `touch` on the
    # 64^3 grid and fuses it into x0h before patchify, so permuting x0h alone would not be
    # a pure permutation of the hand token bag and would muddy the measurement. Random
    # weights are fine here -- the question is structural, not about a trained model.
    margs4 = dict(margs)
    margs4['use_touch'] = False
    margs4['num_blocks'] = 4          # the property is per-block; 4 is enough and much cheaper
    m4 = getattr(models, cfg['models']['denoiser']['name'])(**margs4).to(device).eval()
    if args.no_hand_pe:
        m4.pe_mode = 'none-for-test'
    # Un-zero the hand branch so it can actually speak, and out_layer / adaLN so anything
    # reaches the output at all -- initialize_weights() zeroes all three, and this model
    # loads no checkpoint to overwrite them.
    for blk in m4.blocks:
        torch.nn.init.normal_(blk.cross_attn_hand.to_out.weight, std=0.5)
        torch.nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.02)
    torch.nn.init.normal_(m4.out_layer.weight, std=0.02)

    g4 = torch.Generator(device=device).manual_seed(7)
    x4 = torch.randn(B, m4.in_channels, *([m4.resolution] * 3), device=device, generator=g4)
    c4 = make_cond(m4, B, device, torch.float32, g4)
    c4_perm = dict(c4)
    # Same bag of hand features, different spatial layout.
    idx = torch.randperm(m4.resolution ** 3, generator=g4, device=device)
    c4_perm['x0_hand'] = c4['x0_hand'].reshape(B, m4.in_channels, -1)[:, :, idx].reshape_as(
        c4['x0_hand'])

    def perm_sensitivity():
        with torch.no_grad():
            z0 = m4(x4, t, c4)
            z1 = m4(x4, t, c4_perm)
            return ((z0 - z1).norm() / z0.norm().clamp_min(1e-12)).item()

    saved_pe, m4.pe_mode = m4.pe_mode, 'none-for-test'
    rel_without = perm_sensitivity()
    m4.pe_mode = saved_pe
    rel_with = perm_sensitivity()

    print(f'[4] relative output change from permuting the hand latent:')
    print(f'      without pos_emb on x0h_tokens = {rel_without:.6f}   <- the bug: exactly '
          f'permutation-invariant,')
    print(f'                                                             the model cannot tell '
          f'WHERE the hand is')
    print(f'      with    pos_emb on x0h_tokens = {rel_with:.6f}   <- fix 1')
    # The un-fixed branch is invariant up to float reassociation, so compare the two rather
    # than testing rel_with against an absolute threshold that depends on the random weights.
    ok &= rel_with > 1e-4 and rel_with > 20 * max(rel_without, 1e-9)
    if args.no_hand_pe:
        print('    (--no_hand_pe only affects the model built for checks 1-3, 5)')

    print('\nRESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
