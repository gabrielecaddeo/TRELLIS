import os
import torch
from safetensors.torch import save_file

def _torch_load_cpu(path: str):
    # weights_only is safer (prevents pickled code execution) but not available in all torch versions
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def convert_pt_to_safetensors(pt_path: str, out_path: str):
    obj = _torch_load_cpu(pt_path)

    # Unwrap common checkpoint structure
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]

    # Build dict[str, Tensor]
    if isinstance(obj, dict):
        tensors = {k: v for k, v in obj.items() if torch.is_tensor(v)}
    elif isinstance(obj, (list, tuple)):
        tensors = {f"tensor_{i:06d}": t for i, t in enumerate(obj) if torch.is_tensor(t)}
    else:
        raise TypeError(f"Unsupported .pt content type: {type(obj)}")

    if not tensors:
        raise ValueError(f"No tensors found in {pt_path} (cannot write safetensors).")

    # safetensors requires CPU tensors; contiguous is a good idea
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_file(tensors, out_path)
    return len(tensors)


n = convert_pt_to_safetensors(
    "/home/user/TRELLIS/outputs/flow_conditioned_all_losses/ckpts/denoiser_ema0.9999_step0018000.pt",
    "/home/user/TRELLIS/outputs/flow_conditioned_all_losses/ckpts/denoiser_ema0.9999_step0018000.safetensors",
)
print("Saved tensors:", n)
