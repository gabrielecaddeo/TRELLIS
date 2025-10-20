import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
        # x_float = x.float()
        # ##
        # std = x_float.std(dim=-1, keepdim=True)
        # if (std < 1e-6).any():
        #     print("[Warning] Very low std detected in LayerNorm input.")
        # if torch.isnan(x_float).any():
        #     print("[LayerNorm32] NaNs detected in input")
        # if torch.isinf(x_float).any():
        #     print("[LayerNorm32] Infs detected in input")

        # out = super().forward(x_float)

        # if torch.isnan(out).any():
        #     print("[LayerNorm32] NaNs detected in output")
        # if torch.isinf(out).any():
        #     print("[LayerNorm32] Infs detected in output")

        # return out.to(x.dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    
    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    