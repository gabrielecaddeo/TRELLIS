from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

from ....utils import dist_utils


class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}


class MaskPatcher(torch.nn.Module):
    def __init__(self):
        super(MaskPatcher, self).__init__()

    def forward(self, mask, patch_size=14):
        """
        Inputs:
            mask: tensor, size [B, H, W] (e.g., the original mask might not be 518x518)
            patch_size: the size of each patch, default is 14 (since 518/14=37)
        Outputs:
            patch_ratio: tensor, size [B, 37, 37], where each element represents the ratio of 1's in the corresponding patch of the mask
        """
        # 1. Resize the mask if the shape is not (518, 518) 
        if mask.shape != (518, 518): 
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(518, 518), mode='nearest')  # [B, 1, 518, 518]
        else:
            mask = mask.unsqueeze(1).float()
        
        # 2. Use unfold to divide the mask into non-overlapping patches
        # Unfold parameters: dimension, window size, stride
        patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # At this point, patches have shape [B, 1, 37, 37, 14, 14]
        
        # 3. For each patch, compute the ratio of ones in the mask by taking the mean
        patch_ratio = patches.mean(dim=(-1, -2))  # Shape becomes [B, 1, 37, 37]
        
        # 4. Remove the channel dimension
        patch_ratio = patch_ratio.squeeze(1)  # Final shape is [B, 37, 37]
        
        return patch_ratio
    

class ImageConditionedMixinConditioned:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        self.mask_patcher = MaskPatcher()
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixinConditioned, ImageConditionedMixinConditioned), 'prepare_for_training'):
            super(ImageConditionedMixinConditioned, ImageConditionedMixinConditioned).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, cond, mask_hand, mask_obj, cond_mask, x0_hand, touch, **kwargs):
        """
        Get the conditioning data.
        """
        # Encode / patch everything
        cond_enc       = self.encode_image(cond)        # [B, ..., C]
        cond_mask_enc  = self.encode_image(cond_mask)   # [B, ..., C]
        mask_hand_enc  = self.mask_patcher(mask_hand)   # [B, 37, 37]
        mask_obj_enc   = self.mask_patcher(mask_obj)    # [B, 37, 37]

        # Pack all positive conditions into a single dict
        cond_dict = {
            "cond":      cond_enc,
            "mask_hand": mask_hand_enc,
            "mask_obj":  mask_obj_enc,
            "cond_mask": cond_mask_enc,
            "x0_hand": x0_hand,
            "touch": touch,
        }

        # Build the negative / unconditional versions
        neg_cond_dict = {
            "cond":      torch.zeros_like(cond_enc),
            "mask_hand": torch.zeros_like(mask_hand_enc),
            "mask_obj":  torch.zeros_like(mask_obj_enc),
            "cond_mask": torch.zeros_like(cond_mask_enc),
            "x0_hand": torch.zeros_like(x0_hand),
            "touch": torch.zeros_like(touch),
        }

        # Call the CFG mixin: this will randomly choose, per batch item,
        # whether to use cond_dict[...] or neg_cond_dict[...]
        cond_dict = super().get_cond(cond_dict, neg_cond=neg_cond_dict, **kwargs)

        # Return the whole dict (simplest), or unpack if you really want
        return cond_dict

    
    def get_inference_cond(self, cond, mask_hand, mask_obj, cond_mask, x0_hand, touch, **kwargs):
        """
        Get the conditioning data for inference.
        """

        # 1. Encode everything exactly like in training
        cond_enc       = self.encode_image(cond)
        cond_mask_enc  = self.encode_image(cond_mask)
        mask_hand_enc  = self.mask_patcher(mask_hand)
        mask_obj_enc   = self.mask_patcher(mask_obj)

        # 2. Pack all *positive* conditions into a dict
        cond_dict = {
            "cond":      cond_enc,
            "mask_hand": mask_hand_enc,
            "mask_obj":  mask_obj_enc,
            "cond_mask": cond_mask_enc,
            "x0_hand": x0_hand,
            "touch": touch,
        }

        # 3. Build the *negative/unconditional* versions
        neg_cond_dict = {
            "cond":      torch.zeros_like(cond_enc),
            "mask_hand": torch.zeros_like(mask_hand_enc),
            "mask_obj":  torch.zeros_like(mask_obj_enc),
            "cond_mask": torch.zeros_like(cond_mask_enc),
            "x0_hand": torch.zeros_like(x0_hand),
            "touch": torch.zeros_like(touch),
        }

        # 4. Let the CFG mixin package this for the sampler
        return super().get_inference_cond(cond_dict, neg_cond=neg_cond_dict, **kwargs)


    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}
