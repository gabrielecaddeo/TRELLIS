from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..datasets import TestDatasetConditioned



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
    

class TrellisImageTo3DPipelineConditioned(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)
        

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipelineConditioned":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipelineConditioned, TrellisImageTo3DPipelineConditioned).from_pretrained(path, 'pipeline_conditioned')
        new_pipeline = TrellisImageTo3DPipelineConditioned()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])
        new_pipeline.dataset = TestDatasetConditioned()
        new_pipeline.mask_patcher = MaskPatcher()

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size_old = size
        size = int(size * 1.5)
        
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2

        # Asymmetric padding: more padding on the right, less on the left
        # left_padding = 0  # Align object to the left
        # right_padding = size * 2  # Give much more padding on the right
        
        # # Adjust the bounding box: align object to the left
        # bbox = (bbox[0] - left_padding, bbox[1] - size // 2, bbox[0] + size + right_padding, bbox[3] + size // 2)
        
        # # Make sure the new bbox stays within the image bounds
        # bbox = (max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], output.width), min(bbox[3], output.height))
    
        
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        output.save("/home/user/TRELLIS/bb.png")
        return output
    
    def preprocess_image2(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image with asymmetric padding.
        The object will be aligned to the left or right side of the bounding box with more padding on one side.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        
        # Asymmetric padding: more padding on the right or left
        left_padding = size // 5  # Adjust for asymmetric padding (smaller padding on left)
        right_padding = size * 2  # Much larger padding on the right
        
        # Adjust the bounding box with asymmetric padding
        bbox = (bbox[0] - left_padding, bbox[1] - size // 2, 
                bbox[0] + size + right_padding, bbox[3] + size // 2)
        
        # Make sure the new bbox stays within the image bounds
        bbox = (max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], output.width), min(bbox[3], output.height))
        
        # Crop the image
        output = output.crop(bbox)  # type: ignore
        
        # Now resize to a fixed size while keeping the aspect ratio
        aspect_ratio = output.width / output.height
        target_size = 518
        if output.width > output.height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
        
        # Resize without distorting the aspect ratio
        output = output.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # If necessary, pad to make it exactly 518x518 while maintaining the image centered
        padded_image = Image.new('RGB', (518, 518), (0, 0, 0))
        offset_x = (518 - new_width) // 2
        offset_y = (518 - new_height) // 2
        padded_image.paste(output, (offset_x, offset_y))
        
        padded_image.save("/home/user/TRELLIS/bb_asym.png")
        return padded_image



    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, cond, mask_hand, mask_obj, cond_mask, x0_hand, touch, **kwargs) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond_enc       = self.encode_image(cond)        # [B, ..., C]
        cond_mask_enc  = self.encode_image(cond_mask)   # [B, ..., C]
        mask_hand_enc  = self.mask_patcher(mask_hand.to(self.device))   # [B, 37, 37]
        mask_obj_enc   = self.mask_patcher(mask_obj.to(self.device))    # [B, 37, 37]

        # Pack all positive conditions into a single dict
        cond_dict = {
            "cond":      cond_enc,
            "mask_hand": mask_hand_enc,
            "mask_obj":  mask_obj_enc,
            "cond_mask": cond_mask_enc,
            "x0_hand": x0_hand.to(self.device),
            "touch": touch.to(self.device),
        }

        # Build the negative / unconditional versions
        neg_cond_dict = {
            "cond":      torch.zeros_like(cond_enc).to(self.device),
            "mask_hand": torch.zeros_like(mask_hand_enc).to(self.device),
            "mask_obj":  torch.zeros_like(mask_obj_enc).to(self.device),
            "cond_mask": torch.zeros_like(cond_mask_enc).to(self.device),
            "x0_hand": torch.zeros_like(x0_hand).to(self.device),
            "touch": torch.zeros_like(touch).to(self.device),
        }
        return {
            'cond': cond_dict,
            'neg_cond': neg_cond_dict,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        # for i in range(len(z_s.pred_x_t)): # to decomment this, remover .samples from z_s
        #     torch.save(torch.argwhere(decoder(z_s.pred_x_t[i])<=0)[:, [0, 2, 3, 4]].int(), f"/home/user/TRELLIS/coords_asym/00000{i}.pt")
        # # exit(0)
        coords = torch.argwhere(decoder(z_s)<=0)[:, [0, 2, 3, 4]].int()
        # for i in range(coords.shape[0]):
        #     pred_x_t
        return coords

    def sample_sparse_structure_velocity(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        # print(type(decoder))
        z_s = self.sparse_structure_sampler.sample_velocity_conditioned(
            flow_model,
            noise,
            decoder,
            cond,
            **sampler_params,
            verbose=True
        )
        
        exit(0)
        for i in range(len(z_s.pred_x_t)): # to decomment this, remover .samples from z_s
            torch.save(torch.argwhere(decoder(z_s.pred_x_t[i])<=0)[:, [0, 2, 3, 4]].int(), f"/home/user/TRELLIS/coords_asym_velocity/00000{i}.pt")
        
        coords = torch.argwhere(decoder(z_s)<=0)[:, [0, 2, 3, 4]].int()
        # for i in range(coords.shape[0]):
        #     pred_x_t
        return coords

    def sample_sparse_structure_optimization(
        self,
        noise: torch.Tensor,
        cond: dict,
        target: torch.Tensor,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        
        # Combine default and user-provided sampler parameters
        full_sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        # Run the optimization using the currently configured sampler
        optimization_result = self.sparse_structure_sampler.sample_optimization(
            flow_model,
            noise,
            decoder, 
            target,
            **cond,
            **full_sampler_params,
            verbose=True
        )
        
        # BUG FIX: The sampler now returns a dictionary containing the final SDF.
        # We use this SDF to compute the coordinates. This was a point of failure.
        final_sdf = optimization_result.final_sdf
        coords = torch.argwhere(final_sdf <= 0)[:, [0, 2, 3, 4]].int()
        
        # Add coords to the result object to be used in the next pipeline stage
        optimization_result.coords = coords
        
        return optimization_result

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image2(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    
    def process_data_conditioned(self, data_path: str) -> dict:
        """
        Process the image for conditioning.

        Args:
            data_path (str): The path to the image.

        Returns:
            dict: The processed image.
        """
        image = Image.open(data_path).convert('RGB')
        return image
    
    def run_velocity(
        self,
        root: str,
        instance_name: str,
        view: int = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        data = self.dataset.get_instance(root, instance_name, view=view)

        dicts = self.get_cond(**data)
        coords = self.sample_sparse_structure_velocity(dicts, num_samples, sparse_structure_sampler_params)
        exit(0)
        slat = self.sample_slat(data, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    

    
    ##############################
    ##---- Lolli's Stuff
    ##############################

    @contextmanager
    def _override_sampler(self, guidance_config: dict):
        """
        A context manager to temporarily replace the sparse structure sampler.
        This allows for dynamically choosing an optimization strategy at runtime.
        """
        original_sampler = self.sparse_structure_sampler
        try:
            config = guidance_config.copy()
            sampler_name = config.pop('name', None)
            if not sampler_name:
                raise ValueError("guidance_config must include a 'name' key specifying the sampler class.")
            
            print(f"INFO: Temporarily injecting '{sampler_name}' with config: {config}")
            sampler_class = getattr(samplers, sampler_name)
            
            # Get base arguments (like sigma_min) for the sampler
            base_args = self._pretrained_args['sparse_structure_sampler']['args']
            
            # Instantiate the new sampler
            self.sparse_structure_sampler = sampler_class(**base_args, **config)
            yield
        finally:
            print("INFO: Restoring original sampler.")
            self.sparse_structure_sampler = original_sampler

    # Also includes the choice of sampler
    def run_optimization(
        self,
        image: Image.Image,
        noise: torch.Tensor,
        constraint_sdf: torch.Tensor,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the guidance optimization pipeline. Also gives the mesh/gaussian/radiance_field (?).

        This version allows for dynamically selecting the guidance sampler to use,
        enabling easy comparison between different strategies like D-Flow and OC-Flow.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            constraint_sdf (): Constraint sdf.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
            guidance_sampler (str, optional): The name of the sampler class to use for
                                              optimization (e.g., 'OCFlowAdaptiveSampler',
                                              'FlowMidpointSampler'). If None, uses the
                                              default sampler.
        """
            
        params_copy = sparse_structure_sampler_params.copy()
        
        # Extract guidance-specific configuration from the sampler parameters
        guidance_config = params_copy.pop('guidance_config', None)
        
        target = constraint_sdf.to(self.device)
            
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)

        if guidance_config:
            with self._override_sampler(guidance_config):
                opt_results = self.sample_sparse_structure_optimization(noise, cond, target, params_copy)
        else:
            print("WARNING: No 'guidance_config' provided. Using default sampler for optimization.")
            opt_results = self.sample_sparse_structure_optimization(noise, cond, target, params_copy)
        
        coords = opt_results.coords
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        final_outputs = self.decode_slat(slat, formats)

        return final_outputs

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
