import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.mm_use_sam3_conditioning = getattr(args, 'mm_use_sam3_conditioning', False)
        self.mm_sam3_vision_tower = getattr(args, 'mm_sam3_vision_tower', None)
        self.mm_sam3_blend_alpha = float(getattr(args, 'mm_sam3_blend_alpha', 0.35))
        self.mm_sam3_mask_gamma = float(getattr(args, 'mm_sam3_mask_gamma', 1.0))
        self.mm_sam3_device = getattr(args, 'mm_sam3_device', 'cpu')
        self.mm_sam3_dtype = getattr(args, 'mm_sam3_dtype', 'auto')
        self.mm_sam3_unload_after_forward = getattr(args, 'mm_sam3_unload_after_forward', False)
        self.sam3_is_loaded = False

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_sam3_model(self):
        if self.sam3_is_loaded:
            return

        if not self.mm_sam3_vision_tower:
            raise ValueError(
                "mm_sam3_vision_tower must be set when mm_use_sam3_conditioning=True."
            )

        try:
            from transformers import Sam3Model, Sam3ImageProcessor
        except ImportError as exc:
            raise ImportError(
                "Transformers with SAM3 support is required when "
                "mm_use_sam3_conditioning=True."
            ) from exc

        self.sam3_image_processor = Sam3ImageProcessor.from_pretrained(self.mm_sam3_vision_tower)
        sam3_model = Sam3Model.from_pretrained(self.mm_sam3_vision_tower)
        self.sam3_vision_tower = sam3_model.vision_encoder
        del sam3_model
        self.sam3_vision_tower.requires_grad_(False)
        self.sam3_vision_tower.eval()
        sam3_device = self.device if self.mm_sam3_device in (None, '', 'vision', 'clip') else torch.device(self.mm_sam3_device)
        sam3_dtype = self._resolve_sam3_dtype(sam3_device)
        self.sam3_vision_tower.to(device=sam3_device, dtype=sam3_dtype)
        self.sam3_is_loaded = True

    def unload_sam3_model(self):
        if not self.sam3_is_loaded:
            return
        if hasattr(self, "sam3_vision_tower"):
            del self.sam3_vision_tower
        self.sam3_is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_sam3_dtype(self, sam3_device):
        dtype_name = str(self.mm_sam3_dtype or "auto").lower()
        if dtype_name in ("auto", ""):
            return torch.float32 if sam3_device.type == "cpu" else self.dtype
        if dtype_name in ("fp32", "float32"):
            return torch.float32
        if dtype_name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype_name in ("fp16", "float16", "half"):
            return torch.float16
        raise ValueError(f"Unsupported mm_sam3_dtype: {self.mm_sam3_dtype}")

    def _sam3_input_size(self):
        size = getattr(self.sam3_image_processor, "size", None)
        if isinstance(size, dict):
            height = size.get("height")
            width = size.get("width")
            if height is not None and width is not None:
                return int(height), int(width)

        backbone_config = getattr(self.sam3_vision_tower.config, "backbone_config", None)
        image_size = getattr(backbone_config, "image_size", 1008)
        if isinstance(image_size, (tuple, list)):
            return int(image_size[0]), int(image_size[1])
        return int(image_size), int(image_size)

    def _apply_sam3_conditioning(self, images: torch.Tensor) -> torch.Tensor:
        if (not self.mm_use_sam3_conditioning) or images.ndim != 4 or images.shape[1] != 3:
            return images

        self.load_sam3_model()

        input_dtype = images.dtype
        images_fp32 = images.to(dtype=torch.float32)

        clip_mean = torch.tensor(self.image_processor.image_mean, device=images_fp32.device, dtype=images_fp32.dtype).view(1, 3, 1, 1)
        clip_std = torch.tensor(self.image_processor.image_std, device=images_fp32.device, dtype=images_fp32.dtype).view(1, 3, 1, 1)
        rgb_images = (images_fp32 * clip_std + clip_mean).clamp(0.0, 1.0)

        sam3_height, sam3_width = self._sam3_input_size()
        sam3_pixels = F.interpolate(
            rgb_images,
            size=(sam3_height, sam3_width),
            mode="bilinear",
            align_corners=False,
        )

        sam3_mean = torch.tensor(self.sam3_image_processor.image_mean, device=sam3_pixels.device, dtype=sam3_pixels.dtype).view(1, 3, 1, 1)
        sam3_std = torch.tensor(self.sam3_image_processor.image_std, device=sam3_pixels.device, dtype=sam3_pixels.dtype).view(1, 3, 1, 1)
        sam3_pixels = (sam3_pixels - sam3_mean) / sam3_std

        sam3_param = next(self.sam3_vision_tower.parameters())
        sam3_outputs = self.sam3_vision_tower(
            pixel_values=sam3_pixels.to(device=sam3_param.device, dtype=sam3_param.dtype),
            return_dict=True,
        )
        sam3_features = sam3_outputs.last_hidden_state.to(device=images_fp32.device, dtype=torch.float32)
        saliency = sam3_features.norm(dim=-1)

        num_tokens = saliency.shape[1]
        grid_size = int(num_tokens ** 0.5)
        if grid_size * grid_size != num_tokens:
            warnings.warn(
                "SAM3 conditioning expected a square token grid; skipping conditioning. "
                f"num_tokens={num_tokens}"
            )
            return images

        saliency = saliency.view(saliency.shape[0], 1, grid_size, grid_size)
        saliency = saliency - saliency.amin(dim=(2, 3), keepdim=True)
        saliency = saliency / saliency.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        if self.mm_sam3_mask_gamma != 1.0:
            saliency = saliency.clamp_min(1e-6).pow(self.mm_sam3_mask_gamma)

        saliency = F.interpolate(
            saliency,
            size=rgb_images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        alpha = max(0.0, min(1.0, self.mm_sam3_blend_alpha))
        focused_rgb = rgb_images * (0.5 + 0.5 * saliency)
        fused_rgb = ((1.0 - alpha) * rgb_images + alpha * focused_rgb).clamp(0.0, 1.0)
        fused_normalized = (fused_rgb - clip_mean) / clip_std
        return fused_normalized.to(dtype=input_dtype)

    def _apply_image_conditioning(self, images: torch.Tensor) -> torch.Tensor:
        try:
            return self._apply_sam3_conditioning(images)
        finally:
            if self.mm_sam3_unload_after_forward:
                self.unload_sam3_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_input = self._apply_image_conditioning(image.unsqueeze(0))
                image_forward_out = self.vision_tower(image_input.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_input = self._apply_image_conditioning(images)
            image_forward_outs = self.vision_tower(image_input.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_input = self._apply_image_conditioning(images)
        image_forward_outs = self.vision_tower(image_input.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
