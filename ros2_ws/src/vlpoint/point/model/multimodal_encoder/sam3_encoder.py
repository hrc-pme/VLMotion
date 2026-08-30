import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    Sam3VisionModel,
    Sam3ImageProcessor,
    Sam3VisionConfig,
)


class Sam3VisionTower(nn.Module):
    """
    SAM3 vision encoder wrapper for multimodal projects.

    Expected output from Sam3VisionModel:
        last_hidden_state: [B, N, C]
    where:
        B = batch size
        N = number of image patches
        C = hidden size

    Notes:
        1. Sam3VisionModel is vision-only.
        2. If you need text-conditioned segmentation, use Sam3Model instead.
        3. This wrapper is for extracting visual features to feed into a projector / LLM.
    """

    def __init__(self, vision_tower_name: str, args=None, delay_load: bool = False):
        super().__init__()

        self.vision_tower_name = vision_tower_name
        self.is_loaded = False

        # Common knobs used in VLM codebases
        self.select_layer = getattr(args, "mm_vision_select_layer", -1) if args is not None else -1
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch") if args is not None else "patch"
        self.freeze_vision_tower = getattr(args, "freeze_vision_tower", True) if args is not None else True

        if not delay_load:
            self.load_model()
        else:
            # light-weight config-only path
            self.cfg_only = Sam3VisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        self.image_processor = Sam3ImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = Sam3VisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
        )

        if self.freeze_vision_tower:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()

        self.is_loaded = True

    @property
    def dtype(self):
        if self.is_loaded:
            return self.vision_tower.dtype
        return torch.float32

    @property
    def device(self):
        if self.is_loaded:
            return next(self.vision_tower.parameters()).device
        return torch.device("cpu")

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self.cfg_only

    @property
    def hidden_size(self):
        backbone_config = getattr(self.config, "backbone_config", None)
        if backbone_config is not None and hasattr(backbone_config, "hidden_size"):
            return backbone_config.hidden_size
        return self.config.hidden_size

    @property
    def image_size(self):
        backbone_config = getattr(self.config, "backbone_config", None)
        if backbone_config is not None and hasattr(backbone_config, "image_size"):
            return backbone_config.image_size
        return self.config.image_size

    @property
    def patch_size(self):
        backbone_config = getattr(self.config, "backbone_config", None)
        if backbone_config is not None and hasattr(backbone_config, "patch_size"):
            return backbone_config.patch_size
        return self.config.patch_size

    @property
    def num_patches(self):
        """
        Number of patch tokens for a square image after processor resize/pad.
        """
        image_size = self.image_size
        patch_size = self.patch_size
        if isinstance(image_size, (tuple, list)):
            h, w = image_size
        else:
            h = w = image_size

        if isinstance(patch_size, (tuple, list)):
            ph, pw = patch_size
        else:
            ph = pw = patch_size

        return (h // ph) * (w // pw)

    @property
    def num_patches_per_side(self):
        image_size = self.image_size
        patch_size = self.patch_size

        if isinstance(image_size, (tuple, list)):
            h, w = image_size
            if h != w:
                raise ValueError("num_patches_per_side is only defined for square SAM3 inputs.")
            image_size = h

        if isinstance(patch_size, (tuple, list)):
            ph, pw = patch_size
            if ph != pw:
                raise ValueError("num_patches_per_side is only defined for square SAM3 patches.")
            patch_size = ph

        return image_size // patch_size

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    def _ensure_loaded(self):
        if not self.is_loaded:
            self.load_model()

    def _prepare_images(
        self,
        images: Union[
            torch.Tensor,
            Image.Image,
            List[Image.Image],
            List[torch.Tensor],
        ],
    ) -> torch.Tensor:
        """
        Convert raw images into pixel_values expected by Sam3VisionModel.

        Accepted inputs:
            - preprocessed tensor [B, 3, H, W]
            - single PIL image
            - list[PIL.Image]
            - list[Tensor(C,H,W)]  (will be stacked if shapes match)
        """
        self._ensure_loaded()

        if isinstance(images, torch.Tensor):
            # Assume already pixel_values or image tensor batch
            if images.ndim == 3:
                images = images.unsqueeze(0)
            return images

        if isinstance(images, Image.Image):
            images = [images]

        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], torch.Tensor):
            if images[0].ndim == 3:
                return torch.stack(images, dim=0)
            raise ValueError("Tensor list must contain tensors with shape [C, H, W].")

        encoded = self.image_processor(images=images, return_tensors="pt")
        return encoded["pixel_values"]

    def feature_select(self, image_forward_outs):
        """
        Select and post-process features from vision outputs.

        Sam3VisionModel returns BaseModelOutput with:
            last_hidden_state: [B, N, C]

        select_feature:
            - 'patch': returns [B, N, C]
            - 'cls_patch': same as patch for SAM3 (no cls token)
            - 'mean': returns [B, 1, C]
        """
        image_features = image_forward_outs.last_hidden_state

        if self.select_feature in ("patch", "cls_patch"):
            return image_features
        elif self.select_feature == "mean":
            return image_features.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

    @torch.no_grad()
    def forward(
        self,
        images: Union[
            torch.Tensor,
            Image.Image,
            List[Image.Image],
            List[torch.Tensor],
        ],
    ) -> torch.Tensor:
        """
        Returns:
            image_features: [B, N, C] or [B, 1, C]
        """
        self._ensure_loaded()

        pixel_values = self._prepare_images(images)
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        image_forward_outs = self.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        image_features = self.feature_select(image_forward_outs)
        return image_features

    def forward_with_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Use this when caller already has pixel_values from Sam3ImageProcessor.
        """
        self._ensure_loaded()
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        outputs = self.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        return self.feature_select(outputs)
