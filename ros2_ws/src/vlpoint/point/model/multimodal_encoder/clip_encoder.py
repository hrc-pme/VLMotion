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
        self.tune_sam3_fusion = getattr(args, 'tune_sam3_fusion', False)
        self.mm_sam3_candidate_attention = getattr(
            args, 'mm_sam3_candidate_attention', True
        )
        self.mm_sam3_candidate_loss_weight = float(getattr(
            args, 'mm_sam3_candidate_loss_weight', 0.2
        ))
        self.candidate_attention_loss = None
        self.sam3_is_loaded = False

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        # This adapter is the only trainable part of the two vision backbones.
        # It learns how SAM3 object features and absolute 2-D patch coordinates
        # should change CLIP's semantic patch representation.
        hidden_size = self.config.hidden_size
        self.sam3_fusion = nn.Sequential(
            nn.LayerNorm(hidden_size * 2 + 4),
            nn.Linear(hidden_size * 2 + 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.sam3_fusion_gate = nn.Parameter(
            torch.tensor(float(self.mm_sam3_blend_alpha))
        )
        # A soft, trainable proposal map over SAM3 object-aware patches.  This
        # is produced inside every forward pass from the original image; no
        # pre-cropped candidates or candidate labels are needed in the data.
        candidate_hidden = max(64, hidden_size // 4)
        self.sam3_candidate_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, candidate_hidden),
            nn.GELU(),
            nn.Linear(candidate_hidden, 1),
        )
        # Start close to the original CLIP path while keeping a live gradient
        # route into candidate attention from the first effective update.
        nn.init.normal_(self.sam3_fusion[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.sam3_fusion[-1].bias)
        nn.init.zeros_(self.sam3_candidate_head[-1].weight)
        nn.init.zeros_(self.sam3_candidate_head[-1].bias)
        self.sam3_fusion.requires_grad_(self.tune_sam3_fusion)
        self.sam3_fusion_gate.requires_grad_(self.tune_sam3_fusion)
        self.sam3_candidate_head.requires_grad_(self.tune_sam3_fusion)

    def ensure_floating_sam3_fusion(self, device=None, dtype=None):
        """Undo bitsandbytes conversion while preserving learned adapter weights."""
        try:
            import bitsandbytes as bnb
        except ImportError:
            bnb = None

        def restore(module, target_dtype):
            for name, child in list(module.named_children()):
                if bnb is not None and isinstance(child, bnb.nn.Linear4bit):
                    quant_state = getattr(child.weight, "quant_state", None)
                    if quant_state is None:
                        # Transformers replaces Linear modules before loading
                        # checkpoint tensors. SAM3 adapters do not exist in the
                        # RoboPoint base, so their newly initialized Params4bit
                        # still contain ordinary floating-point values and have
                        # no quantization metadata. Preserve those values
                        # directly instead of trying to dequantize them.
                        if not child.weight.data.is_floating_point():
                            raise RuntimeError(
                                f"Cannot restore SAM3 adapter {name}: missing "
                                "quant_state on a non-floating weight"
                            )
                        weight = child.weight.data.to(
                            device=device, dtype=target_dtype
                        )
                    else:
                        weight = bnb.functional.dequantize_4bit(
                            child.weight.data, quant_state
                        ).to(device=device, dtype=target_dtype)
                    replacement = nn.Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        device=device,
                        dtype=target_dtype,
                    )
                    with torch.no_grad():
                        replacement.weight.copy_(weight)
                        if child.bias is not None:
                            replacement.bias.copy_(
                                child.bias.to(device=device, dtype=target_dtype)
                            )
                    setattr(module, name, replacement)
                else:
                    restore(child, target_dtype)

        fusion_dtype = dtype or torch.float32
        restore(self.sam3_fusion, fusion_dtype)
        restore(self.sam3_candidate_head, torch.float32)
        self.sam3_fusion.to(device=device, dtype=fusion_dtype)
        # Keep this small scoring head in FP32. Its early gradients are much
        # smaller than BF16 resolution while fusion warms up.
        self.sam3_candidate_head.to(device=device, dtype=torch.float32)
        self.sam3_fusion.requires_grad_(self.tune_sam3_fusion)
        self.sam3_candidate_head.requires_grad_(self.tune_sam3_fusion)

    def load_sam3_model(self):
        if self.sam3_is_loaded:
            return

        if not self.mm_sam3_vision_tower:
            raise ValueError(
                "mm_sam3_vision_tower must be set when mm_use_sam3_conditioning=True."
            )

        try:
            from .sam3_encoder import Sam3VisionTower
        except ImportError as exc:
            raise ImportError(
                "Transformers with SAM3 support is required when "
                "mm_use_sam3_conditioning=True."
            ) from exc

        self.sam3_tower = Sam3VisionTower(
            self.mm_sam3_vision_tower,
            args=self,
            delay_load=False,
        )
        self.sam3_image_processor = self.sam3_tower.image_processor
        self.sam3_tower.vision_tower.requires_grad_(False)
        self.sam3_tower.vision_tower.eval()
        sam3_device = self.device if self.mm_sam3_device in (None, '', 'vision', 'clip') else torch.device(self.mm_sam3_device)
        sam3_dtype = self._resolve_sam3_dtype(sam3_device)
        self.sam3_tower.to(device=sam3_device, dtype=sam3_dtype)
        self.sam3_is_loaded = True

    def unload_sam3_model(self):
        if not self.sam3_is_loaded:
            return
        if hasattr(self, "sam3_tower"):
            del self.sam3_tower
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

        backbone_config = getattr(self.sam3_tower.vision_tower.config, "backbone_config", None)
        image_size = getattr(backbone_config, "image_size", 1008)
        if isinstance(image_size, (tuple, list)):
            return int(image_size[0]), int(image_size[1])
        return int(image_size), int(image_size)

    def _sam3_features_for_clip(self, images: torch.Tensor, clip_features: torch.Tensor):
        """Return SAM3 object features aligned with CLIP's patch grid."""
        if (not self.mm_use_sam3_conditioning) or images.ndim != 4 or images.shape[1] != 3:
            return None

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

        sam3_param = next(self.sam3_tower.vision_tower.parameters())
        # SAM3 stays frozen. Gradients are required only through sam3_fusion.
        with torch.no_grad():
            sam3_outputs = self.sam3_tower.vision_tower(
                pixel_values=sam3_pixels.to(device=sam3_param.device, dtype=sam3_param.dtype),
                return_dict=True,
            )
        sam3_features = sam3_outputs.last_hidden_state.to(device=images_fp32.device, dtype=torch.float32)

        sam_tokens = sam3_features.shape[1]
        sam_grid = int(sam_tokens ** 0.5)
        clip_tokens = clip_features.shape[1]
        clip_grid = int(clip_tokens ** 0.5)
        if sam_grid * sam_grid != sam_tokens or clip_grid * clip_grid != clip_tokens:
            warnings.warn(
                "SAM3/CLIP fusion expected square patch grids; skipping fusion. "
                f"sam_tokens={sam_tokens}, clip_tokens={clip_tokens}"
            )
            return None

        # Align SAM3's spatial grid with CLIP. Channel interpolation is a
        # parameter-free bridge; the trainable adapter below learns the useful
        # combination without having to unfreeze either large backbone.
        sam3_features = sam3_features.transpose(1, 2).reshape(
            sam3_features.shape[0], sam3_features.shape[2], sam_grid, sam_grid
        )
        sam3_features = F.interpolate(
            sam3_features,
            size=(clip_grid, clip_grid),
            mode="bilinear",
            align_corners=False,
        ).flatten(2).transpose(1, 2)
        sam3_features = F.interpolate(
            sam3_features.unsqueeze(1),
            size=(clip_tokens, clip_features.shape[-1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return F.layer_norm(sam3_features, (sam3_features.shape[-1],))

    def _spatial_features(self, batch_size, token_count, device, dtype, image_sizes=None):
        grid = int(token_count ** 0.5)
        if grid * grid != token_count:
            coords = torch.zeros(batch_size, token_count, 4, device=device, dtype=dtype)
            valid = torch.ones(batch_size, token_count, device=device, dtype=torch.bool)
            return coords, valid

        axis = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / grid
        yy_pad, xx_pad = torch.meshgrid(axis, axis, indexing="ij")
        xx_pad = xx_pad.reshape(1, token_count).expand(batch_size, -1)
        yy_pad = yy_pad.reshape(1, token_count).expand(batch_size, -1)
        xx, yy = xx_pad.clone(), yy_pad.clone()
        valid = torch.ones_like(xx, dtype=torch.bool)

        # CLIP receives a square letterboxed image. Convert patch locations
        # back to normalized coordinates in the unpadded original image so
        # portrait training images and square inference images share geometry.
        if image_sizes is not None and len(image_sizes) == batch_size:
            for index, size in enumerate(image_sizes):
                width, height = float(size[0]), float(size[1])
                if width < height:
                    scale = width / height
                    offset = (1.0 - scale) / 2.0
                    valid[index] = (xx_pad[index] >= offset) & (xx_pad[index] <= offset + scale)
                    xx[index] = (xx_pad[index] - offset) / max(scale, 1e-6)
                elif height < width:
                    scale = height / width
                    offset = (1.0 - scale) / 2.0
                    valid[index] = (yy_pad[index] >= offset) & (yy_pad[index] <= offset + scale)
                    yy[index] = (yy_pad[index] - offset) / max(scale, 1e-6)

        xx = xx.clamp(0.0, 1.0)
        yy = yy.clamp(0.0, 1.0)
        x_signed, y_signed = xx * 2.0 - 1.0, yy * 2.0 - 1.0
        coords = torch.stack(
            (x_signed, y_signed, x_signed.square(), y_signed.square()), dim=-1
        ).to(dtype=dtype)
        coords = coords.masked_fill(~valid.unsqueeze(-1), 0)
        return coords, valid

    def _candidate_supervision_loss(self, logits, spatial, valid_patches, targets):
        if targets is None:
            return None
        targets = targets.to(device=logits.device, dtype=torch.float32)
        patch_xy = (spatial[..., :2].to(dtype=torch.float32) + 1.0) / 2.0
        losses = []
        for batch_index in range(logits.shape[0]):
            target = targets[batch_index]
            target = target[(target[:, 0] >= 0) & (target[:, 1] >= 0)]
            if target.numel() == 0:
                continue
            distance_sq = (
                patch_xy[batch_index, :, None, :] - target[None, :, :]
            ).square().sum(dim=-1)
            heatmap = torch.exp(-distance_sq.min(dim=-1).values / (2.0 * 0.04 ** 2))
            per_patch = F.binary_cross_entropy_with_logits(
                logits[batch_index, :, 0].float(), heatmap, reduction="none"
            )
            weights = 1.0 + 8.0 * heatmap
            mask = valid_patches[batch_index].float()
            losses.append((per_patch * weights * mask).sum() / (weights * mask).sum().clamp_min(1.0))
        return torch.stack(losses).mean() if losses else None

    def _fuse_sam3_clip(self, images, clip_features, image_sizes=None, candidate_targets=None):
        self.candidate_attention_loss = None
        if not self.mm_use_sam3_conditioning:
            return clip_features
        try:
            sam_features = self._sam3_features_for_clip(images, clip_features)
            if sam_features is None:
                return clip_features
            work_dtype = self.sam3_fusion[1].weight.dtype
            clip_work = clip_features.to(dtype=work_dtype)
            sam_work = sam_features.to(device=clip_work.device, dtype=work_dtype)
            if self.mm_sam3_candidate_attention:
                candidate_dtype = next(
                    self.sam3_candidate_head.parameters()
                ).dtype
                candidate_logits = self.sam3_candidate_head(
                    sam_work.to(dtype=candidate_dtype)
                )
                candidate_scores = torch.sigmoid(candidate_logits).to(dtype=work_dtype)
                gamma = max(float(self.mm_sam3_mask_gamma), 1e-3)
                candidate_scores = candidate_scores.pow(gamma)
                # The zero-initialized head starts at a neutral multiplier of
                # one. Supervised coordinate loss then learns which SAM3
                # object patches deserve more or less attention.
                sam_work = sam_work * (0.5 + candidate_scores)
                self.last_candidate_attention = candidate_scores.detach()
            spatial, valid_patches = self._spatial_features(
                clip_work.shape[0], clip_work.shape[1], clip_work.device,
                work_dtype, image_sizes=image_sizes
            )
            if self.mm_sam3_candidate_attention:
                self.candidate_attention_loss = self._candidate_supervision_loss(
                    candidate_logits, spatial, valid_patches, candidate_targets
                )
            delta = self.sam3_fusion(torch.cat((clip_work, sam_work, spatial), dim=-1))
            gate = torch.tanh(self.sam3_fusion_gate.to(dtype=work_dtype))
            return (clip_work + gate * delta).to(dtype=clip_features.dtype)
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

    def forward(self, images, image_sizes=None, candidate_targets=None):
        if type(images) is list:
            image_features = []
            for image in images:
                image_input = image.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                with torch.no_grad():
                    image_forward_out = self.vision_tower(image_input, output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_feature = self._fuse_sam3_clip(
                    image.unsqueeze(0), image_feature,
                    image_sizes=None, candidate_targets=None
                )
                image_features.append(image_feature)
        else:
            image_input = images.to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                image_forward_outs = self.vision_tower(image_input, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = self._fuse_sam3_clip(
                images, image_features, image_sizes=image_sizes,
                candidate_targets=candidate_targets
            )

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
