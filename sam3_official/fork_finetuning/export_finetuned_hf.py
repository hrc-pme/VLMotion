#!/usr/bin/env python3
"""Overlay native SAM3 image fine-tuning weights onto an HF SAM3 model."""

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import Sam3Model, Sam3Processor

from convert_sam3_to_hf import convert_old_keys_to_new_keys, split_qkv


IGNORED_HF_KEYS = {
    "geometry_encoder.points_direct_project.bias",
    "geometry_encoder.points_direct_project.weight",
    "geometry_encoder.points_pool_project.bias",
    "geometry_encoder.points_pool_project.weight",
    "geometry_encoder.points_pos_enc_project.bias",
    "geometry_encoder.points_pos_enc_project.weight",
}
ROPE_BUFFER_PATTERN = re.compile(
    r"^vision_encoder\.backbone\.layers\.\d+\.rotary_emb\.rope_embeddings$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="facebook/sam3")
    parser.add_argument(
        "--allow-mask-decoder-update",
        action="store_true",
        help="Allow and require fine-tuned segmentation weights to update the HF mask decoder.",
    )
    return parser.parse_args()


def tensor_fingerprint(state_dict, prefix):
    digest = hashlib.sha256()
    matched = 0
    for key in sorted(state_dict):
        if not key.startswith(prefix):
            continue
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        matched += 1
    if matched == 0:
        raise ValueError(f"No tensors matched prefix {prefix!r}")
    return digest.hexdigest(), matched


def convert_finetuned_state_dict(native_state):
    mapping = convert_old_keys_to_new_keys(list(native_state))
    converted = {}
    for old_key, tensor in native_state.items():
        new_key = mapping[old_key]
        if new_key == "vision_encoder.backbone.embeddings.position_embeddings":
            tensor = tensor[:, 1:, :]
        converted[new_key] = tensor

    converted = split_qkv(converted)
    projection_key = "text_encoder.text_projection.weight"
    if projection_key in converted:
        converted[projection_key] = converted[projection_key].T
    return converted


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF base model: {args.base_model}")
    model = Sam3Model.from_pretrained(args.base_model, local_files_only=True)
    model.eval()
    model_state = model.state_dict()
    mask_before, mask_tensor_count = tensor_fingerprint(model_state, "mask_decoder.")

    print(f"Loading fine-tuned checkpoint: {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    native_state = checkpoint["model"]
    native_tensor_count = len(native_state)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    checkpoint_steps = checkpoint.get("steps", {})
    del checkpoint

    converted = convert_finetuned_state_dict(native_state)
    del native_state
    gc.collect()

    all_converted_tensor_count = len(converted)
    non_hf_keys = sorted(set(converted) - set(model_state))
    ignored_keys = [
        key for key in non_hf_keys
        if key in IGNORED_HF_KEYS or ROPE_BUFFER_PATTERN.fullmatch(key)
    ]
    unexpected_converted = sorted(set(non_hf_keys) - set(ignored_keys))
    for key in ignored_keys:
        del converted[key]
    shape_mismatches = [
        {
            "key": key,
            "fine_tuned": list(tensor.shape),
            "base": list(model_state[key].shape),
        }
        for key, tensor in converted.items()
        if key in model_state and tensor.shape != model_state[key].shape
    ]
    if unexpected_converted or shape_mismatches:
        raise ValueError(
            f"Conversion mismatch: unexpected={unexpected_converted[:10]} "
            f"shape_mismatches={shape_mismatches[:10]}"
        )

    changed_keys = [
        key for key, tensor in converted.items()
        if not torch.equal(model_state[key], tensor)
    ]
    changed_parameters = sum(converted[key].numel() for key in changed_keys)
    if not changed_keys:
        raise ValueError("Fine-tuned checkpoint is identical to the HF base model")

    incompatible = model.load_state_dict(converted, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected load keys: {incompatible.unexpected_keys}")

    output_state = model.state_dict()
    mask_after, _ = tensor_fingerprint(output_state, "mask_decoder.")
    mask_decoder_changed = mask_before != mask_after
    changed_mask_keys = [
        key for key in changed_keys if key.startswith("mask_decoder.")
    ]
    if mask_decoder_changed and not args.allow_mask_decoder_update:
        raise ValueError("Base mask decoder changed during detector weight overlay")
    if args.allow_mask_decoder_update and not changed_mask_keys:
        raise ValueError(
            "Segmentation export requested, but no HF mask decoder tensors changed"
        )

    verification_keys = sorted(converted)[::max(1, len(converted) // 32)]
    failed_verification = [
        key for key in verification_keys
        if not torch.equal(output_state[key], converted[key])
    ]
    if failed_verification:
        raise ValueError(f"Loaded tensor verification failed: {failed_verification}")

    print(f"Saving local HF model: {args.output_dir}")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    processor = Sam3Processor.from_pretrained(
        args.base_model,
        local_files_only=True,
    )
    processor.save_pretrained(args.output_dir)

    manifest = {
        "base_model": args.base_model,
        "fine_tuned_checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_steps": checkpoint_steps,
        "native_fine_tuned_tensor_count": native_tensor_count,
        "converted_tensor_count_before_hf_filter": all_converted_tensor_count,
        "loaded_hf_tensor_count": len(converted),
        "ignored_non_hf_tensor_count": len(ignored_keys),
        "ignored_non_hf_tensors": ignored_keys,
        "changed_tensor_count": len(changed_keys),
        "changed_parameter_count": changed_parameters,
        "changed_mask_decoder_tensor_count": len(changed_mask_keys),
        "mask_decoder_updated": mask_decoder_changed,
        "preserved_tensor_count": len(incompatible.missing_keys),
        "mask_decoder_tensor_count": mask_tensor_count,
        "mask_decoder_sha256": mask_after,
        "unexpected_key_count": 0,
        "shape_mismatch_count": 0,
        "verification_tensor_count": len(verification_keys),
    }
    (args.output_dir / "finetuning_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
