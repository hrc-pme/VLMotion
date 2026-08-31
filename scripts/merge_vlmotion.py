#!/usr/bin/env python3
"""Merge the RoboPoint base with VLMotion LoRA and visual fusion weights."""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from point.model.language_model.llava_llama import (
    LlavaConfig,
    LlavaLlamaForCausalLM,
)


def normalize_non_lora_keys(state_dict):
    normalized = {
        (key[11:] if key.startswith("base_model.") else key): value
        for key, value in state_dict.items()
    }
    if any(key.startswith("model.model.") for key in normalized):
        normalized = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in normalized.items()
        }
    return normalized


def verify_fusion(state_dict, require_candidate_head=False):
    gate_key = next(
        (key for key in state_dict if key.endswith("sam3_fusion_gate")), None
    )
    final_key = next(
        (key for key in state_dict if key.endswith("sam3_fusion.3.weight")), None
    )
    if gate_key is None or final_key is None:
        raise RuntimeError("SAM3 fusion weights are missing from non_lora_trainables.bin")

    gate = state_dict[gate_key].float()
    final_weight = state_dict[final_key].float()
    if gate.abs().max().item() == 0 or torch.count_nonzero(final_weight).item() == 0:
        raise RuntimeError(
            "SAM3 fusion is inactive (zero gate or zero final layer); refusing to merge."
        )
    print(
        "SAM3 fusion check OK: "
        f"gate={gate.item():.6f}, final_norm={final_weight.norm().item():.6f}, "
        f"final_nonzero={torch.count_nonzero(final_weight).item()}/{final_weight.numel()}"
    )
    if require_candidate_head:
        candidate_key = next(
            (key for key in state_dict if key.endswith("sam3_candidate_head.3.weight")),
            None,
        )
        if candidate_key is None:
            raise RuntimeError("SAM3 candidate-attention weights are missing")
        candidate_weight = state_dict[candidate_key].float()
        if torch.count_nonzero(candidate_weight).item() == 0:
            raise RuntimeError("SAM3 candidate-attention head did not learn; refusing to merge")
        print(
            "SAM3 candidate attention check OK: "
            f"norm={candidate_weight.norm().item():.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
    )
    parser.add_argument(
        "--adapter", type=Path, default=Path("/workspace/checkpoints/vlmotion")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/workspace/checkpoints/vlmotion-merged")
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--require-candidate-head", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to merge this 13B model on this machine.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Output is not empty: {args.output}. Use a new path to avoid overwriting."
        )

    adapter_config = args.adapter / "adapter_config.json"
    non_lora_path = args.adapter / "non_lora_trainables.bin"
    if not adapter_config.is_file():
        raise FileNotFoundError(adapter_config)
    if not non_lora_path.is_file():
        raise FileNotFoundError(non_lora_path)

    non_lora = torch.load(non_lora_path, map_location="cpu", weights_only=False)
    verify_fusion(non_lora, require_candidate_head=args.require_candidate_head)

    config = LlavaConfig.from_pretrained(args.adapter)
    # This records QLoRA training-time loading; the merged output is FP16.
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")

    print(f"Loading FP16 base model on GPU: {args.base_model}")
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )

    print("Applying SAM3 fusion and other non-LoRA trainables...")
    incompatible = model.load_state_dict(
        normalize_non_lora_keys(non_lora), strict=False
    )
    unexpected_fusion = [
        key for key in incompatible.unexpected_keys if "sam3_fusion" in key
    ]
    if unexpected_fusion:
        raise RuntimeError(f"Fusion keys did not load: {unexpected_fusion}")

    print("Loading and merging LoRA...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload(safe_merge=True)
    model.config._name_or_path = str(args.output)
    model.config.use_cache = True

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged FP16 model to: {args.output}")
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.save_pretrained(args.output)

    marker = args.output / "MERGED_OK"
    marker.write_text(
        f"base={args.base_model}\nadapter={args.adapter}\n",
        encoding="utf-8",
    )
    total_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    print(f"Merge complete: {args.output} ({total_bytes / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()
