# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import ast
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
import inspect
import string
import random
import re
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from point.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from point.train.llava_trainer import LLaVATrainer

from point import conversation as conversation_lib
from point.model import *
from point.mm_utils import tokenizer_image_token
from point.candidate_contact_sheet import (
    build_candidate_contact_sheet,
    candidate_prompt_lines,
    deduplicate_candidates,
)

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_use_sam3_conditioning: bool = field(default=False)
    mm_sam3_vision_tower: Optional[str] = field(default=None)
    mm_sam3_blend_alpha: float = field(default=0.35)
    mm_sam3_mask_gamma: float = field(default=1.0)
    mm_sam3_device: str = field(default="cpu")
    mm_sam3_dtype: str = field(default="auto")
    mm_sam3_candidate_attention: bool = field(default=True)
    mm_sam3_candidate_loss_weight: float = field(default=0.2)
    tune_sam3_fusion: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    sam3_candidate_cache: Optional[str] = None
    sam3_dynamic_candidate_view: bool = False
    sam3_max_candidates: int = 16
    sam3_intent_focused_targets: bool = False


def supervised_intent_label(text: str) -> str:
    """Build an explicit semantic class target for supervised training only."""
    normalized = text.lower()
    upward = re.search(
        r"\b(?:up|upward|upwards|upstairs|ascend|ascending|rise|rising)\b"
        r"|\bupper floor\b|\bhigher floor\b|\bgo higher\b|\bgoing higher\b",
        normalized,
    )
    downward = re.search(
        r"\b(?:down|downward|downwards|downstairs|descend|descending)\b"
        r"|\blower floor\b",
        normalized,
    )
    if bool(upward) == bool(downward):
        return ""
    return "UPWARD_TRAVEL" if upward else "DOWNWARD_TRAVEL"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def ensure_floating_linear_children(module, device, dtype):
    """Restore trainable adapters accidentally converted to 4-bit layers."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        bnb = None

    converted = 0
    for name, child in list(module.named_children()):
        if bnb is not None and isinstance(child, bnb.nn.Linear4bit):
            quant_state = getattr(child.weight, "quant_state", None)
            if quant_state is None:
                # A module missing from the base checkpoint can be wrapped as
                # Linear4bit before it ever gets quantized. In that case its
                # Params4bit payload is still the freshly initialized floating
                # tensor and can be copied without dequantization.
                if not child.weight.data.is_floating_point():
                    raise RuntimeError(
                        f"Cannot dequantize {name}: quant_state is missing on "
                        "a non-floating weight"
                    )
                weight = child.weight.data.to(device=device, dtype=dtype)
            else:
                weight = bnb.functional.dequantize_4bit(
                    child.weight.data, quant_state
                ).to(device=device, dtype=dtype)
            replacement = torch.nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                replacement.weight.copy_(weight)
                if child.bias is not None:
                    replacement.bias.copy_(
                        child.bias.to(device=device, dtype=dtype)
                    )
            setattr(module, name, replacement)
            converted += 1
        else:
            converted += ensure_floating_linear_children(child, device, dtype)
    return converted


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = [
            'mm_projector', 'sam3_fusion', 'sam3_fusion_gate',
        ]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                # Keep every image token.  The upstream single-image code moved
                # image tokens to the front by deleting all of them and adding
                # one back, which silently collapsed two-image examples.
                image_count = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                image_prefix = '\n'.join([DEFAULT_IMAGE_TOKEN] * image_count)
                sentence['value'] = image_prefix + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.sam3_candidate_cache = None
        if data_args.sam3_dynamic_candidate_view:
            if not data_args.sam3_candidate_cache:
                raise ValueError(
                    "--sam3_candidate_cache is required for dynamic candidate training"
                )
            with open(data_args.sam3_candidate_cache, "r") as candidate_file:
                self.sam3_candidate_cache = json.load(candidate_file)
        # Build an image-level union of all coordinate annotations. For paired
        # up/down records this supervises the visual candidate head to find all
        # valid buttons, while the language loss learns which subset matches
        # the request. Nothing is cropped or written to a derived dataset.
        candidate_targets = {}
        for sample in list_data_dict:
            image_value = sample.get("image")
            if not isinstance(image_value, str):
                continue
            answer = next(
                (turn.get("value", "") for turn in sample.get("conversations", [])
                 if turn.get("from") == "gpt"),
                "",
            )
            try:
                points = ast.literal_eval(answer)
            except (ValueError, SyntaxError):
                continue
            bucket = candidate_targets.setdefault(image_value, set())
            for point in points if isinstance(points, list) else []:
                if isinstance(point, (tuple, list)) and len(point) == 2:
                    bucket.add((round(float(point[0]), 6), round(float(point[1]), 6)))
        self.candidate_targets = {
            key: sorted(value) for key, value in candidate_targets.items()
        }

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            image_value = sample.get('image')
            image_count = len(image_value) if isinstance(image_value, list) else int(image_value is not None)
            img_tokens = 128 * image_count
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_value = self.list_data_dict[i]['image']
            image_files = image_value if isinstance(image_value, list) else [image_value]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                if width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                return result

            raw_images = []
            for image_file in image_files:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                raw_images.append(image)

            candidate_detections = None
            if self.sam3_candidate_cache is not None and isinstance(image_value, str):
                candidate_detections = deduplicate_candidates(copy.deepcopy(
                    self.sam3_candidate_cache.get(image_value, [])
                ))[:self.data_args.sam3_max_candidates]
                # Confidence rank is correlated with labels in this small
                # dataset. Shuffle every access so candidate letters cannot
                # become a shortcut for direction or operability.
                random.shuffle(candidate_detections)
                for candidate_index, detection in enumerate(candidate_detections):
                    detection["label"] = (
                        string.ascii_uppercase[candidate_index]
                        if candidate_index < len(string.ascii_uppercase)
                        else f"P{candidate_index + 1}"
                    )
                if candidate_detections:
                    raw_images.append(build_candidate_contact_sheet(
                        raw_images[0], candidate_detections
                    ))

            processed_images = []
            original_sizes = [image.size for image in raw_images]
            for image in raw_images:
                if self.data_args.image_aspect_ratio == 'pad':
                    image = expand2square(
                        image, tuple(int(x * 255) for x in processor.image_mean)
                    )
                processed_images.append(
                    processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                )
            image = (
                processed_images
                if isinstance(image_value, list) or candidate_detections
                else processed_images[0]
            )
            conversations = copy.deepcopy([e["conversations"] for e in sources])
            if candidate_detections:
                # Supervise the exact centers produced by SAM3.  The original
                # annotations determine which candidates are correct and how
                # many must be returned; snapping only removes coordinate
                # regression noise between annotation centers and detections.
                gpt_turn = next(
                    turn for turn in conversations[0] if turn.get("from") == "gpt"
                )
                try:
                    annotated_points = ast.literal_eval(gpt_turn["value"])
                except (ValueError, SyntaxError):
                    annotated_points = []
                snapped_centers = []
                selected_candidate_labels = []
                for point in annotated_points if isinstance(annotated_points, list) else []:
                    if not isinstance(point, (tuple, list)) or len(point) != 2:
                        continue
                    nearest = min(
                        candidate_detections,
                        key=lambda detection: (
                            (detection["center"][0] - float(point[0])) ** 2
                            + (detection["center"][1] - float(point[1])) ** 2
                        ),
                    )
                    center = tuple(round(float(value), 4) for value in nearest["center"])
                    if center not in snapped_centers:
                        snapped_centers.append(center)
                        selected_candidate_labels.append(nearest["label"])

                operable_candidate_labels = []
                for point in self.candidate_targets.get(image_value, []):
                    nearest = min(
                        candidate_detections,
                        key=lambda detection: (
                            (detection["center"][0] - float(point[0])) ** 2
                            + (detection["center"][1] - float(point[1])) ** 2
                        ),
                    )
                    if nearest["label"] not in operable_candidate_labels:
                        operable_candidate_labels.append(nearest["label"])
                all_candidate_labels = [
                    detection["label"] for detection in candidate_detections
                ]
                non_operable_candidate_labels = [
                    label for label in all_candidate_labels
                    if label not in operable_candidate_labels
                ]
                request_mismatch_labels = [
                    label for label in operable_candidate_labels
                    if label not in selected_candidate_labels
                ]

                def render_labels(labels):
                    return ", ".join(labels) if labels else "none"

                if snapped_centers:
                    if self.data_args.sam3_intent_focused_targets:
                        human_text = next(
                            turn.get("value", "") for turn in conversations[0]
                            if turn.get("from") == "human"
                        )
                        intent_label = supervised_intent_label(human_text)
                        if not intent_label:
                            raise ValueError(
                                "Intent-focused training requires an explicit semantic "
                                f"label for {image_value!r}: {human_text!r}"
                            )
                        # The SAM3 auxiliary loss already supervises all
                        # operable controls. Keep the language target compact
                        # so request-dependent tokens dominate its loss.
                        gpt_turn["value"] = (
                            "Interpreted intent: "
                            + intent_label
                            + "\nSelected candidates: "
                            + render_labels(selected_candidate_labels)
                            + "\nFinal coordinates: "
                            + str(snapped_centers)
                        )
                    else:
                        gpt_turn["value"] = (
                            "Operable candidates: "
                            + render_labels(operable_candidate_labels)
                            + "\nNon-operable candidates: "
                            + render_labels(non_operable_candidate_labels)
                            + "\nOperable but not matched to this request: "
                            + render_labels(request_mismatch_labels)
                            + "\nSelected candidates: "
                            + render_labels(selected_candidate_labels)
                            + "\nFinal coordinates: "
                            + str(snapped_centers)
                        )

                human = next(
                    turn for turn in conversations[0] if turn.get("from") == "human"
                )
                human["value"] = human["value"].replace(
                    DEFAULT_IMAGE_TOKEN,
                    DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN,
                    1,
                )
                human["value"] += (
                    "\nThe first image is the complete original scene. The second image "
                    "contains enlarged SAM3 object candidates. Candidate centers in the "
                    "original image are:\n"
                    + "\n".join(candidate_prompt_lines(candidate_detections))
                    + "\nUse the user's request and visible candidate features to choose "
                    "the correct candidates. First identify which candidates are real, "
                    "physically operable controls rather than displays, labels, or decorative "
                    "objects, then select only operable controls whose visible meaning "
                    "matches the user's request. The selected count is determined by the "
                    "image and is not fixed."
                )
                if self.data_args.sam3_intent_focused_targets:
                    human["value"] += (
                        "\nFirst interpret the requested action semantically. The same scene "
                        "paired with a different request must be allowed to select different "
                        "candidates. Respond as:\nInterpreted intent: UPWARD_TRAVEL or "
                        "DOWNWARD_TRAVEL\nSelected candidates: B, ...\n"
                        "Final coordinates: [(x1, y1), ...]"
                    )
                else:
                    human["value"] += (
                        " Classify every candidate. Respond as:\n"
                        "Operable candidates: A, ...\nNon-operable candidates: C, ...\n"
                        "Operable but not matched to this request: D, ...\n"
                        "Selected candidates: B, ...\n"
                        "Final coordinates: [(x1, y1), ...]"
                    )
            sources = preprocess_multimodal(conversations, self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = (
                original_sizes if isinstance(image, list) else original_sizes[0]
            )
            if isinstance(image_value, str):
                data_dict['candidate_targets'] = torch.tensor(
                    self.candidate_targets.get(image_value, []), dtype=torch.float32
                ).reshape(-1, 2)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if isinstance(images[0], list):
                # llava_arch consumes image features in token order across the
                # batch, so preserve each example's ordered image list.
                batch['images'] = [image for group in images for image in group]
                batch['image_sizes'] = [
                    size for instance in instances for size in instance.get('image_size')
                ]
            elif all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
                batch['image_sizes'] = [instance.get('image_size') for instance in instances]
            else:
                batch['images'] = images
                batch['image_sizes'] = [instance.get('image_size') for instance in instances]

            if 'candidate_targets' in instances[0]:
                target_groups = []
                if isinstance(images[0], list):
                    for instance, image_group in zip(instances, images):
                        target_groups.append(instance['candidate_targets'])
                        target_groups.extend([
                            torch.empty(0, 2, dtype=torch.float32)
                            for _ in image_group[1:]
                        ])
                else:
                    target_groups = [x['candidate_targets'] for x in instances]
                max_targets = max(x.shape[0] for x in target_groups)
                padded_targets = torch.full(
                    (len(target_groups), max_targets, 2), -1.0, dtype=torch.float32
                )
                for index, targets in enumerate(target_groups):
                    count = targets.shape[0]
                    if count:
                        padded_targets[index, :count] = targets
                batch['candidate_targets'] = padded_targets

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                # Keep the trainable multimodal layers in a normal floating
                # dtype. bitsandbytes Linear4bit modules are intended for
                # frozen pretrained weights, not newly initialized adapters.
                llm_int8_skip_modules=[
                    "mm_projector", "sam3_fusion", "sam3_candidate_head"
                ],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                low_cpu_mem_usage=True,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=True,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )
            model.config.pad_token_id = tokenizer.pad_token_id
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        if hasattr(vision_tower, "ensure_floating_sam3_fusion"):
            vision_tower.ensure_floating_sam3_fusion(
                device=training_args.device,
                dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            )
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        if hasattr(vision_tower, "sam3_candidate_head"):
            vision_tower.sam3_candidate_head.to(
                device=training_args.device, dtype=torch.float32
            )

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            # PEFT has already marked only LoRA parameters trainable. Do not
            # freeze them again when jointly training the multimodal adapter.
            if not training_args.lora_enable:
                model.requires_grad_(False)
            mm_projector = model.get_model().mm_projector
            converted_layers = ensure_floating_linear_children(
                mm_projector,
                device=training_args.device,
                dtype=compute_dtype,
            )
            if converted_layers:
                rank0_print(
                    f"Restored {converted_layers} quantized mm_projector layer(s) "
                    f"to trainable {compute_dtype}."
                )
            for p in mm_projector.parameters():
                if not (p.is_floating_point() or p.is_complex()):
                    raise RuntimeError(
                        "mm_projector still contains non-floating parameters"
                    )
                p.requires_grad = True

        # SAM3 and CLIP remain frozen; only the small cross-vision spatial
        # adapter learns from the supervised relation examples.
        if model_args.mm_use_sam3_conditioning and model_args.tune_sam3_fusion:
            # from_pretrained initializes missing standalone parameters after
            # constructing the model and can reset this new gate to zero. A
            # A zero gate blocks the gradient path into fusion, so restore the
            # requested nonzero starting value before optimization.
            with torch.no_grad():
                vision_tower.sam3_fusion_gate.fill_(
                    model_args.mm_sam3_blend_alpha
                )
            for p in vision_tower.sam3_fusion.parameters():
                p.requires_grad = True
            for p in vision_tower.sam3_candidate_head.parameters():
                p.requires_grad = True
            vision_tower.sam3_fusion_gate.requires_grad = True
        model.config.tune_sam3_fusion = model_args.tune_sam3_fusion
        model.config.mm_sam3_candidate_loss_weight = (
            model_args.mm_sam3_candidate_loss_weight
        )
        training_args.tune_sam3_fusion = model_args.tune_sam3_fusion

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer_kwargs = dict(model=model, args=training_args, **data_module)
    if "processing_class" in inspect.signature(LLaVATrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = LLaVATrainer(**trainer_kwargs)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
    with open(os.path.join(training_args.output_dir, 'done.md'), 'w') as f:
        f.write("done")


if __name__ == "__main__":
    train()
