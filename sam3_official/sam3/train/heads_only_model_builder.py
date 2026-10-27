"""SAM3 model builders for small-data, task-head-only fine-tuning."""

import logging

from sam3.model_builder import build_sam3_image_model


_HEAD_PREFIXES = (
    "class_embed.",
    "instance_class_embed.",
    "dot_prod_scoring.",
    "transformer.decoder.bbox_embed.",
    "transformer.decoder.instance_bbox_embed.",
    "transformer.decoder.presence_token",
    "segmentation_head.",
)

_CONCEPT_PROMPT_PREFIXES = (
    "dot_prod_scoring.prompt_mlp.",
    "dot_prod_scoring.prompt_proj.",
    "instance_dot_prod_scoring.prompt_mlp.",
    "instance_dot_prod_scoring.prompt_proj.",
)

_CONCEPT_QUERY_SCORING_PREFIXES = _CONCEPT_PROMPT_PREFIXES + (
    "dot_prod_scoring.hs_proj.",
    "instance_dot_prod_scoring.hs_proj.",
)

_LAST_DECODER_O2O_PREFIXES = (
    "transformer.decoder.layers.5.",
    "transformer.decoder.bbox_embed.",
    "transformer.decoder.instance_bbox_embed.",
    "dot_prod_scoring.hs_proj.",
    "instance_dot_prod_scoring.hs_proj.",
)

_LAST_TWO_DECODER_O2O_PREFIXES = (
    "transformer.decoder.layers.4.",
    "transformer.decoder.layers.5.",
    "transformer.decoder.bbox_embed.",
    "transformer.decoder.instance_bbox_embed.",
    "dot_prod_scoring.hs_proj.",
    "instance_dot_prod_scoring.hs_proj.",
)


def _build_with_trainable_prefixes(trainable_prefixes, **kwargs):
    model = build_sam3_image_model(**kwargs)

    trainable_names = []
    frozen_names = []
    trainable_parameters = 0
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        is_trainable = name.startswith(trainable_prefixes)
        parameter.requires_grad_(is_trainable)
        if is_trainable:
            trainable_names.append(name)
            trainable_parameters += parameter.numel()
        else:
            frozen_names.append(name)
            frozen_parameters += parameter.numel()

    if not trainable_names:
        raise RuntimeError("Heads-only SAM3 configuration matched no parameters")

    logging.info(
        "Partially frozen SAM3: %d trainable tensors (%d parameters), "
        "%d frozen tensors (%d parameters)",
        len(trainable_names),
        trainable_parameters,
        len(frozen_names),
        frozen_parameters,
    )
    logging.info("Partially frozen SAM3 trainable parameters: %s", trainable_names)
    return model


def build_sam3_image_model_heads_only(**kwargs):
    """Build SAM3 while only training scoring, box, presence, and mask heads."""
    return _build_with_trainable_prefixes(_HEAD_PREFIXES, **kwargs)


def build_sam3_image_model_decoder_and_heads(**kwargs):
    """Build SAM3 while training its transformer decoder and task heads."""
    return _build_with_trainable_prefixes(
        _HEAD_PREFIXES + ("transformer.decoder.",), **kwargs
    )


def build_sam3_image_model_concept_prompt_only(**kwargs):
    """Train only the text-prompt side of the concept scoring head."""
    return _build_with_trainable_prefixes(_CONCEPT_PROMPT_PREFIXES, **kwargs)


def build_sam3_image_model_concept_query_scoring(**kwargs):
    """Train prompt- and query-side concept scoring projections only."""
    return _build_with_trainable_prefixes(
        _CONCEPT_QUERY_SCORING_PREFIXES, **kwargs
    )


def build_sam3_image_model_last_decoder_o2o(**kwargs):
    """Train the final query decoder layer plus O2O scoring and box heads."""
    return _build_with_trainable_prefixes(_LAST_DECODER_O2O_PREFIXES, **kwargs)


def build_sam3_image_model_last_two_decoder_o2o(**kwargs):
    """Train the final two query decoder layers plus O2O scoring and box heads."""
    return _build_with_trainable_prefixes(_LAST_TWO_DECODER_O2O_PREFIXES, **kwargs)
