"""Error-driven box prompt sampling for interactive PCS training."""

import torch

from sam3.model.box_ops import box_cxcywh_to_xyxy, box_iou


class ErrorDrivenBoxPromptSampler:
    """Append one detached positive or negative exemplar per batch item."""

    def __init__(
        self,
        score_threshold=0.25,
        match_iou_threshold=0.5,
        negative_max_iou=0.1,
        repeated_prompt_iou=0.9,
        prompt_probability=1.0,
    ):
        self.score_threshold = score_threshold
        self.match_iou_threshold = match_iou_threshold
        self.negative_max_iou = negative_max_iou
        self.repeated_prompt_iou = repeated_prompt_iou
        if not 0.0 <= prompt_probability <= 1.0:
            raise ValueError("prompt_probability must be in [0, 1]")
        self.prompt_probability = prompt_probability

    @torch.no_grad()
    def sample(self, geo_prompt, find_target, previous_out):
        pred_boxes = previous_out["pred_boxes"].detach()
        scores = previous_out["pred_logits"].detach().sigmoid().squeeze(-1)
        presence = previous_out.get("presence_logit_dec")
        if presence is not None:
            presence = presence.detach().sigmoid()
            while presence.ndim < scores.ndim:
                presence = presence.unsqueeze(-1)
            scores = scores * presence

        gt_boxes = find_target.boxes_padded.detach()
        num_boxes = find_target.num_boxes.detach()
        batch_size = pred_boxes.shape[0]
        sampled_boxes = pred_boxes.new_zeros((1, batch_size, 4))
        sampled_labels = torch.zeros(
            (1, batch_size), dtype=torch.long, device=pred_boxes.device
        )
        sampled_mask = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=pred_boxes.device
        )
        metadata = []

        for batch_index in range(batch_size):
            if torch.rand((), device=pred_boxes.device) >= self.prompt_probability:
                metadata.append({"source": "text_only_replay"})
                continue

            target_count = int(num_boxes[batch_index].item())
            targets = gt_boxes[batch_index, :target_count]
            predictions = pred_boxes[batch_index]
            prediction_scores = scores[batch_index]
            active = prediction_scores >= self.score_threshold

            if target_count:
                pair_iou, _ = box_iou(
                    box_cxcywh_to_xyxy(predictions),
                    box_cxcywh_to_xyxy(targets),
                )
                covered = (pair_iou[active] >= self.match_iou_threshold).any(dim=0)
                missing = (~covered).nonzero().flatten()
            else:
                pair_iou = predictions.new_zeros((len(predictions), 0))
                missing = torch.empty(0, dtype=torch.long, device=predictions.device)

            existing_boxes = geo_prompt.box_embeddings
            existing_labels = geo_prompt.box_labels
            existing_mask = geo_prompt.box_mask

            selected_box = None
            selected_label = None
            source = None
            if len(missing):
                areas = targets[missing, 2] * targets[missing, 3]
                for target_index in missing[areas.argsort(descending=True)]:
                    candidate = targets[target_index]
                    repeated = self._already_prompted(
                        candidate,
                        True,
                        batch_index,
                        existing_boxes,
                        existing_labels,
                        existing_mask,
                    )
                    if not repeated:
                        selected_box = candidate
                        selected_label = True
                        source = "false_negative"
                        break

            if selected_box is None:
                if target_count:
                    max_target_iou = pair_iou.max(dim=1).values
                else:
                    max_target_iou = prediction_scores.new_zeros(len(predictions))
                false_positive = active & (max_target_iou < self.negative_max_iou)
                for prediction_index in prediction_scores.argsort(descending=True):
                    if not false_positive[prediction_index]:
                        continue
                    candidate = predictions[prediction_index]
                    repeated = self._already_prompted(
                        candidate,
                        False,
                        batch_index,
                        existing_boxes,
                        existing_labels,
                        existing_mask,
                    )
                    if not repeated:
                        selected_box = candidate
                        selected_label = False
                        source = "false_positive"
                        break

            if selected_box is not None:
                sampled_boxes[0, batch_index] = selected_box
                sampled_labels[0, batch_index] = int(selected_label)
                sampled_mask[batch_index, 0] = False
            metadata.append({"source": source})

        geo_prompt.append_boxes(sampled_boxes, sampled_labels, sampled_mask)
        return geo_prompt, metadata

    def _already_prompted(
        self,
        candidate,
        label,
        batch_index,
        existing_boxes,
        existing_labels,
        existing_mask,
    ):
        if existing_boxes is None or existing_boxes.shape[0] == 0:
            return False
        valid = ~existing_mask[batch_index]
        valid &= existing_labels[:, batch_index].bool() == label
        if not valid.any():
            return False
        prior = existing_boxes[:, batch_index][valid]
        iou, _ = box_iou(
            box_cxcywh_to_xyxy(candidate[None]),
            box_cxcywh_to_xyxy(prior),
        )
        return bool((iou >= self.repeated_prompt_iou).any())
