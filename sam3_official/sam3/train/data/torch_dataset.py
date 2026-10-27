# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import json
import random
from collections import defaultdict
from typing import Callable, Iterable, Optional

from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset


class CocoConceptBalancedDataset(Dataset):
    """Draw an equal, deterministic number of samples per concept each epoch."""

    def __init__(self, dataset: Dataset, ann_file: str, samples_per_concept: int, seed=42):
        self.dataset = dataset
        self.samples_per_concept = samples_per_concept
        self.seed = seed
        with open(ann_file, encoding="utf-8") as annotation_file:
            coco = json.load(annotation_file)
        concept_by_id = {int(image["id"]): image["concept"] for image in coco["images"]}
        self.indices_by_concept = defaultdict(list)
        for dataset_index, datapoint_id in enumerate(dataset.ids):
            metadata = dataset.coco.loadImagesFromDatapoint(datapoint_id)
            coco_image_id = int(metadata[0]["coco_img_id"])
            self.indices_by_concept[concept_by_id[coco_image_id]].append(dataset_index)
        if len(self.indices_by_concept) < 2:
            raise ValueError("Balanced sampling requires at least two concepts")
        if any(not indices for indices in self.indices_by_concept.values()):
            raise ValueError("Every concept must contain at least one sample")
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        selected = {}
        for group_index, (concept, source_indices) in enumerate(
            sorted(self.indices_by_concept.items())
        ):
            rng = random.Random(self.seed + epoch * 1009 + group_index)
            order = list(source_indices)
            rng.shuffle(order)
            selected[concept] = [
                order[index % len(order)] for index in range(self.samples_per_concept)
            ]
        self.epoch_indices = [
            selected[concept][index]
            for index in range(self.samples_per_concept)
            for concept in sorted(selected)
        ]

    def __len__(self):
        return len(self.epoch_indices)

    def __getitem__(self, index):
        return self.dataset[self.epoch_indices[index]]


class TorchDataset:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        enable_distributed_sampler=True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        assert not isinstance(self.dataset, IterableDataset), "Not supported yet"
        if enable_distributed_sampler:
            self.sampler = DistributedSampler(self.dataset, shuffle=self.shuffle)
        else:
            # pyrefly: ignore [bad-assignment]
            self.sampler = None

    def get_loader(self, epoch) -> Iterable:
        if self.sampler:
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            # pyrefly: ignore [missing-attribute]
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )
