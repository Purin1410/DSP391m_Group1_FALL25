from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
import numpy as np
from PIL import Image
from torch import FloatTensor, LongTensor
from pathlib import Path

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
# - lazy_load=False: List[(str, Image.Image, List[str])]
# - lazy_load=True:  List[(str, Tuple[int,int], List[str])]
Data = List[Tuple[str, object, List[str]]]


from torch.utils.data import Sampler
import random

class BucketedBatchSampler(Sampler):
    def __init__(self, data: Data, max_pixels_per_batch: int, max_batch_size: int, shuffle: bool = True, drop_last: bool = False, maxlen: int = 200, max_image_size: int = 32e4):
        self.data = data
        self.max_pixels_per_batch = max_pixels_per_batch
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.maxlen = maxlen
        self.max_image_size = max_image_size
        
        self.batches = self._build_batches()

    def _build_batches(self):
        # Create list of indices
        indices = list(range(len(self.data)))
        
        # Function to get area
        def get_area(idx):
            fea = self.data[idx][1]
            if hasattr(fea, "size"):
                return fea.size[0] * fea.size[1]
            return fea[0] * fea[1]
            
        # Filter indices by maxlen and max_image_size
        valid_indices = []
        for idx in indices:
            _, fea, lab = self.data[idx]
            size = get_area(idx)
            if len(lab) > self.maxlen:
                continue
            if size > self.max_image_size:
                continue
            valid_indices.append(idx)
            
        valid_indices.sort(key=get_area)
        
        batches = []
        current_batch = []
        biggest_image_size = 0
        
        for idx in valid_indices:
            size = get_area(idx)
            if size > biggest_image_size:
                biggest_image_size = size
            
            batch_image_size = biggest_image_size * (len(current_batch) + 1)
            
            if batch_image_size > self.max_pixels_per_batch or len(current_batch) == self.max_batch_size:
                if len(current_batch) > 0:
                    batches.append(current_batch)
                current_batch = []
                biggest_image_size = size
                
            current_batch.append(idx)
            
        if len(current_batch) > 0 and not self.drop_last:
            batches.append(current_batch)
            
        print(f"total {len(batches)} batch data loaded")
        return batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def build_validation_dataset(
    archive: str,
    folder: str,
    lazy_load: bool = False,
):
    if folder != "all":
        data = extract_data(root_dir=archive, split=folder, lazy_load=lazy_load)
    else:
        data = []
        for folder_name in ["2014", "2016", "2019"]:
            data += extract_data(root_dir=archive, split=folder_name, lazy_load=lazy_load)

    return data


def build_train_dataset(
    archive: str,
    folder: str,
    lazy_load: bool = False,
):
    if folder == "train":
        data = extract_data(root_dir=archive, split=folder, lazy_load=lazy_load)
    return data


# ---------------------------------------------------------------------
# Batch class
# ---------------------------------------------------------------------
@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]
    tgt: Optional[LongTensor] = None
    out: Optional[LongTensor] = None
    labels: Optional[LongTensor] = None
    lengths: Optional[LongTensor] = None

    def __len__(self) -> int:
        return len(self.img_bases)

    def pin_memory(self):
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.pin_memory(),
            mask=self.mask.pin_memory(),
            indices=self.indices,
            tgt=None if self.tgt is None else self.tgt.pin_memory(),
            out=None if self.out is None else self.out.pin_memory(),
            labels=None if self.labels is None else self.labels.pin_memory(),
            lengths=None if self.lengths is None else self.lengths.pin_memory(),
        )

    def to(self, device, non_blocking=True) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device, non_blocking=non_blocking),
            mask=self.mask.to(device, non_blocking=non_blocking),
            indices=self.indices,
            tgt=None if self.tgt is None else self.tgt.to(device, non_blocking=non_blocking),
            out=None if self.out is None else self.out.to(device, non_blocking=non_blocking),
            labels=None if self.labels is None else self.labels.to(device, non_blocking=non_blocking),
            lengths=None if self.lengths is None else self.lengths.to(device, non_blocking=non_blocking),
        )
