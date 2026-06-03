"""Inference-only KV-cache for the CoMER decoder.

Active only under:
    eval mode  AND  torch.no_grad()

Never referenced during training or validation loss computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor


@dataclass
class DecoderKVCache:
    """Per-layer KV tensors for incremental beam-search decoding.

    Shapes
    ------
    self_k / self_v : List[Tensor]  length = num_layers
        each [B_active, H, max_len, Hd]   -- grows in-place at write_pos
    cross_k / cross_v : List[Tensor]  length = num_layers
        each [B_original, H, S, Hd]      -- static, never duplicated per beam
    cross_pre_sum / cross_final_sum : List[Tensor]  length = num_layers-1
        each fp32 [B_active, H, S]        -- running ARM coverage sums
    beam_to_batch_idx : Tensor  [B_active]
        maps each active beam hypothesis → original batch index for cross K/V lookup
    memory_key_padding_mask : Tensor  [B_original, S]
        source padding mask for cross-attention masking
    height : int   spatial height of the encoder feature map (needed by ARM)
    cur_len : int  number of tokens decoded so far (0 before first step)
    max_len : int
    batch_size : int   original batch size (before beam expand)
    beam_size : int
    num_layers : int
    num_heads : int
    head_dim : int
    """
    self_k: List[Tensor]
    self_v: List[Tensor]
    cross_k: List[Tensor]
    cross_v: List[Tensor]
    # ARM coverage running sums (fp32); one entry per *inter-layer* gap = num_layers - 1
    cross_pre_sum: List[Tensor]
    cross_final_sum: List[Tensor]
    beam_to_batch_idx: Tensor
    memory_key_padding_mask: Tensor
    height: int
    cur_len: int
    max_len: int
    batch_size: int
    beam_size: int
    num_layers: int
    num_heads: int
    head_dim: int

    # ------------------------------------------------------------------ #
    #  Beam expand: called once after the SOS step, before reorder loop   #
    # ------------------------------------------------------------------ #
    def expand_beam_(self, beam_size: int) -> None:
        """Expand beam-indexed tensors from [B, ...] to [B*beam_size, ...].

        Only self_k, self_v, cross_pre_sum, cross_final_sum, and
        beam_to_batch_idx are duplicated -- cross K/V stay original-batch
        indexed and are accessed via beam_to_batch_idx.
        """
        if beam_size == 1:
            return

        def _expand(t: Tensor) -> Tensor:
            # t: [B, ...] → [B*beam_size, ...]  (interleaved repeat)
            return t.repeat_interleave(beam_size, dim=0)

        self.self_k = [_expand(t) for t in self.self_k]
        self.self_v = [_expand(t) for t in self.self_v]
        self.cross_pre_sum = [_expand(t) for t in self.cross_pre_sum]
        self.cross_final_sum = [_expand(t) for t in self.cross_final_sum]
        self.beam_to_batch_idx = self.beam_to_batch_idx.repeat_interleave(beam_size)
        self.beam_size = beam_size

    # ------------------------------------------------------------------ #
    #  Beam reorder: called after beam_scorer.process() each step         #
    # ------------------------------------------------------------------ #
    def reorder_(self, beam_idx: Tensor) -> None:
        """Reorder active hypotheses according to beam_idx.

        beam_idx: [B_active] LongTensor — parent index for each surviving beam.
        Cross K/V are NOT touched (they remain original-batch indexed).
        beam_to_batch_idx is reordered from the *old* ordering.
        """
        idx = beam_idx

        self.self_k = [t[idx].clone() for t in self.self_k]
        self.self_v = [t[idx].clone() for t in self.self_v]
        self.cross_pre_sum = [t[idx].clone() for t in self.cross_pre_sum]
        self.cross_final_sum = [t[idx].clone() for t in self.cross_final_sum]
        self.beam_to_batch_idx = self.beam_to_batch_idx[idx].clone()
