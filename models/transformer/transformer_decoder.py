import copy
from functools import partial
from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .arm import AttentionRefinementModule
from .attention import MultiheadAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        arm: Optional[AttentionRefinementModule],
        norm=None,
        tree_bias_layers: str = "all",
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.tree_bias_layers = tree_bias_layers

        self.arm = arm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        height: int,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        rel_bias: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt

        arm = None
        for i, mod in enumerate(self.layers):
            layer_rel_bias = rel_bias
            if self.tree_bias_layers == "last1" and i != self.num_layers - 1:
                layer_rel_bias = None

            output, attn = mod(
                output,
                memory,
                arm,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                rel_bias=layer_rel_bias,
            )
            if i != len(self.layers) - 1 and self.arm is not None:
                arm = partial(self.arm, attn, memory_key_padding_mask, height)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def forward_step(self, tgt_step: Tensor, cache: "DecoderKVCache", rel_bias_step: "Optional[Tensor]") -> Tensor:
        """Incremental single-token decode through all layers.

        Parameters
        ----------
        tgt_step : Tensor  [1, B_active, D]
        cache : DecoderKVCache  (cur_len must already be set to the write position)
        rel_bias_step : Optional Tensor  [B_active*H, 1, cur_len+1]

        Returns
        -------
        output : Tensor  [1, B_active, D]  final hidden (after norm)
        """
        from .kv_cache import DecoderKVCache as _DKV  # local import to avoid circularity
        B_active = tgt_step.shape[1]
        output = tgt_step
        write_pos = cache.cur_len

        for i, mod in enumerate(self.layers):
            layer_rel_bias = rel_bias_step
            if self.tree_bias_layers == "last1" and i != self.num_layers - 1:
                layer_rel_bias = None

            # Determine ARM bias for this layer (layer 0 has no ARM)
            arm_bias = None
            if i > 0 and self.arm is not None:
                # ARM uses previous-layer final sum and current-layer pre-ARM sum
                # Both sums have index (i-1) in cache because there are num_layers-1 gaps
                sum_idx = i - 1
                # Gather mask for active beams
                mask_active = cache.memory_key_padding_mask[cache.beam_to_batch_idx]  # [B_active, S]
                arm_bias = self.arm.forward_from_sums(
                    prev_attn_sum=cache.cross_final_sum[sum_idx],
                    curr_attn_sum=cache.cross_pre_sum[sum_idx],
                    key_padding_mask=mask_active,
                    h=cache.height,
                    dtype=output.dtype,
                )

            output, cross_attn, pre_arm_attn = mod.forward_step(
                tgt_step=output,
                cache=cache,
                layer_idx=i,
                arm_bias=arm_bias,
                rel_bias_step=layer_rel_bias,
            )

            # Update ARM running sums for next layer (only for layers 0 .. num_layers-2)
            if i < self.num_layers - 1 and self.arm is not None:
                sum_idx = i
                # pre_arm_attn  → feeds cross_pre_sum   (layer i current, pre-ARM)
                # cross_attn    → feeds cross_final_sum  (layer i final, post-ARM)
                # Both are [B_active*H, 1, S]; reshape to [B_active, H, S] for accumulation
                H = cache.num_heads
                S = cache.cross_pre_sum[sum_idx].shape[-1]
                pre = pre_arm_attn.view(B_active, H, S).float()
                fin = cross_attn.view(B_active, H, S).float()
                cache.cross_pre_sum[sum_idx] = cache.cross_pre_sum[sum_idx] + pre
                cache.cross_final_sum[sum_idx] = cache.cross_final_sum[sum_idx] + fin

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        arm: Optional[AttentionRefinementModule],
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        rel_bias: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            rel_bias=rel_bias,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(
            tgt,
            memory,
            memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn

    def forward_step(
        self,
        tgt_step: Tensor,
        cache: "DecoderKVCache",
        layer_idx: int,
        arm_bias: Optional[Tensor],
        rel_bias_step: Optional[Tensor],
    ):
        """Single-token incremental decode through this layer.

        Parameters
        ----------
        tgt_step : Tensor  [1, B_active, D]  embedded + positional query token
        cache : DecoderKVCache
        layer_idx : int  which layer this is (0-indexed)
        arm_bias : Optional Tensor  [B_active*H, 1, S]  pre-computed ARM bias
        rel_bias_step : Optional Tensor  [B_active*H, 1, cur_len+1]  tree bias row

        Returns
        -------
        tgt_step : Tensor  [1, B_active, D]
        cross_attn : Tensor  [B_active*H, 1, S]  (used to update ARM sums)
        pre_arm_cross_attn : Tensor  [B_active*H, 1, S]  (before ARM re-softmax)
        """
        from .kv_cache import DecoderKVCache as _DKV  # local import to avoid circularity
        write_pos = cache.cur_len

        # --- Causal self-attention (incremental) ---
        tgt2, _ = self.self_attn.forward_cached_self(
            query_step=tgt_step,
            cache_k=cache.self_k[layer_idx],
            cache_v=cache.self_v[layer_idx],
            write_pos=write_pos,
            rel_bias_step=rel_bias_step,
        )
        tgt_step = tgt_step + self.dropout1(tgt2)
        tgt_step = self.norm1(tgt_step)

        # --- Cross-attention (incremental, static K/V) ---
        # Pre-ARM cross attention
        tgt2_pre, pre_arm_attn = self.multihead_attn.forward_cached_cross(
            query_step=tgt_step,
            static_k=cache.cross_k[layer_idx],
            static_v=cache.cross_v[layer_idx],
            key_padding_mask=cache.memory_key_padding_mask,
            beam_to_batch_idx=cache.beam_to_batch_idx,
            arm_bias=None,  # compute without ARM first for sum update
        )

        # If ARM bias is available, apply it and redo softmax inside forward_cached_cross
        if arm_bias is not None:
            tgt2, cross_attn = self.multihead_attn.forward_cached_cross(
                query_step=tgt_step,
                static_k=cache.cross_k[layer_idx],
                static_v=cache.cross_v[layer_idx],
                key_padding_mask=cache.memory_key_padding_mask,
                beam_to_batch_idx=cache.beam_to_batch_idx,
                arm_bias=arm_bias,
            )
        else:
            tgt2, cross_attn = tgt2_pre, pre_arm_attn

        tgt_step = tgt_step + self.dropout2(tgt2)
        tgt_step = self.norm2(tgt_step)

        # --- Feed-forward ---
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_step))))
        tgt_step = tgt_step + self.dropout3(tgt2)
        tgt_step = self.norm3(tgt_step)

        return tgt_step, cross_attn, pre_arm_attn

