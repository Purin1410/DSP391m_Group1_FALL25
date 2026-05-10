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
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

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
    ) -> Tensor:
        output = tgt

        arm = None
        for i, mod in enumerate(self.layers):
            output, attn = mod(
                output,
                memory,
                arm,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if i != len(self.layers) - 1 and self.arm is not None:
                arm = partial(self.arm, attn, memory_key_padding_mask, height)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def forward_step(
        self,
        tgt: Tensor,
        cache: dict,
        beam_origin: Tensor,
        memory_key_padding_mask: Tensor,
        height: int,
    ) -> Tuple[Tensor, dict]:
        """
        tgt: [1, B_beam, D]
        cache: Decoder full cache dict
        beam_origin: [B_beam]
        memory_key_padding_mask: [B, src_len]
        """
        output = tgt

        for i, mod in enumerate(self.layers):
            arm_fn = None
            if self.arm is not None and i > 0:
                def arm_fn_closure(first_pass_attn: Tensor, rows_s: Tensor, src_idx: int, i_val=i):
                    # first_pass_attn: [count_s, nhead, 1, src_len]
                    p = cache["arm"]["final_attn_cumsum"][i_val - 1][rows_s] # [count_s, nhead, src_len]
                    s = cache["arm"]["first_pass_cumsum"][i_val][rows_s] # [count_s, nhead, src_len]
                    
                    p_flat = p.view(-1, p.size(-1))
                    first_pass_flat = first_pass_attn.view(-1, 1, first_pass_attn.size(-1))
                    
                    # self_attn_cumsum includes the current element
                    s_flat = s.view(-1, s.size(-1)) + first_pass_flat.squeeze(1)
                    
                    mask = memory_key_padding_mask[src_idx:src_idx+1].expand(rows_s.numel(), -1)
                    
                    cov = self.arm.forward_step(p_flat, s_flat, first_pass_flat, mask, height)
                    
                    # Update first pass cumsum inplace
                    cache["arm"]["first_pass_cumsum"][i_val][rows_s] += first_pass_attn.squeeze(2)
                    
                    return cov.view(rows_s.numel(), self.arm.nhead, 1, -1)
                    
                arm_fn = arm_fn_closure

            output, layer_cache, attn = mod.forward_step(
                tgt=output,
                cache=cache["layers"][i],
                cross_k=cache["cross_kv"][i]["k"],
                cross_v=cache["cross_kv"][i]["v"],
                beam_origin=beam_origin,
                memory_key_padding_mask=memory_key_padding_mask,
                arm_fn=arm_fn,
            )
            cache["layers"][i] = layer_cache

            if self.arm is not None:
                attn_reshaped = attn.view(-1, self.arm.nhead, attn.size(-1)) # [B_beam, nhead, src_len]
                cache["arm"]["final_attn_cumsum"][i] += attn_reshaped

        if self.norm is not None:
            output = self.norm(output)

        return output, cache



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
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
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
        tgt: Tensor,
        cache: dict,
        cross_k: Tensor,
        cross_v: Tensor,
        beam_origin: Tensor,
        memory_key_padding_mask: Tensor,
        arm_fn=None,
    ) -> Tuple[Tensor, dict, Tensor]:
        """
        tgt: [1, B_beam, D]
        cache: layer cache dict {"self_k": ..., "self_v": ...}
        cross_k, cross_v: [B * nhead, src_len, head_dim]
        beam_origin: [B_beam]
        memory_key_padding_mask: [B, src_len]
        """
        tgt2, new_self_k, new_self_v, _ = self.self_attn.forward_step(
            query=tgt,
            cached_k=cache.get("self_k"),
            cached_v=cache.get("self_v"),
        )
        cache["self_k"] = new_self_k
        cache["self_v"] = new_self_v

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, _, _, attn = self.multihead_attn.forward_step(
            query=tgt,
            cached_k=cross_k,
            cached_v=cross_v,
            beam_origin=beam_origin,
            arm_fn=arm_fn,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, cache, attn
