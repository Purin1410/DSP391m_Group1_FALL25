from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from datamodule.vocab import VocabInfo

from .pos_enc import WordPosEnc
from .transformer.arm import AttentionRefinementModule
from .transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from utils.generation_utils import DecodeModel


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
    arm_norm_impl: str = "legacy",
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage, norm_impl=arm_norm_impl)
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder


class Decoder(DecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        vocab_info: VocabInfo,
        arm_norm_impl: str = "legacy",
    ):
        super().__init__()
        self.vocab_info = vocab_info

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_info.vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            arm_norm_impl=arm_norm_impl,
        )

        self.proj = nn.Linear(d_model, vocab_info.vocab_size)
        # Causal mask cache: keyed by (device_type, device_index, dtype_str)
        # so CPU->CUDA or dtype changes don't reuse a stale/wrong-device mask.
        self._causal_mask_cache = {}

    def _build_attention_mask(self, length, device=None, dtype=torch.bool):
        if device is None:
            device = self.device
        cache_key = (device.type, device.index, str(dtype))
        cached = self._causal_mask_cache.get(cache_key)
        if cached is not None and cached.size(0) >= length:
            return cached[:length, :length]

        # lazily create causal attention mask (upper triangular = True)
        mask = torch.full(
            (length, length), fill_value=1, dtype=dtype, device=device
        )
        mask.triu_(1)  # zero out the lower diagonal
        self._causal_mask_cache[cache_key] = mask
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        B_tgt, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == self.vocab_info.pad_id

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out

    def init_decode_cache(
        self, batch_beam_size: int, src: FloatTensor, src_mask: LongTensor
    ) -> dict:
        """
        src: [B, H, W, D]
        src_mask: [B, H, W]
        """
        B, h, w, d = src.shape
        assert batch_beam_size % B == 0
        beam_size = batch_beam_size // B
        
        src_flat = rearrange(src, "b h w d -> (h w) b d")
        src_mask_flat = rearrange(src_mask, "b h w -> b (h w)")
        
        cross_kv = []
        for mod in self.model.layers:
            cross_k, cross_v = mod.multihead_attn.project_kv(src_flat, src_flat)
            # shape: [B * nhead, src_len, head_dim]
            cross_kv.append({"k": cross_k, "v": cross_v})
            
        cache = {
            "layers": [
                {"self_k": None, "self_v": None}
                for _ in range(self.model.num_layers)
            ],
            "cross_kv": cross_kv,
            "memory_key_padding_mask": src_mask_flat,
            "height": h,
            "beam_origin": torch.arange(B, device=src.device).repeat_interleave(beam_size),
        }
        
        if self.model.arm is not None:
            cache["arm"] = {
                "final_attn_cumsum": [
                    torch.zeros((batch_beam_size, self.model.arm.nhead, h * w), device=src.device, dtype=src.dtype)
                    for _ in range(self.model.num_layers)
                ],
                "first_pass_cumsum": [
                    torch.zeros((batch_beam_size, self.model.arm.nhead, h * w), device=src.device, dtype=src.dtype)
                    for _ in range(self.model.num_layers)
                ],
            }
            
        return cache

    def reorder_decode_cache(self, cache: dict, new_order: LongTensor) -> dict:
        B_beam = new_order.size(0)
        nhead = cache["cross_kv"][0]["k"].size(0) // cache["memory_key_padding_mask"].size(0)
        
        new_order_expanded = new_order.unsqueeze(1) * nhead + torch.arange(nhead, device=new_order.device).unsqueeze(0)
        new_order_expanded = new_order_expanded.view(-1)
        
        for i in range(self.model.num_layers):
            if cache["layers"][i]["self_k"] is not None:
                cache["layers"][i]["self_k"] = cache["layers"][i]["self_k"][new_order_expanded]
                cache["layers"][i]["self_v"] = cache["layers"][i]["self_v"][new_order_expanded]
                
            if "arm" in cache:
                cache["arm"]["final_attn_cumsum"][i] = cache["arm"]["final_attn_cumsum"][i][new_order]
                cache["arm"]["first_pass_cumsum"][i] = cache["arm"]["first_pass_cumsum"][i][new_order]
                
        cache["beam_origin"] = cache["beam_origin"][new_order]
        return cache

    def decode_step(self, last_tokens: LongTensor, cache: dict, step: int, input_ids: Optional[LongTensor] = None) -> Tuple[FloatTensor, dict]:
        """
        last_tokens: [B_beam, 1]
        """
        tgt = self.word_embed(last_tokens) # [B_beam, 1, D]
        
        emb = self.pos_enc.pe[step:step+1, :]
        tgt = tgt + emb[None, :, :]
        tgt = self.norm(tgt)
        
        tgt = rearrange(tgt, "b l d -> l b d") # [1, B_beam, D]
        
        out, cache = self.model.forward_step(
            tgt=tgt,
            cache=cache,
            beam_origin=cache["beam_origin"],
            memory_key_padding_mask=cache["memory_key_padding_mask"],
            height=cache["height"]
        )
        
        out = rearrange(out, "l b d -> b l d") # [B_beam, 1, D]
        out = self.proj(out) # [B_beam, 1, vocab_size]
        
        return out.squeeze(1), cache

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        s, sm = src[0], src_mask[0]
        B_tgt = input_ids.shape[0]
        B_src = s.shape[0]
        if B_tgt > B_src:
            # This path is used by _rate() which passes original-size src with
            # beam-expanded tgt.  The main decode loop (B2) pre-expands src
            # before the loop, so this branch is NOT hit per-token in _beam_search.
            assert B_tgt % B_src == 0
            m = B_tgt // B_src
            s = s.unsqueeze(1).expand(-1, m, -1, -1, -1).reshape(B_tgt, s.shape[1], s.shape[2], s.shape[3])
            sm = sm.unsqueeze(1).expand(-1, m, -1, -1).reshape(B_tgt, sm.shape[1], sm.shape[2])
        word_out = self(s, sm, input_ids)
        return word_out

