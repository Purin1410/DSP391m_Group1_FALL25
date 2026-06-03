from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from utils.vocab_info import VocabInfo

from .pos_enc import WordPosEnc
from .transformer.arm import AttentionRefinementModule
from .transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from .transformer.tree_bias import TreeRelationBuilder, TreeRelativeBias, CausalR2LTreeRelationBuilder
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
    tree_bias_layers: str = "all",
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm, tree_bias_layers=tree_bias_layers)
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
        use_tree_bias: bool = True,
        tree_bias_num_buckets: int = 16,
        tree_bias_mode: str = "full",
        tree_bias_layers: str = "all",
        tree_bias_rel_set: str = "full",
        use_bidirectional: bool = False,
    ):
        super().__init__()
        self.vocab_info = vocab_info
        self.use_bidirectional = bool(use_bidirectional)

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
            tree_bias_layers=tree_bias_layers,
        )

        self.proj = nn.Linear(d_model, vocab_info.vocab_size)

        # -----------------------------
        # Tree-structure relative bias
        # -----------------------------
        self.use_tree_bias = bool(use_tree_bias)
        self.tree_bias_layers = tree_bias_layers

        if self.use_tree_bias:
            if vocab_info is None or vocab_info.words is None or not hasattr(vocab_info.words, "idx2word"):
                raise ValueError("Tree bias requires vocab_info.words.idx2word")

            type_size = 6 if self.use_bidirectional else 5
            self._tree_builder = TreeRelationBuilder(
                id2tok=vocab_info.words.idx2word,
                pad_id=vocab_info.pad_id,
                num_buckets=tree_bias_num_buckets,
                mode=tree_bias_mode,
                rel_set=tree_bias_rel_set,
                type_size=type_size,
            )
            if self.use_bidirectional:
                self._tree_builder_r2l = CausalR2LTreeRelationBuilder(
                    id2tok=vocab_info.words.idx2word,
                    pad_id=vocab_info.pad_id,
                    num_buckets=tree_bias_num_buckets,
                    mode=tree_bias_mode,
                    rel_set=tree_bias_rel_set,
                    type_size=type_size,
                )
            else:
                self._tree_builder_r2l = None

            self._tree_rel_bias = TreeRelativeBias(
                num_heads=nhead,
                num_relations=self._tree_builder.num_relations,
            )
        else:
            self._tree_builder = None
            self._tree_builder_r2l = None
            self._tree_rel_bias = None
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

    def _build_rel_ids_for_tgt(self, tgt: torch.LongTensor) -> torch.LongTensor:
        if self.use_bidirectional:
            half_B = tgt.shape[0] // 2
            rel_ids_l2r = self._tree_builder.build(tgt[:half_B])
            rel_ids_r2l = self._tree_builder_r2l.build(tgt[half_B:])
            return torch.cat([rel_ids_l2r, rel_ids_r2l], dim=0)
        else:
            return self._tree_builder.build(tgt)

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor, rel_ids: Optional[LongTensor] = None
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

        rel_bias = None
        if self.use_tree_bias and self._tree_rel_bias is not None:
            if rel_ids is None:
                rel_ids = self._build_rel_ids_for_tgt(tgt)
            rel_bias = self._tree_rel_bias(rel_ids, flatten=True)

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
            rel_bias=rel_bias,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out


    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor, rel_ids: Optional[LongTensor] = None
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        return self(src[0], src_mask[0], input_ids, rel_ids=rel_ids)

    # ------------------------------------------------------------------
    # Incremental KV-cache inference APIs (eval / no-grad only)
    # ------------------------------------------------------------------

    def init_decode_cache(
        self,
        src: FloatTensor,    # [B, h, w, D]
        src_mask: LongTensor,  # [B, h, w]
        max_len: int,
    ) -> "DecoderKVCache":
        """Pre-project encoder memory and allocate all KV/ARM buffers.

        Must be called in eval mode under torch.no_grad().

        Parameters
        ----------
        src : [B, h, w, D]   encoder feature map
        src_mask : [B, h, w] True = padded
        max_len : int

        Returns
        -------
        DecoderKVCache  ready for the first transform_step call
        """
        from .transformer.kv_cache import DecoderKVCache
        from einops import rearrange as _re

        B, h, w, D = src.shape
        S = h * w
        num_layers = self.model.num_layers
        num_heads = self.model.layers[0].self_attn.num_heads
        head_dim = self.model.layers[0].self_attn.head_dim

        # Flatten memory: [S, B, D]
        memory = _re(src, "b h w d -> (h w) b d")
        mem_mask = _re(src_mask, "b h w -> b (h w)")   # [B, S]

        # Pre-project cross K/V for each layer
        cross_k = []
        cross_v = []
        for layer in self.model.layers:
            ck, cv = layer.multihead_attn.project_static_kv(memory)
            cross_k.append(ck)  # [B, H, S, Hd]
            cross_v.append(cv)

        # Allocate self K/V buffers (filled incrementally)
        device = src.device
        dtype = src.dtype
        self_k = [torch.zeros(B, num_heads, max_len, head_dim, device=device, dtype=dtype)
                  for _ in range(num_layers)]
        self_v = [torch.zeros(B, num_heads, max_len, head_dim, device=device, dtype=dtype)
                  for _ in range(num_layers)]

        # Allocate ARM fp32 running sums (num_layers - 1 inter-layer gaps)
        num_gaps = max(num_layers - 1, 0)
        cross_pre_sum = [torch.zeros(B, num_heads, S, dtype=torch.float32, device=device)
                         for _ in range(num_gaps)]
        cross_final_sum = [torch.zeros(B, num_heads, S, dtype=torch.float32, device=device)
                           for _ in range(num_gaps)]

        cache = DecoderKVCache(
            self_k=self_k,
            self_v=self_v,
            cross_k=cross_k,
            cross_v=cross_v,
            cross_pre_sum=cross_pre_sum,
            cross_final_sum=cross_final_sum,
            beam_to_batch_idx=torch.arange(B, device=device, dtype=torch.long),
            memory_key_padding_mask=mem_mask,
            height=h,
            cur_len=0,
            max_len=max_len,
            batch_size=B,
            beam_size=1,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        return cache

    def transform_step(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        token_ids: LongTensor,          # [B_active, cur_len+1]  full prefix including new token
        cache: "DecoderKVCache",
        rel_ids_step: Optional[LongTensor] = None,  # [B_active, 1, cur_len+1]
    ) -> FloatTensor:
        """Incremental one-step decode using KV cache.

        Must be called in eval mode under torch.no_grad().
        Increments cache.cur_len after the forward pass.

        Parameters
        ----------
        token_ids : [B_active, cur_len+1]  — only the last column is the new token
        cache : DecoderKVCache
        rel_ids_step : optional [B_active, 1, cur_len+1] tree relation ids for
            the new query token attending to the current prefix

        Returns
        -------
        logits : FloatTensor  [B_active, vocab_size]  (last token logits)
        """
        from .transformer.kv_cache import DecoderKVCache

        B_active = token_ids.shape[0]
        write_pos = cache.cur_len  # 0-indexed position being written

        # Embed only the last (new) token
        new_tok = token_ids[:, -1:]                           # [B_active, 1]
        tgt_embed = self.word_embed(new_tok)                  # [B_active, 1, D]

        # Add positional encoding at row `write_pos` only
        pos_enc = self.pos_enc.pe[write_pos:write_pos + 1, :]  # [1, D]
        tgt_embed = tgt_embed + pos_enc.unsqueeze(0)           # [B_active, 1, D]
        tgt_embed = self.norm(tgt_embed)

        # Seq-first for transformer layers: [1, B_active, D]
        tgt_step = tgt_embed.transpose(0, 1).contiguous()

        # Build tree-bias row if enabled
        rel_bias_step = None
        if self.use_tree_bias and self._tree_rel_bias is not None and rel_ids_step is not None:
            # rel_ids_step: [B_active, 1, cur_len+1]
            rel_bias_step = self._tree_rel_bias(rel_ids_step, flatten=True)  # [B_active*H, 1, cur_len+1]

        # Run through decoder stack
        out = self.model.forward_step(
            tgt_step=tgt_step,
            cache=cache,
            rel_bias_step=rel_bias_step,
        )  # [1, B_active, D]

        out = out.transpose(0, 1)  # [B_active, 1, D]
        logits = self.proj(out[:, 0, :])  # [B_active, vocab_size]

        # Increment position counter
        cache.cur_len += 1

        return logits
