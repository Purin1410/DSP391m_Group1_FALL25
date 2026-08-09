import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from utils.bidirectional import BidirectionalLayout
from utils.vocab_info import VocabInfo

from .pos_enc import WordPosEnc
from .transformer.arm import AttentionRefinementModule
from .transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from .transformer.tree_bias import TreeRelationBuilder, TreeRelativeBias
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

    decoder = TransformerDecoder(
        decoder_layer, num_decoder_layers, arm, tree_bias_layers=tree_bias_layers
    )
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
            tree_bias_layers=tree_bias_layers,
        )

        self.proj = nn.Linear(d_model, vocab_info.vocab_size)

        # -----------------------------------------------------------------
        # LiSRB: tree-structure relative bias, L2R-only.
        #
        # CoMER always decodes bidirectionally (see to_bi_tgt_out in
        # utils/utils.py): rows [0, B) of `tgt` are L2R, rows [B, 2B) are
        # R2L. The bias below is only ever computed from and applied to the
        # L2R half; the R2L half gets a plain torch.zeros(...) that never
        # touches self._tree_rel_bias.emb, so it carries no gradient and no
        # value from the bias table -- see _build_rel_bias_for_tgt below and
        # utils/bidirectional.py for the single source of truth on the
        # L2R/R2L split. Neither attention.py nor transformer_decoder.py
        # know this convention exists; they only see one additive tensor.
        # -----------------------------------------------------------------
        self.use_tree_bias = bool(use_tree_bias)
        self.tree_bias_layers = tree_bias_layers

        if self.use_tree_bias:
            if vocab_info is None or vocab_info.words is None or not hasattr(
                vocab_info.words, "idx2word"
            ):
                raise ValueError("Tree bias requires vocab_info.words.idx2word")

            self._tree_builder = TreeRelationBuilder(
                id2tok=vocab_info.words.idx2word,
                pad_id=vocab_info.pad_id,
                num_buckets=tree_bias_num_buckets,
                mode=tree_bias_mode,
                rel_set=tree_bias_rel_set,
            )
            if not (
                self._tree_builder.sup_ids
                or self._tree_builder.sub_ids
                or self._tree_builder.frac_ids
            ):
                warnings.warn(
                    "Tree bias is enabled but none of ^, _, \\frac/\\dfrac/\\tfrac "
                    "were found in the vocabulary -- every relation will resolve "
                    "to TYPE_ROOT and the bias becomes a no-op. Check that "
                    "vocab_info.words.idx2word matches the training dictionary."
                )

            self._tree_rel_bias = TreeRelativeBias(
                num_heads=nhead,
                num_relations=self._tree_builder.num_relations,
            )
        else:
            self._tree_builder = None
            self._tree_rel_bias = None

        self._causal_mask_cache = {}

    def _build_rel_bias_for_tgt(self, tgt: torch.LongTensor) -> Optional[torch.Tensor]:
        """Build the additive self-attention bias for a bidirectional `tgt`
        batch, biasing only the L2R half. Returns None when tree bias is
        disabled."""
        if not self.use_tree_bias or self._tree_rel_bias is None:
            return None

        l2r_slice, r2l_slice = BidirectionalLayout.split(tgt.shape[0])

        rel_ids_l2r = self._tree_builder.build(tgt[l2r_slice])  # [B, L, L]
        rel_bias_l2r = self._tree_rel_bias(rel_ids_l2r, flatten=True)  # [B*H, L, L]

        n_r2l = r2l_slice.stop - r2l_slice.start
        # Plain zeros, never routed through self._tree_rel_bias.emb: the R2L
        # half gets no bias and no gradient from the bias table, by
        # construction rather than by coincidence.
        rel_bias_r2l = torch.zeros(
            n_r2l * self._tree_rel_bias.num_heads,
            rel_bias_l2r.size(1),
            rel_bias_l2r.size(2),
            dtype=rel_bias_l2r.dtype,
            device=rel_bias_l2r.device,
        )

        return torch.cat([rel_bias_l2r, rel_bias_r2l], dim=0)  # [2B*H, L, L]

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
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor,
        return_aux: bool = False, capture_embed: bool = False,
        capture_cross_attn: bool = False, capture_self_attn: bool = False
    ):
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

        rel_bias = self._build_rel_bias_for_tgt(tgt)

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        model_out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            return_attn_maps=capture_cross_attn,
            return_self_attn_maps=capture_self_attn,
            rel_bias=rel_bias,
        )

        attn_maps = None
        self_attn_maps = None
        if capture_cross_attn and capture_self_attn:
            out, attn_payload = model_out
            attn_maps = attn_payload.get("cross_attn")
            self_attn_maps = attn_payload.get("self_attn")
        elif capture_cross_attn:
            out, attn_maps = model_out
        elif capture_self_attn:
            out, attn_payload = model_out
            self_attn_maps = attn_payload.get("self_attn")
        else:
            out = model_out

        out = rearrange(out, "l b d -> b l d")
        embed_seq = out.detach() if capture_embed else None
        logits = self.proj(out)

        if return_aux:
            aux = {
                "embed_seq": embed_seq,
                "cross_attn": attn_maps,
                "self_attn": self_attn_maps,
                "height": h,
            }
            return logits, aux
        return logits


    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        return self(src[0], src_mask[0], input_ids)

