# 00 Task brief

Implement true KV-cache for CoMER inference/beam search in `Purin1410/DSP391m_Group1_FALL25` branch `LiSRB`.

Current issue: beam search calls `self.transform(src, src_mask, input_ids, rel_ids=rel_ids)[:, -1, :]` at each step, so the decoder recomputes the whole prefix repeatedly. Existing tree `rel_cache` / `relation_states` are not attention KV-cache.

Goal: add an inference-only incremental path that reuses attention K/V while producing the same logits, sequences, and beam scores as the full-prefix fallback.

Must preserve:
- training forward/loss behavior,
- validation loss behavior,
- full-prefix `Decoder.forward(...)`,
- `_rate()` bidirectional rescoring,
- ARM coverage behavior,
- tree relative bias behavior,
- beam scoring semantics.

Core APIs to add:
- `Decoder.init_decode_cache(...)`
- `Decoder.transform_step(...)`
- `TransformerDecoder.forward_step(...)`
- `TransformerDecoderLayer.forward_step(...)`
- `MultiheadAttention.forward_cached_self(...)`
- `MultiheadAttention.forward_cached_cross(...)`
- `AttentionRefinementModule.forward_from_sums(...)`
- `DecoderKVCache.expand_beam_()` and `.reorder_()`

Recommended implementation order:
1. `TreeRelativeBias` supports non-square `(B, T, S)`.
2. Add `DecoderKVCache` and tests for expand/reorder.
3. Add static cross K/V projection/cache.
4. Add incremental self-attention K/V.
5. Add ARM running sums.
6. Integrate beam search.
7. Add exactness tests and lightweight benchmark.
