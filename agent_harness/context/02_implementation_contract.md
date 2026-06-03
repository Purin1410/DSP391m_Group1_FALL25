# 02 Implementation contract

## Activation and flags
Add config/kwargs support without breaking old calls:
```python
use_tree_state_cache = kwargs.pop("use_tree_state_cache", kwargs.pop("use_cache", True))
use_kv_cache = kwargs.pop("use_kv_cache", True)
```

Cached path is active only when:
```python
use_kv_cache_active = use_kv_cache and (not self.training) and (not torch.is_grad_enabled())
```
Otherwise use the existing full-prefix path.

## Cache object
Create `models/transformer/kv_cache.py` with a dataclass roughly:
```python
@dataclass
class DecoderKVCache:
    self_k: List[Tensor]      # [B_active, H, max_len, Hd]
    self_v: List[Tensor]
    cross_k: List[Tensor]     # [B_original, H, S, Hd]
    cross_v: List[Tensor]
    cross_pre_sum: List[Tensor]    # fp32 [B_active, H, S]
    cross_final_sum: List[Tensor]  # fp32 [B_active, H, S]
    beam_to_batch_idx: Tensor      # [B_active]
    memory_key_padding_mask: Tensor # [B_original, S]
    height: int
    cur_len: int
    max_len: int
    batch_size: int
    beam_size: int
    num_layers: int
    num_heads: int
    head_dim: int
```

`expand_beam_(beam_size)` expands only beam-indexed tensors: self K/V, ARM sums, and `beam_to_batch_idx`. Do not duplicate cross K/V. `reorder_(beam_idx)` reorders self K/V, ARM sums, and `beam_to_batch_idx`; cross K/V remains original-batch indexed.

## Tree relative bias
`TreeRelativeBias.forward` must accept any `(B, T, S)`, not only square `(B, L, L)`. For incremental decode use `rel_ids_step = rel_cache[:active, cur_len-1:cur_len, :cur_len]`. Square input output must remain exactly unchanged.

## Attention helpers
Keep `MultiheadAttention.forward(...)` signature compatible. Add helpers:
- `_project_q(query)`
- `_project_kv(key, value)`
- `project_static_kv(memory)`
- `forward_cached_self(query_step, cache_k, cache_v, write_pos, rel_bias_step=None)`
- `forward_cached_cross(query_step, static_k, static_v, key_padding_mask, arm_bias=None)`

Self-attention cache writes new K/V at `write_pos` and attends only to `:write_pos+1`; no causal mask needed. Cross-attention uses static K/V and active memory mask gathered by `beam_to_batch_idx`.

## ARM exactness
Original ARM semantics:
- `curr_attn` is pre-ARM current-layer attention.
- `prev_attn` is final previous-layer attention.
- ARM uses cumulative previous target positions only.

Add `AttentionRefinementModule.forward_from_sums(prev_attn_sum, curr_attn_sum, key_padding_mask, h, dtype)` returning `[B_active * H, 1, S]` equivalent to full `forward(... )[:, -1:, :]`.

Rules:
- `cross_pre_sum` and `cross_final_sum` are fp32.
- Compute ARM bias from sums before current token.
- Update sums after current token cross-attention is done.
- Layer 0 has no ARM; layer i>0 uses previous layer final sum and current layer pre sum.

## Decoder step
`Decoder.init_decode_cache(src, src_mask, max_len)`:
- flatten memory `[B,h,w,D] -> [S,B,D]` for projection,
- store `memory_key_padding_mask` as `[B,S]`,
- precompute per-layer cross K/V `[B,H,S,Hd]`,
- allocate self K/V `[B,H,max_len,Hd]`,
- allocate fp32 ARM sums `[B,H,S]`,
- set `beam_to_batch_idx = arange(B)` and `cur_len=0`.

`Decoder.transform_step(src, src_mask, token_ids, cache, rel_ids_step=None)`:
- embed current token only,
- add positional encoding row `cache.cur_len`,
- build `rel_bias_step` if tree bias is enabled,
- call `TransformerDecoder.forward_step`,
- project one-step hidden to logits,
- increment `cache.cur_len` once.

## Beam integration timeline
1. Initialize `input_ids` with SOS or EOS for R2L.
2. Build tree relation state/cache as existing code does.
3. Initialize KV cache.
4. Process the start token through `transform_step`; coverage is zero at pos 0, then sums update.
5. Select first generated token(s).
6. Expand beam cache and tree state/cache.
7. For each step: compute logits from `input_ids[:, -1]`; process beam; `kv_cache.reorder_(beam_idx)`; append selected token; then update tree relation state/cache for selected token.
8. Do not write `beam_next_tokens` to self-KV until the next `transform_step`.
9. Finalize using existing beam scorer semantics.

## Non-goals
- No training path rewrite.
- No SDPA/FlashAttention dependency.
- No quantization, eviction, sliding window, or PyTorch upgrade.
- No disabling ARM/tree-bias/bidirectional paths.
