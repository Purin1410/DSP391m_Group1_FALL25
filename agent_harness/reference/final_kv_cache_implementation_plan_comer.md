# Final Implementation Plan: KV-cache inference cho CoMER có ARM + Tree Bias

**Repo:** `Purin1410/DSP391m_Group1_FALL25`  
**Branch được phân tích:** `LiSRB`  
**Ngày tổng hợp:** 2026-06-03  
**Mục tiêu:** thêm KV-cache thật cho inference/beam search để tăng tốc decoding, **không đổi behavior training**, **không đổi kết quả inference**, và **không tắt ARM / tree-bias / bidirectional path**.

---

## 0. Tóm tắt quyết định cuối cùng

Sau khi tổng hợp implement plan ban đầu, review, và phản biện lại review, hướng implement cuối cùng là:

1. **Giữ nguyên full-prefix path cho training, validation loss và bidirectional rescoring `_rate()`.**
2. **Thêm inference-only incremental path**:
   - `Decoder.init_decode_cache(...)`
   - `Decoder.transform_step(...)`
   - `TransformerDecoder.forward_step(...)`
   - `TransformerDecoderLayer.forward_step(...)`
   - `MultiheadAttention.forward_cached_self(...)`
   - `MultiheadAttention.forward_cached_cross(...)`
3. **Cross-attention K/V cache là Phase 1**, không phải Phase 2.
4. **Self-attention K/V cache là Phase 2**.
5. **ARM không được tắt**; implement bằng **fp32 running coverage sums**:
   - `cross_pre_sum`
   - `cross_final_sum`
6. **Tree-bias cache hiện có phải được reuse**, nhưng `TreeRelativeBias` cần support non-square row `(B, 1, cur_len)`.
7. **Beam expand/reorder phải được implement rõ ràng**, với thứ tự:
   - process SOS/EOS start token
   - expand beam cache
   - mỗi step: compute logits từ last token -> beam process -> reorder cache -> append token vào `input_ids` -> update tree relation cache/state
8. **Cross K/V lưu theo batch, không theo beam**, để tránh duplicate memory.
9. **Fallback full-prefix `use_kv_cache=False` phải luôn tồn tại** để regression test exactness.

---

## 1. Căn cứ từ repo hiện tại

### 1.1 Config mặc định đang bật ARM và tree-bias

Trong `configs/crohme_config.yaml`, config mặc định có:

```yaml
use_tree_bias: true
tree_bias_layers: all
cross_coverage: true
self_coverage: true
beam_size: 10
max_len: 200
```

Điều này nghĩa là KV-cache phải support đầy đủ:

- causal decoder self-attention,
- encoder-decoder cross-attention,
- ARM cross/self coverage,
- tree relative bias,
- beam search,
- optional bidirectional mode.

Không được lách bằng cách tắt ARM hoặc tắt tree-bias.

### 1.2 Training/full-prefix path hiện tại

Training/validation loss đi qua:

```python
out_hat = self(batch.imgs, batch.mask, batch.tgt, rel_ids=batch.rel_ids)
```

sau đó:

```python
CoMER.forward(...)
  feature, mask = encoder(...)
  out = decoder(feature, mask, tgt, rel_ids=rel_ids)
```

`Decoder.forward(...)` hiện build full target sequence, full causal mask, full rel-bias, rồi chạy full `TransformerDecoder`.

**Kết luận:** không sửa behavior của path này. Tất cả KV-cache code phải nằm ở inference-only path.

### 1.3 Inference hiện tại chưa có KV-cache attention thật

Trong beam search hiện tại, mỗi step làm:

```python
next_token_logits = (
    self.transform(src, src_mask, input_ids, rel_ids=rel_ids)[:, -1, :] / temperature
)
```

`input_ids` là full prefix, nên mỗi step decoder recompute toàn bộ prefix rồi chỉ lấy logits vị trí cuối. Cache hiện tại chủ yếu là `rel_cache` / `relation_states` cho tree relation ids, **không phải KV-cache attention**.

### 1.4 ARM shape đã xác định: `cumsum(dim=2)` là qua target time T

Code ARM hiện tại:

```python
curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)

attns = []
if self.cross_coverage:
    attns.append(prev_attn)
if self.self_coverage:
    attns.append(curr_attn)
attns = torch.cat(attns, dim=1)

attns = attns.cumsum(dim=2) - attns
attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)
```

Sau `rearrange`, shape là:

```text
[B, H_or_2H, T, S]
```

Trong đó:

- `T` = target/query length,
- `S = h*w` = source spatial length.

Vì vậy `cumsum(dim=2)` là cumulative sum qua target positions trước đó. Review đúng khi yêu cầu test, nhưng nghi ngờ “cumsum qua source dimension” là không đúng với code hiện tại.

**Kết luận:** running sum qua decode steps là đúng hướng.

---

## 2. External reference constraints

### 2.1 KV-cache concept

KV-cache trong autoregressive generation lưu Key/Value của các token đã xử lý để tránh recompute attention prefix lặp lại. Hugging Face Transformers phân biệt `DynamicCache` và `StaticCache`; với repo này `max_len` cố định nên static pre-allocation phù hợp hơn.

Reference:

- https://huggingface.co/docs/transformers/kv_cache

### 2.2 PyTorch version constraint

Repo pin:

```yaml
python=3.7.16
pytorch=1.8.1
torchvision=0.9.1
cudatoolkit=11.1
```

Vì vậy **không dựa vào PyTorch 2.x `scaled_dot_product_attention`** trong implementation chính. Có thể ghi chú future optimization, nhưng không dùng làm blocker.

References:

- https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html
- https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://github.com/pytorch/pytorch/issues/119811

### 2.3 ARM cần attention weights

`scaled_dot_product_attention` trả output tensor, không trả attention weights trực tiếp. ARM của repo cần attention weights để tính coverage. Vì vậy nếu sau này upgrade PyTorch, SDPA/FlashAttention vẫn không thể thay thẳng cho ARM path nếu không có cách lấy attention weights.

---

## 3. Non-goals / phạm vi không làm

Không làm trong PR đầu:

1. Không đổi training forward/loss.
2. Không đổi teacher forcing target builder.
3. Không đổi beam scoring semantics.
4. Không đổi `_rate()` bidirectional rescoring.
5. Không quantize KV-cache.
6. Không implement cache eviction/sliding window.
7. Không upgrade PyTorch.
8. Không thay `MultiheadAttention.forward(...)` signature đang dùng bởi training.
9. Không tắt ARM, tree-bias, cross_coverage, self_coverage.

---

## 4. Design tổng thể

### 4.1 Public flags đề xuất

Trong config:

```yaml
inference:
  use_tree_state_cache: true
  use_kv_cache: true
  cache_cross_kv: true
  kv_cache_debug_exact: false
```

Backward compatibility:

```python
use_tree_state_cache = kwargs.pop("use_tree_state_cache", kwargs.pop("use_cache", True))
use_kv_cache = kwargs.pop("use_kv_cache", True)
```

### 4.2 Inference activation guard

KV-cache chỉ active khi:

```python
use_kv_cache_active = (
    use_kv_cache
    and not self.training
    and not torch.is_grad_enabled()
)
```

Nếu không thỏa guard, fallback full-prefix.

### 4.3 Cache object

Tạo file mới:

```text
models/transformer/kv_cache.py
```

Đề xuất dataclass:

```python
from dataclasses import dataclass
from typing import List
import torch
from torch import Tensor

@dataclass
class DecoderKVCache:
    # Self-attention: per active hypothesis / beam.
    # Stored as [B_active, H, max_len, Hd] for easier reorder.
    self_k: List[Tensor]
    self_v: List[Tensor]

    # Cross-attention: per original batch, not per beam.
    # Stored as [B_original, H, S, Hd].
    cross_k: List[Tensor]
    cross_v: List[Tensor]

    # ARM running sums: per active hypothesis / beam.
    # Stored fp32 as [B_active, H, S].
    cross_pre_sum: List[Tensor]
    cross_final_sum: List[Tensor]

    # Map each active hypothesis to original batch id.
    beam_to_batch_idx: Tensor

    memory_key_padding_mask: Tensor  # [B_original, S]
    height: int
    cur_len: int
    max_len: int
    batch_size: int
    beam_size: int
    num_layers: int
    num_heads: int
    head_dim: int
```

### 4.4 Mask strategy

`memory_key_padding_mask` hiện trong decoder full-prefix sau rearrange là:

```python
src_mask = rearrange(src_mask, "b h w -> b (h w)")
```

Cross K/V cache lưu theo original batch, nhưng ARM / attention runtime query theo active beams.

Trong `forward_step`, cần lấy mask theo active beams:

```python
active_memory_mask = memory_key_padding_mask.index_select(0, cache.beam_to_batch_idx)
```

Cross K/V vẫn lấy theo original batch:

```python
k_cross = cache.cross_k[layer_idx].index_select(0, cache.beam_to_batch_idx)
v_cross = cache.cross_v[layer_idx].index_select(0, cache.beam_to_batch_idx)
```

---

## 5. Code changes theo file

## 5.1 `models/transformer/tree_bias.py`

### Problem

`TreeRelativeBias.forward(...)` hiện yêu cầu square rel_ids `(B, L, L)`, nhưng incremental self-attention cần row:

```text
[B_active, 1, cur_len]
```

### Change

Sửa method để support generic `(B, T, S)`:

```python
class TreeRelativeBias(nn.Module):
    def forward(self, rel_ids: torch.LongTensor, flatten: bool = False) -> torch.Tensor:
        if rel_ids.dim() != 3:
            raise ValueError(f"rel_ids must be (B, T, S), got {tuple(rel_ids.shape)}")

        B, T, S = rel_ids.shape
        bias = self.emb(rel_ids).permute(0, 3, 1, 2).contiguous()

        if flatten:
            return bias.view(B * self.num_heads, T, S)

        return bias
```

### Exactness requirement

Full-prefix square input `(B, L, L)` phải cho output y hệt code cũ.

---

## 5.2 `models/transformer/attention.py`

### Strategy

Không đổi signature `MultiheadAttention.forward(...)`.

Thêm methods mới:

```python
def _project_q(self, query: Tensor) -> Tensor: ...
def _project_kv(self, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]: ...
def project_static_kv(self, memory: Tensor) -> Tuple[Tensor, Tensor]: ...
def forward_cached_self(...): ...
def forward_cached_cross(...): ...
```

Hoặc gộp thành một `forward_cached(...)`, nhưng tách self/cross sẽ dễ review hơn.

### Projection helpers

Với `_qkv_same_embed_dim=True`:

```python
q_weight = self.in_proj_weight[:embed_dim, :]
q_bias = self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None

kv_weight = self.in_proj_weight[embed_dim:, :]
kv_bias = self.in_proj_bias[embed_dim:] if self.in_proj_bias is not None else None
k, v = F.linear(memory, kv_weight, kv_bias).chunk(2, dim=-1)
```

Với `use_separate_proj_weight=True`, dùng `q_proj_weight`, `k_proj_weight`, `v_proj_weight`.

### Self-attention cached

Input:

```python
query_step: [1, B_active, D]
cache_k: [B_active, H, max_len, Hd]
cache_v: [B_active, H, max_len, Hd]
write_pos: int
rel_bias_step: Optional[[B_active*H, 1, write_pos+1]]
```

Output:

```python
attn_output: [1, B_active, D]
attention: [B_active*H, 1, write_pos+1]
```

Pseudo-code:

```python
q = project_q(query_step) * scaling
k_new, v_new = project_kv(query_step, query_step)

q = q.view(1, B, H, Hd).permute(1, 2, 0, 3)       # [B, H, 1, Hd]
k_new = k_new.view(1, B, H, Hd).permute(1, 2, 0, 3)
v_new = v_new.view(1, B, H, Hd).permute(1, 2, 0, 3)

cache_k[:, :, write_pos:write_pos+1, :] = k_new
cache_v[:, :, write_pos:write_pos+1, :] = v_new

k_all = cache_k[:, :, :write_pos+1, :]
v_all = cache_v[:, :, :write_pos+1, :]

dots = torch.matmul(q, k_all.transpose(-2, -1))
dots = dots.view(B * H, 1, write_pos + 1)

if rel_bias_step is not None:
    dots = dots + rel_bias_step.to(dots.dtype)

attention = softmax + dropout(training=self.training)
out = bmm(attention, v_all_flat)
out = out -> [1, B, D] -> out_proj
```

Không cần causal mask vì key range chỉ đến `write_pos`.

### Cross-attention cached

Input:

```python
query_step: [1, B_active, D]
static_k: [B_active, H, S, Hd]
static_v: [B_active, H, S, Hd]
key_padding_mask: [B_active, S]
arm_bias: Optional[[B_active*H, 1, S]]
```

Output:

```python
attn_output: [1, B_active, D]
pre_arm_attention: [B_active*H, 1, S]
final_attention: [B_active*H, 1, S]
```

Pseudo-code:

```python
q = project_q(query_step) * scaling
q = q.view(1, B, H, Hd).permute(1, 2, 0, 3)

dots = torch.matmul(q, static_k.transpose(-2, -1))
dots = dots.view(B * H, 1, S)

apply key_padding_mask

pre_attention = softmax(dots)

if arm_bias is not None:
    dots = dots - arm_bias.to(dots.dtype)
    apply key_padding_mask again
    final_attention = softmax(dots)
else:
    final_attention = pre_attention

out = bmm(final_attention, static_v_flat)
out = out -> [1, B, D] -> out_proj
```

---

## 5.3 `models/transformer/arm.py`

### Add method

```python
def forward_from_sums(
    self,
    prev_attn_sum: Optional[Tensor],  # [B_active, H, S], fp32
    curr_attn_sum: Optional[Tensor],  # [B_active, H, S], fp32
    key_padding_mask: Tensor,         # [B_active, S]
    h: int,
    dtype: torch.dtype,
) -> Tensor:
    """
    Return ARM correction with shape [B_active*H, 1, S].
    Equivalent to full forward(... )[:, -1:, :] when sums contain all previous target positions.
    """
```

### Implementation sketch

```python
attns = []
if self.cross_coverage:
    assert prev_attn_sum is not None
    attns.append(prev_attn_sum)
if self.self_coverage:
    assert curr_attn_sum is not None
    attns.append(curr_attn_sum)

attns = torch.cat(attns, dim=1).to(dtype)          # [B, H_or_2H, S]
B, _, S = attns.shape
attns = rearrange(attns, "b n (h w) -> b n h w", h=h)

mask = repeat(key_padding_mask, "b (h w) -> b () h w", h=h)

cov = self.conv(attns)
cov = self.act(cov)
cov = cov.masked_fill(mask, 0.0)
cov = self.proj(cov)
cov = self.post_norm(cov, mask)
cov = rearrange(cov, "b n h w -> (b n) 1 (h w)")
return cov
```

### Precision rule

`cross_pre_sum` and `cross_final_sum` must be fp32 even under fp16 inference. Convert to model dtype only inside `forward_from_sums`.

### Important ARM semantics

Original attention does:

```python
attention = mask_softmax_dropout(attn_output_weights)
if arm is not None:
    attn_output_weights -= arm(attention)
    attention = mask_softmax_dropout(attn_output_weights)
```

Therefore:

- `curr_attn` passed to ARM = **pre-ARM attention** of current layer.
- `prev_attn` passed to ARM = **final attention** returned by previous layer.
- Running sums must store both `cross_pre_sum[layer]` and `cross_final_sum[layer]`.

---

## 5.4 `models/transformer/transformer_decoder.py`

### Add `TransformerDecoder.forward_step`

Signature:

```python
def forward_step(
    self,
    tgt_step: Tensor,                 # [1, B_active, D]
    cache: DecoderKVCache,
    rel_bias_step: Optional[Tensor],  # [B_active*H, 1, cur_len]
) -> Tensor:
```

Pseudo-code:

```python
output = tgt_step

for layer_idx, layer in enumerate(self.layers):
    layer_rel_bias = rel_bias_step
    if self.tree_bias_layers == "last1" and layer_idx != self.num_layers - 1:
        layer_rel_bias = None

    output = layer.forward_step(
        tgt_step=output,
        cache=cache,
        layer_idx=layer_idx,
        rel_bias_step=layer_rel_bias,
        arm_module=self.arm if layer_idx > 0 else None,
    )

if self.norm is not None:
    output = self.norm(output)

return output
```

Full-prefix code passes `arm=None` for layer 0, then passes ARM from layer 1 onward using previous layer attention. Incremental path should mirror this.

### Add `TransformerDecoderLayer.forward_step`

Signature:

```python
def forward_step(
    self,
    tgt_step: Tensor,                 # [1, B_active, D]
    cache: DecoderKVCache,
    layer_idx: int,
    rel_bias_step: Optional[Tensor],
    arm_module: Optional[AttentionRefinementModule],
) -> Tensor:
```

Pseudo-code:

```python
pos = cache.cur_len
B_active = tgt_step.size(1)

# 1. Self-attention with KV cache
tgt2, _ = self.self_attn.forward_cached_self(
    query_step=tgt_step,
    cache_k=cache.self_k[layer_idx],
    cache_v=cache.self_v[layer_idx],
    write_pos=pos,
    rel_bias_step=rel_bias_step,
)
tgt = tgt_step + self.dropout1(tgt2)
tgt = self.norm1(tgt)

# 2. Cross-attention with static K/V
beam_to_batch = cache.beam_to_batch_idx
cross_k = cache.cross_k[layer_idx].index_select(0, beam_to_batch)
cross_v = cache.cross_v[layer_idx].index_select(0, beam_to_batch)
memory_mask = cache.memory_key_padding_mask.index_select(0, beam_to_batch)

arm_bias = None
if arm_module is not None:
    prev_sum = cache.cross_final_sum[layer_idx - 1]
    curr_sum = cache.cross_pre_sum[layer_idx]
    arm_bias = arm_module.forward_from_sums(
        prev_attn_sum=prev_sum,
        curr_attn_sum=curr_sum,
        key_padding_mask=memory_mask,
        h=cache.height,
        dtype=tgt.dtype,
    )

tgt2, pre_attn, final_attn = self.multihead_attn.forward_cached_cross(
    query_step=tgt,
    static_k=cross_k,
    static_v=cross_v,
    key_padding_mask=memory_mask,
    arm_bias=arm_bias,
)

# Update ARM sums AFTER using them for current token.
cache.cross_pre_sum[layer_idx] += pre_attn.view(B_active, cache.num_heads, 1, -1).squeeze(2).float()
cache.cross_final_sum[layer_idx] += final_attn.view(B_active, cache.num_heads, 1, -1).squeeze(2).float()

tgt = tgt + self.dropout2(tgt2)
tgt = self.norm2(tgt)

tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
tgt = tgt + self.dropout3(tgt2)
tgt = self.norm3(tgt)

return tgt
```

---

## 5.5 `models/decoder.py`

### Add `init_decode_cache`

Signature:

```python
def init_decode_cache(
    self,
    src: FloatTensor,      # [B, h, w, D]
    src_mask: LongTensor,  # [B, h, w]
    max_len: int,
) -> DecoderKVCache:
```

Tasks:

```python
h = src.shape[1]
memory = rearrange(src, "b h w d -> (h w) b d")
memory_mask = rearrange(src_mask, "b h w -> b (h w)")
```

Then:

1. Precompute cross K/V per layer: `[B, H, S, Hd]`.
2. Allocate self K/V per layer: `[B, H, max_len, Hd]`.
3. Allocate ARM sums: `[B, H, S]` with dtype `torch.float32`.
4. Set `beam_to_batch_idx = torch.arange(B, device=src.device)` and `cur_len = 0`.

### Add `transform_step`

Signature:

```python
def transform_step(
    self,
    src: List[FloatTensor],
    src_mask: List[LongTensor],
    token_ids: LongTensor,              # [B_active]
    cache: DecoderKVCache,
    rel_ids_step: Optional[LongTensor] = None,
) -> FloatTensor:
```

Pseudo-code:

```python
B_active = token_ids.size(0)
pos = cache.cur_len

# Embed current token only.
tgt = token_ids[:, None]        # [B, 1]
tgt = self.word_embed(tgt)      # [B, 1, D]

# Add position encoding row `pos` only.
tgt = tgt + self.pos_enc.pe[pos:pos+1, :][None, :, :]
tgt = self.norm(tgt)
tgt = rearrange(tgt, "b l d -> l b d")  # [1, B, D]

rel_bias_step = None
if self.use_tree_bias and self._tree_rel_bias is not None:
    assert rel_ids_step is not None
    rel_bias_step = self._tree_rel_bias(rel_ids_step, flatten=True)

out = self.model.forward_step(
    tgt_step=tgt,
    cache=cache,
    rel_bias_step=rel_bias_step,
)

out = rearrange(out, "l b d -> b l d")
logits = self.proj(out)[:, 0, :]

cache.cur_len += 1
return logits
```

---

## 5.6 `utils/generation_utils.py` / beam search integration

Actual beam search appears in `DecodeModel._beam_search(...)`.

### Rename / add flags

```python
def _beam_search(
    ...,
    use_tree_state_cache: bool = True,
    use_kv_cache: bool = True,
    ...
):
```

Maintain legacy:

```python
if "use_cache" exists in call path:
    use_tree_state_cache = use_cache
```

### SOS warm-up sequence

Current code starts with:

```python
input_ids = torch.full((batch_size, 1), fill_value=sos_or_eos)
cur_len = 1
```

With KV-cache active:

```python
kv_cache = self.init_decode_cache(src[0], src_mask[0], max_len=max_len)

start_token_ids = input_ids[:, 0]  # [B]
rel_ids_step = None
if use_tree_state_cache:
    rel_ids_step = rel_cache[:B, 0:1, 0:1]

next_token_logits = self.transform_step(
    src, src_mask,
    token_ids=start_token_ids,
    cache=kv_cache,
    rel_ids_step=rel_ids_step,
) / temperature
# kv_cache.cur_len == 1
```

Then process top-k for first generated token and expand beam.

### Beam expansion

After `cur_len == 1` expansion of `input_ids`, `src`, `src_mask`, and tree relation cache/state, call:

```python
kv_cache.expand_beam_(beam_size)
```

If cross K/V is batch-indexed:

- do **not** expand `cross_k/v`.
- expand only:
  - `self_k/v`
  - `cross_pre_sum`
  - `cross_final_sum`
  - `beam_to_batch_idx`

### Main loop after SOS

At loop step for current prefix:

```python
last_token_ids = input_ids[:, -1]

rel_ids_step = None
if use_tree_state_cache:
    rel_ids_step = rel_cache[:active, cur_len-1:cur_len, :cur_len]

next_token_logits = self.transform_step(
    src, src_mask,
    token_ids=last_token_ids,
    cache=kv_cache,
    rel_ids_step=rel_ids_step,
) / temperature
```

### Reorder order

After `beam_scorer.process(...)` returns `beam_idx` and `beam_next_tokens`:

```python
if use_kv_cache_active:
    kv_cache.reorder_(beam_idx)

input_ids = torch.cat(
    (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)),
    dim=-1,
)

# Then update relation_states and rel_cache with beam_next_tokens.
```

Do not write `beam_next_tokens` into self-KV immediately. It will be processed at the next loop iteration.

---

## 6. Cache methods detail

## 6.1 `expand_beam_`

```python
def expand_beam_(self, beam_size: int) -> None:
    B = self.batch_size
    H = self.num_heads

    for i in range(self.num_layers):
        self.self_k[i] = (
            self.self_k[i]
            .unsqueeze(1)                         # [B, 1, H, max_len, Hd]
            .expand(-1, beam_size, -1, -1, -1)
            .reshape(B * beam_size, H, self.max_len, self.head_dim)
            .contiguous()
        )
        self.self_v[i] = (
            self.self_v[i]
            .unsqueeze(1)
            .expand(-1, beam_size, -1, -1, -1)
            .reshape(B * beam_size, H, self.max_len, self.head_dim)
            .contiguous()
        )
        self.cross_pre_sum[i] = (
            self.cross_pre_sum[i]
            .unsqueeze(1)                         # [B, 1, H, S]
            .expand(-1, beam_size, -1, -1)
            .reshape(B * beam_size, H, -1)
            .contiguous()
        )
        self.cross_final_sum[i] = (
            self.cross_final_sum[i]
            .unsqueeze(1)
            .expand(-1, beam_size, -1, -1)
            .reshape(B * beam_size, H, -1)
            .contiguous()
        )

    self.beam_to_batch_idx = (
        torch.arange(B, device=self.beam_to_batch_idx.device)
        .repeat_interleave(beam_size)
    )
    self.beam_size = beam_size
```

## 6.2 `reorder_`

```python
def reorder_(self, beam_idx: Tensor) -> None:
    """
    beam_idx: [B_active], indexes current active hypotheses.
    """
    for i in range(self.num_layers):
        self.self_k[i] = self.self_k[i].index_select(0, beam_idx).contiguous()
        self.self_v[i] = self.self_v[i].index_select(0, beam_idx).contiguous()

        self.cross_pre_sum[i] = self.cross_pre_sum[i].index_select(0, beam_idx).contiguous()
        self.cross_final_sum[i] = self.cross_final_sum[i].index_select(0, beam_idx).contiguous()

    self.beam_to_batch_idx = self.beam_to_batch_idx.index_select(0, beam_idx).contiguous()
```

Because cache tensors are stored as `[B_active, H, ...]`, no flatten/unflatten by head is needed. This is less error-prone than `[B_active*H, ...]`.

---

## 7. Exactness notes and pitfalls

### 7.1 Dropout

Cached path must run only in eval/no-grad. In train mode, dropout makes full-prefix vs incremental behavior non-equivalent.

### 7.2 BatchNorm inside ARM

`MaskBatchNorm2d` wraps `BatchNorm1d`. In eval mode it uses running stats, so processing one step vs full prefix should be equivalent if input values match. In train mode it would not be equivalent.

### 7.3 ARM sums update order

Use previous sums to compute ARM correction for current token. Update sums only after current token cross-attention is fully computed.

Wrong:

```python
sum += current_attn
arm(sum)
```

Correct:

```python
arm(sum_before_current)
sum += current_attn
```

### 7.4 Pre vs final attention

`curr_attn` for ARM is pre-ARM attention of the current layer. `prev_attn` for ARM is final attention of previous layer. Store both.

### 7.5 Cross K/V per batch

Cross K/V must not be reordered by beam. Only `beam_to_batch_idx` changes.

### 7.6 Tree relation row

For incremental step at prefix length `cur_len`, use:

```python
rel_ids_step = rel_cache[:active, cur_len-1:cur_len, :cur_len]
```

This is the row for the current last token attending to all previous/current keys.

### 7.7 Bidirectional mode

Keep `_rate()` full-prefix. In beam search, initial token differs:

- L2R starts with SOS.
- R2L starts with EOS in current repo logic.

KV-cache should operate after existing bidirectional duplication/state initialization, not replace it.

---

## 8. Revised implementation order

### Step 1 — `TreeRelativeBias` non-square support

Files:

- `models/transformer/tree_bias.py`

Deliverables:

- Allow `(B, T, S)`.
- Square full-prefix behavior unchanged.

Tests:

- old square rel_ids output exact.
- row rel_ids `(B, 1, S)` works.

---

### Step 2 — Add KV cache dataclass

Files:

- `models/transformer/kv_cache.py`

Deliverables:

- `DecoderKVCache`
- `expand_beam_`
- `reorder_`

Tests:

- known tensor content expand/reorder correctness.
- duplicate parent beam case, e.g. `beam_idx=[2,0,1,2]`.

---

### Step 3 — Cross K/V static cache

Files:

- `models/transformer/attention.py`
- `models/decoder.py`

Deliverables:

- `project_static_kv(memory)`
- `Decoder.init_decode_cache(...)`
- Cross K/V stored `[B, H, S, Hd]`.

Tests:

- For a fixed query, cached cross-attention output == original cross-attention output with full memory projection.
- Masked padded source positions remain ignored.

---

### Step 4 — Self-attention incremental KV

Files:

- `models/transformer/attention.py`
- `models/transformer/transformer_decoder.py`

Deliverables:

- `forward_cached_self(...)`
- `TransformerDecoderLayer.forward_step(...)` self-attn section.

Tests:

- no ARM, no tree-bias: full-prefix last hidden == incremental hidden for every t.
- with tree-bias row: full-prefix last hidden == incremental hidden.

---

### Step 5 — ARM running sums

Files:

- `models/transformer/arm.py`
- `models/transformer/transformer_decoder.py`

Deliverables:

- `forward_from_sums(...)`
- pre/final sums.

Tests:

- `forward_from_sums` equals full ARM last row.
- Full decoder logits with ARM enabled: incremental logits == full-prefix logits.

---

### Step 6 — Integrate into beam search

Files:

- `utils/generation_utils.py`
- maybe `models/decoder.py`

Deliverables:

- `use_kv_cache` flag.
- SOS warm-up.
- Beam expand/reorder integration.
- Fallback path.

Tests:

- `beam_size=1`: sequence/logits exact.
- `beam_size=3/10`: hypotheses identical.
- L2R and bidirectional.
- tree_bias on/off.
- ARM on/off test harness.

---

### Step 7 — Benchmark

Metrics:

- wall-clock decode time per batch.
- tokens/sec or formulas/sec.
- GPU memory peak.
- exactness mismatch count.
- speedup with:
  - no cache,
  - cross KV only,
  - cross + self KV,
  - cross + self KV + ARM sums.

---

## 9. Test plan chi tiết

### 9.1 Unit: TreeRelativeBias

```python
def test_tree_relative_bias_square_unchanged():
    ...

def test_tree_relative_bias_row_shape():
    rel_ids = torch.randint(0, num_rel, (B, 1, S))
    bias = module(rel_ids, flatten=True)
    assert bias.shape == (B * H, 1, S)
```

### 9.2 Unit: cross K/V projection

```python
def test_cross_kv_static_matches_full_projection():
    ...
```

### 9.3 Unit: self KV exactness

```python
def test_self_attention_cached_matches_full_prefix():
    ...
```

### 9.4 Unit: ARM coverage dimension / sums

```python
def test_arm_forward_from_sums_matches_full_last_row():
    arm.eval()
    prev = torch.rand(B*H, T, S)
    curr = torch.rand(B*H, T, S)

    full = arm(prev, mask, h, curr)

    prev_sum = prev.view(B, H, T, S)[:, :, :T-1, :].sum(dim=2)
    curr_sum = curr.view(B, H, T, S)[:, :, :T-1, :].sum(dim=2)

    step = arm.forward_from_sums(prev_sum, curr_sum, mask, h, dtype=prev.dtype)

    assert_allclose(step, full[:, -1:, :], atol=1e-5, rtol=1e-5)
```

### 9.5 Unit: SOS coverage

```python
def test_sos_has_zero_coverage_then_updates_sums():
    ...
```

### 9.6 Integration: decoder step exactness

```python
def test_decoder_transform_step_matches_transform_last_logits():
    for t in range(1, L + 1):
        full_logits = decoder.transform(src, src_mask, ids[:, :t], rel_ids=rel[:, :t, :t])[:, -1, :]
        step_logits = decoder.transform_step(... current token ...)
        assert_allclose(step_logits, full_logits, atol=1e-5, rtol=1e-5)
```

### 9.7 Integration: beam exactness

```python
def test_beam_search_kv_cache_exact():
    hyps_full = model.beam_search(..., use_kv_cache=False)
    hyps_cache = model.beam_search(..., use_kv_cache=True)
    assert [h.seq for h in hyps_cache] == [h.seq for h in hyps_full]
    assert_allclose([h.score ...], [h.score ...])
```

### 9.8 Reorder exactness

```python
def test_cache_reorder_with_duplicate_parent():
    beam_idx = torch.tensor([2, 0, 1, 2], device=device)
    ...
```

### 9.9 fp16/fp32 ARM sums drift

```python
def test_arm_sums_fp32_are_closer_than_fp16():
    ...
```

---

## 10. Suggested PR breakdown

### PR 1 — Safe primitives

- `TreeRelativeBias` non-square.
- `DecoderKVCache`.
- `expand_beam_` / `reorder_`.
- Tests only, no beam behavior change.

### PR 2 — Cross K/V static cache

- Attention projection helpers.
- `init_decode_cache`.
- Tests against original cross-attention.

### PR 3 — Self-attention incremental

- `forward_cached_self`.
- `forward_step` skeleton.
- Test no-ARM exactness.

### PR 4 — ARM incremental

- `forward_from_sums`.
- pre/final sums.
- Full decoder step exactness.

### PR 5 — Beam integration

- `use_kv_cache` flag.
- SOS warm-up.
- expand/reorder.
- Exactness tests.

### PR 6 — Benchmark + docs

- Add benchmark script.
- Add README section.
- Add known caveats.

---

## 11. Benchmark script outline

Create:

```text
scripts/benchmark_kv_cache.py
```

CLI:

```bash
python scripts/benchmark_kv_cache.py \
  --config configs/crohme_config.yaml \
  --checkpoint path/to.ckpt \
  --batch-size 4 \
  --beam-size 10 \
  --max-len 200 \
  --warmup 5 \
  --iters 20
```

Report:

```text
mode                      latency_ms   speedup   peak_mem_MB   exact_match
full_prefix               ...
cross_kv_only             ...
cross_self_kv_no_arm      ...
full_kv_cache_with_arm    ...
```

Exactness mode:

```bash
python scripts/benchmark_kv_cache.py --check-exact
```

---

## 12. Final checklist trước merge

- [ ] Training loss path unchanged.
- [ ] Validation loss path unchanged.
- [ ] `_rate()` unchanged/full-prefix.
- [ ] `use_kv_cache=False` exactly reproduces old inference.
- [ ] `use_kv_cache=True` beam output identical to fallback.
- [ ] ARM enabled exactness passes.
- [ ] Tree-bias enabled exactness passes.
- [ ] Bidirectional mode exactness passes.
- [ ] Beam duplicate parent reorder test passes.
- [ ] SOS warm-up test passes.
- [ ] fp16 uses fp32 ARM sums.
- [ ] No dependency on PyTorch 2.x SDPA.
- [ ] Config flags documented.
- [ ] Benchmark shows measurable latency improvement.

---

## 13. Appendix: high-level decode timeline

```text
Encode image once.
Flatten memory once.
Precompute cross K/V once per decoder layer.

Init input_ids = [SOS] or [EOS for R2L].
Init tree relation state/cache.
Init KV-cache cur_len = 0.

Process start token with transform_step:
  write self K/V at pos 0
  cross-attend with static K/V
  ARM coverage is zero
  update ARM sums
  cur_len = 1

Select first generated tokens.
Expand beam:
  input_ids repeat
  tree cache/state repeat
  self K/V repeat
  ARM sums repeat
  cross K/V stays batch-indexed
  beam_to_batch_idx repeat_interleave

For each next step:
  token = input_ids[:, -1]
  rel row = rel_cache[:, cur_len-1:cur_len, :cur_len]
  logits = transform_step(token, rel row)
  beam_scorer.process
  reorder KV-cache by beam_idx
  append selected token to input_ids
  update tree relation state/cache with selected token
  cur_len += 1

Finalize beam.
If bidirectional mode: keep final rescoring full-prefix.
```

---

## 14. Appendix: why not SDPA in first implementation

Do not use `torch.nn.functional.scaled_dot_product_attention` in first implementation because:

1. Repo environment is PyTorch 1.8.1.
2. ARM needs attention weights.
3. PyTorch SDPA returns output tensor, not attention weights.
4. Current attention code already has custom ARM and rel-bias behavior.

Keep manual `bmm -> mask -> softmax -> bmm` path for exactness.

---

## 15. Appendix: accepted / rejected review points

### Accepted from review

- Cross K/V must be Phase 1.
- Cross K/V should be batch-indexed, not beam-indexed.
- Need explicit `expand_beam_`.
- Need explicit `reorder_`.
- Need explicit SOS warm-up.
- Need fp32 ARM sums.
- Need flag rename to avoid `use_cache` confusion.
- Need ARM equivalence tests.

### Rejected / modified from review

- The claim that ARM `cumsum(dim=2)` may be source dimension is rejected for current repo code; after `rearrange`, `dim=2` is target time `T`.
- Subclass/swap `CachedMultiheadAttention` is not required. Safer plan is to keep `forward()` unchanged and add `forward_cached_*` helper methods.
- SDPA is a future optimization only, not part of the first implementation.

---

## 16. Source references

Repository / review sources:

- Uploaded consolidated repo file: `Purin1410-DSP391m_Group1_FALL25-2026-06-03T16-10-30.md`
- Uploaded review file: `kv_cache_plan_review.md`

External references:

- Hugging Face Transformers KV-cache documentation: https://huggingface.co/docs/transformers/kv_cache
- PyTorch MultiheadAttention documentation: https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html
- PyTorch scaled_dot_product_attention documentation: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch issue: SDPA does not return attention weights: https://github.com/pytorch/pytorch/issues/119811
