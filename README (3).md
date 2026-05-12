# Coding Agent Instructions: Port Tree-Structure Relative Bias into Refactored CoMER

## Goal

Implement the custom method from the read-only reference folder `comer/` into the refactored CoMER codebase at `DSP391m_Group1_FALL25/`.

The method is a **tree-structure relative bias for decoder self-attention**. It builds a lightweight LaTeX structural tree from target token ids, converts pairwise token relations into relation ids, maps those ids to per-head learnable bias values, and adds the bias to decoder **self-attention logits before masking/softmax**.

## Source of Truth

Use these inputs exactly:

1. `comer/`  
   Read-only reference. It contains the working baseline CoMER plus my custom tree relative bias method. You may inspect it, but **do not edit anything inside `comer/`**.

2. `DSP391m_Group1_FALL25/`  
   Target repo. This is the refactored/faster CoMER implementation. Implement the method here.

Important: the old `comer/` method may use older patterns such as global vocabulary access through `CROHMEDatamodule.shared_vocab`. **Do not copy that design into the refactored repo.** Preserve the refactor's explicit `VocabInfo` flow.

## Non-Negotiable Constraints

- Do not modify files under `comer/`.
- Do not revert refactor optimizations in the target repo.
- Do not introduce global vocabulary imports in model code.
- Keep `vocab_info` as the source for ids and vocabulary metadata.
- Preserve existing bidirectional training and beam-search behavior.
- Apply the tree relative bias only to decoder **self-attention**, not encoder-decoder cross-attention.
- Add the bias to attention logits after `QK^T` and before causal/padding masks, softmax, dropout, and ARM refinement.
- Initialize the relative bias embedding with zeros so the model starts from baseline behavior.
- Keep the implementation compatible with the existing Python/PyTorch style of the repo.

## Files to Inspect First

Read these reference files in `comer/`:

- `comer/models/transformer/tree_bias.py`
- `comer/models/decoder.py`
- `comer/models/comer.py`
- `comer/models/transformer/attention.py`
- `comer/models/transformer/transformer_decoder.py`
- `comer/configs/crohme_config.yaml`

Then edit only the corresponding target files in `DSP391m_Group1_FALL25/`.

## Target Files to Change

Implement changes in these target files:

- `configs/crohme_config.yaml`
- `models/comer.py`
- `models/decoder.py`
- `models/transformer/tree_bias.py` new file
- `models/transformer/attention.py`
- `models/transformer/transformer_decoder.py`

Only touch other files if a test or import requires it.

## Implementation Plan

### 1. Add `models/transformer/tree_bias.py`

Port the reference implementation from `comer/models/transformer/tree_bias.py` into the target repo.

This file should define:

- relation type constants:
  - `TYPE_ROOT = 0`
  - `TYPE_SUP = 1`
  - `TYPE_SUB = 2`
  - `TYPE_NUM = 3`
  - `TYPE_DEN = 4`
  - `TYPE_OTHER = 5`
- `distance_bucket_tensor(d, num_buckets)`
- `_CtxMark` dataclass
- `TreeRelationBuilder`
- `TreeRelativeBias`

Expected behavior:

- Input to `TreeRelationBuilder.build`: `tgt_ids` with shape `[B, L]`.
- Output from `TreeRelationBuilder.build`: relation ids with shape `[B, L, L]`, same device as `tgt_ids`.
- `TreeRelativeBias.forward(rel_ids)` returns bias with shape `[B, H, L, L]`.
- `TreeRelativeBias.forward(rel_ids, flatten=True)` may return `[B * H, L, L]` if useful.
- Bias embedding must be `nn.Embedding(num_relations, num_heads)` and initialized with `nn.init.zeros_(self.emb.weight)`.

Keep the fast design from the reference:

- Parse token paths with a lightweight stack parser.
- Move target ids to CPU once for parsing only.
- Compute pairwise LCP, tree distance, buckets, and relation ids with vectorized torch ops on the original target device.
- Mask pairs involving PAD to relation id `0`.

### 2. Add config flags

In `configs/crohme_config.yaml`, inside `model:`, add these decoder options near the existing decoder settings:

```yaml
  # -------- tree relative bias --------
  use_tree_bias: true
  tree_bias_num_buckets: 16
  tree_bias_mode: full        # dist_only | type_only | full
  tree_bias_layers: all       # all | last1
  tree_bias_rel_set: full     # supsub | supsub_frac | full
```

### 3. Wire config through `models/comer.py`

In `CoMER.__init__`, read these values from `mcfg`:

```python
use_tree_bias = mcfg.get("use_tree_bias", True)
tree_bias_num_buckets = mcfg.get("tree_bias_num_buckets", 16)
tree_bias_mode = mcfg.get("tree_bias_mode", "full")
tree_bias_layers = mcfg.get("tree_bias_layers", "all")
tree_bias_rel_set = mcfg.get("tree_bias_rel_set", "full")
```

Pass them into `Decoder(...)` together with the existing arguments and `vocab_info`.

Do not change encoder behavior, feature duplication, beam search, or the public `forward` signature.

### 4. Update `models/decoder.py`

Import the new classes:

```python
from .transformer.tree_bias import TreeRelationBuilder, TreeRelativeBias
```

Extend `_build_transformer_decoder(...)` with:

```python
tree_bias_layers: str = "all"
```

and pass it into `TransformerDecoder(..., tree_bias_layers=tree_bias_layers)`.

Extend `Decoder.__init__` with:

```python
use_tree_bias: bool = True,
tree_bias_num_buckets: int = 16,
tree_bias_mode: str = "full",
tree_bias_layers: str = "all",
tree_bias_rel_set: str = "full",
```

Preserve these refactored design points:

- Keep `self.vocab_info = vocab_info`.
- Keep `nn.Embedding(vocab_info.vocab_size, d_model)`.
- Keep `self.proj = nn.Linear(d_model, vocab_info.vocab_size)`.
- Keep `_causal_mask_cache` and `_build_attention_mask(length, device=None, dtype=torch.bool)`.
- Keep `tgt_pad_mask = tgt == self.vocab_info.pad_id`.

Build tree-bias modules like this, using `vocab_info`, not global datamodule state:

```python
self.use_tree_bias = bool(use_tree_bias)
self.tree_bias_layers = tree_bias_layers

if self.use_tree_bias:
    if vocab_info is None or vocab_info.words is None or not hasattr(vocab_info.words, "idx2word"):
        raise ValueError("Tree bias requires vocab_info.words.idx2word")

    self._tree_builder = TreeRelationBuilder(
        id2tok=vocab_info.words.idx2word,
        pad_id=vocab_info.pad_id,
        num_buckets=tree_bias_num_buckets,
        mode=tree_bias_mode,
        rel_set=tree_bias_rel_set,
    )
    self._tree_rel_bias = TreeRelativeBias(
        num_heads=nhead,
        num_relations=self._tree_builder.num_relations,
    )
else:
    self._tree_builder = None
    self._tree_rel_bias = None
```

In `Decoder.forward`, compute `rel_bias` before token embedding, while `tgt` is still token ids:

```python
rel_bias = None
if self.use_tree_bias and self._tree_builder is not None:
    rel_ids = self._tree_builder.build(tgt)      # [B, L, L]
    rel_bias = self._tree_rel_bias(rel_ids)      # [B, H, L, L]
```

Then pass it into the transformer decoder:

```python
out = self.model(
    tgt=tgt,
    memory=src,
    height=h,
    tgt_mask=tgt_mask,
    tgt_key_padding_mask=tgt_pad_mask,
    memory_key_padding_mask=src_mask,
    rel_bias=rel_bias,
)
```

### 5. Update `models/transformer/transformer_decoder.py`

Extend `TransformerDecoder.__init__`:

```python
tree_bias_layers: str = "all"
```

Store it:

```python
self.tree_bias_layers = tree_bias_layers
```

Extend `TransformerDecoder.forward(...)` with:

```python
rel_bias: Optional[Tensor] = None
```

Inside the layer loop, choose whether the current layer receives the bias:

```python
layer_rel_bias = rel_bias
if self.tree_bias_layers == "last1" and i != self.num_layers - 1:
    layer_rel_bias = None
```

Pass `layer_rel_bias` into the decoder layer call.

Extend `TransformerDecoderLayer.forward(...)` with:

```python
rel_bias: Optional[Tensor] = None
```

Pass it only to `self.self_attn(...)`:

```python
tgt2 = self.self_attn(
    tgt,
    tgt,
    tgt,
    attn_mask=tgt_mask,
    key_padding_mask=tgt_key_padding_mask,
    rel_bias=rel_bias,
)[0]
```

Do not pass `rel_bias` to `self.multihead_attn(...)`, because that is cross-attention.

### 6. Update `models/transformer/attention.py`

Extend `MultiheadAttention.forward(...)` with:

```python
rel_bias: Optional[Tensor] = None
```

Forward this argument into `multi_head_attention_forward(...)` in all branches, including the `use_separate_proj_weight=True` branch.

Extend `multi_head_attention_forward(...)` with:

```python
rel_bias: Optional[Tensor] = None
```

After:

```python
attn_output_weights = torch.bmm(q, k.transpose(1, 2))
```

and before `mask_softmax_dropout(...)`, add validated relative bias support:

```python
if rel_bias is not None:
    if rel_bias.dim() == 3:
        if rel_bias.size(0) == bsz:
            rel_bias = rel_bias.unsqueeze(1).expand(bsz, num_heads, tgt_len, src_len)
            rel_bias = rel_bias.contiguous().view(bsz * num_heads, tgt_len, src_len)
        elif rel_bias.size(0) == bsz * num_heads:
            pass
        else:
            raise RuntimeError(
                f"rel_bias has invalid first dim: {rel_bias.size(0)} "
                f"expected {bsz} or {bsz * num_heads}"
            )
    elif rel_bias.dim() == 4:
        if rel_bias.size(0) == bsz and rel_bias.size(1) == num_heads:
            rel_bias = rel_bias.contiguous().view(bsz * num_heads, tgt_len, src_len)
        else:
            raise RuntimeError(
                f"rel_bias has invalid shape: {tuple(rel_bias.shape)}; "
                f"expected ({bsz}, {num_heads}, {tgt_len}, {src_len})"
            )
    else:
        raise RuntimeError(f"rel_bias must have dim 3 or 4, got {rel_bias.dim()}")

    if rel_bias.size(1) != tgt_len or rel_bias.size(2) != src_len:
        raise RuntimeError(
            f"rel_bias length mismatch: got {tuple(rel_bias.shape)}, "
            f"expected (*, {tgt_len}, {src_len})"
        )

    attn_output_weights = attn_output_weights + rel_bias.to(dtype=attn_output_weights.dtype)
```

Keep the existing attention mask, key padding mask, softmax, dropout, and ARM behavior unchanged after this point.

## Expected Data Flow

Training forward path:

```text
Batch.tgt [2B, L]
  -> Decoder.forward(..., tgt)
  -> TreeRelationBuilder.build(tgt) gives rel_ids [2B, L, L]
  -> TreeRelativeBias(rel_ids) gives rel_bias [2B, H, L, L]
  -> TransformerDecoder.forward(..., rel_bias)
  -> each selected TransformerDecoderLayer
  -> self_attn(..., rel_bias)
  -> attention logits = QK^T + rel_bias
  -> causal mask / PAD mask / softmax / dropout
```

Beam-search path:

```text
input_ids full prefix [batch * beams, current_len]
  -> Decoder.transform(...)
  -> Decoder.forward(..., input_ids)
  -> tree bias is recomputed for the current prefix
```

This is expected. Do not cache tree relations across decode steps unless you implement it safely and prove correctness.

## Important Pitfalls to Avoid

- Do not copy old commented code blocks from `comer/`.
- Do not import `CROHMEDatamodule` inside `models/decoder.py`.
- Do not use `vocab.PAD_IDX`; use `self.vocab_info.pad_id`.
- Do not change `utils/generation_utils.py` unless absolutely necessary.
- Do not apply tree bias to cross-attention.
- Do not add bias after softmax; it must be added to logits before softmax.
- Do not use `.tolist()` on large GPU tensors inside hot loops. The reference builder copies token ids to CPU once for lightweight parsing, then uses vectorized torch ops.
- Do not break CPU execution. The smoke tests should run without GPU.

## Validation

Run at least these checks from the target repo root:

```bash
python -m py_compile \
  models/transformer/tree_bias.py \
  models/transformer/attention.py \
  models/transformer/transformer_decoder.py \
  models/decoder.py \
  models/comer.py

python test_beam.py
```

Also run a small tree-bias smoke test:

```bash
python - <<'PY'
import torch
from models.transformer.tree_bias import TreeRelationBuilder, TreeRelativeBias

id2tok = {
    0: '<pad>', 1: '<sos>', 2: '<eos>', 3: 'x', 4: '^', 5: '{', 6: '2', 7: '}',
    8: '\\frac', 9: 'a', 10: 'b'
}
ids = torch.tensor([[1, 3, 4, 5, 6, 7, 2, 0]], dtype=torch.long)
builder = TreeRelationBuilder(id2tok=id2tok, pad_id=0, num_buckets=16, mode='full', rel_set='full')
rel_ids = builder.build(ids)
bias = TreeRelativeBias(num_heads=8, num_relations=builder.num_relations)(rel_ids)
assert rel_ids.shape == (1, 8, 8), rel_ids.shape
assert bias.shape == (1, 8, 8, 8), bias.shape
assert torch.isfinite(bias).all()
print('tree bias smoke test passed')
PY
```

If training data is available, run one minimal train/validation step with the normal config to confirm the complete model path works.

## Acceptance Criteria

The task is complete only if:

- `models/transformer/tree_bias.py` exists in the target repo.
- Tree-bias config flags exist in `configs/crohme_config.yaml`.
- `CoMER` passes tree-bias config values into `Decoder`.
- `Decoder` builds relation ids from raw target token ids and passes per-head `rel_bias` into the transformer decoder.
- `TransformerDecoder` can apply the bias to all layers or only the last layer according to `tree_bias_layers`.
- `TransformerDecoderLayer` applies the bias only in decoder self-attention.
- `MultiheadAttention` adds `rel_bias` to attention logits before masks/softmax.
- No code under `comer/` is modified.
- No global datamodule vocabulary dependency is introduced.
- `python test_beam.py` and `py_compile` pass.
