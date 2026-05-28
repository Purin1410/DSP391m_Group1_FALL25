# Optimize Beam-Time Tree Relative Bias Cache

## Goal

Make bidirectional tree-relative bias fast during beam search without changing model semantics, target order, training loss, or decoding results.

Current code already added beam-time relation states, but the hot loop still has avoidable CPU overhead:

- `materialize_state()` converts `rel_list` to `torch.tensor(...)` at every decode step.
- `clone_state()` copies a Python list of length `max_len * max_len` for every beam child.
- `_rate()` still calls `transform(..., rel_ids=None)`, so it silently falls back to full relation-id building.
- Tests are mostly smoke tests, not exact equivalence tests.

Fix these while preserving exact output equivalence.

## Files to edit

Main files:

- `models/transformer/tree_bias.py`
- `utils/generation_utils.py`
- `models/decoder.py` only if needed
- `test_beam.py` or new `tests/test_beam_rel_cache.py`
- optional benchmark: `tests/benchmark_beam_rel_cache.py`

Do not change:

- target order `[L2R batch; R2L batch]`
- `TreeRelationBuilder.build()`
- `CausalR2LTreeRelationBuilder.build()`
- attention math
- beam scoring semantics
- tokenizer or dictionary assumptions

## Required implementation

### 1. Replace Python `rel_list` with tensor state

In both `L2RState` and `R2LState`, replace:

```python
rel_list: List[int]
```

with:

```python
rel_cpu: torch.LongTensor
```

Initialize as:

```python
rel_cpu=torch.zeros((max_len, max_len), dtype=torch.long)
```

Do this in `init_state()` for both builders.

### 2. Update relation entries in place

Replace every write like:

```python
state.rel_list[row_offset + j] = rid
```

with:

```python
state.rel_cpu[pos, j] = rid
```

For L2R, keep symmetric fill if current builder intends full pair matrix:

```python
state.rel_cpu[pos, j] = self._pair_to_rel_id(pi, pj)
state.rel_cpu[j, pos] = self._pair_to_rel_id(pj, pi)
```

For R2L, only fill causal lower triangle:

```python
state.rel_cpu[i, j] = self._pair_to_rel_id(pi, pj)
```

Do not backfill old R2L rows when a later operator resolves `UNK`. Old rows must preserve causal prefix semantics.

### 3. Make materialization zero-copy except device transfer

Replace:

```python
rel = torch.tensor(state.rel_list, dtype=torch.long).view(state.max_len, state.max_len)
rel_sliced = rel[:cur_len, :cur_len]
return rel_sliced.to(device, non_blocking=True)
```

with:

```python
return state.rel_cpu[:cur_len, :cur_len].to(device, non_blocking=True)
```

Do not call `torch.tensor(...)` in beam loop.

### 4. Clone tensor state with tensor clone

In `clone_state()`, replace Python list copy of relation ids with:

```python
rel_cpu=state.rel_cpu.clone()
```

For R2L, keep the existing deep clone logic for `CtxNode`, `Frame`, `Operand`, and `path_refs`. Do not share mutable nodes across beams.

### 5. Optimize materialize batch path

In `_beam_search()`, materialization currently loops over states and stacks tensors. Keep this first because it is simple and safe. After tensor state is implemented, optional micro-optimization:

```python
rel_ids = torch.stack(
    [builder.materialize_state(state, cur_len, self.device) for state in relation_states],
    dim=0,
)
```

Only optimize further if profiler proves stack or CPU to GPU copy dominates.

### 6. Fix `_rate()` fallback

In `utils/generation_utils.py`, `_rate()` must explicitly precompute and pass `rel_ids` when tree bias is enabled.

Target behavior:

```python
rel_ids = None
if getattr(self, "use_tree_bias", False):
    rel_ids = self._build_rel_ids_for_tgt(tgt)
out_hat = self.transform(src, src_mask, tgt, rel_ids=rel_ids) / temperature
```

If `_build_rel_ids_for_tgt` is only defined on `Decoder`, guard with `hasattr(self, "_build_rel_ids_for_tgt")`.

### 7. Keep cache optional

`beam_search(..., use_cache=True)` must still support `use_cache=False`.

- `use_cache=True`: use beam relation states.
- `use_cache=False`: old full-prefix relation-id fallback.

This is required for equivalence tests.

## Correctness constraints

### R2L causal rule

For R2L, row `i` may use only prefix tokens `tokens[0:i+1]`.

Example:

```text
L2R: x ^ { 2 }
R2L: } 2 { ^ x

Expected rows:
R
R U
R U U
R S S R
R S S R R
```

Do not use future tokens to resolve current row.

### Braces rule

Only exact `{` and `}` are layout delimiters.

Visible tokens are normal content:

- `\{`
- `\}`
- optionally `\lbrace`
- optionally `\rbrace`

Do not treat visible braces as layout grouping unless the dictionary explicitly maps them to raw `{` or `}`.

### Beam clone rule

When a parent beam is duplicated into multiple children, every child must get an independent deep clone of the relation state.

Never do this for mutable state:

```python
new_state = old_state
```

Never share `CtxNode` objects across child beams.

## Required tests

### A. State vs full builder equivalence

For every prefix of each sequence, compare cached state materialization to full builder output on that prefix.

Cases:

```text
x ^ { 2 }
x _ { i }
x ^ { a _ { i } }
\frac { a } { b }
\frac { x ^ { 2 } } { y _ { i } }
\{ x \}
```

Check both L2R and R2L where applicable.

Pseudo:

```python
state = builder.init_state(max_len, start_token)
for tok in tokens[1:]:
    builder.append_state(state, tok)
    prefix = torch.tensor([state.tokens], dtype=torch.long)
    full = builder.build(prefix)[0]
    cached = builder.materialize_state(state, len(state.tokens), torch.device("cpu"))
    assert torch.equal(cached, full)
```

### B. Beam clone independence

Create one R2L parent state, clone it twice, append different next tokens, and assert:

- child paths differ when expected
- child relation matrices differ when expected
- parent is unchanged
- no `CtxNode` object is shared between child states

### C. Decoder equivalence

With dropout off:

```python
logits_fallback = decoder(src, mask, input_ids, rel_ids=None)
logits_cached = decoder(src, mask, input_ids, rel_ids=cached_rel_ids)
assert torch.allclose(logits_fallback, logits_cached, atol=1e-6)
```

### D. Beam equivalence

With fixed seed and dropout off:

```python
old_hyps = model.beam_search(..., use_cache=False)
new_hyps = model.beam_search(..., use_cache=True)
assert old_hyps == new_hyps
```

### E. Benchmark

Add a small benchmark that reports:

- beam search time with `use_cache=False`
- beam search time with current optimized `use_cache=True`
- time spent in `materialize_state`
- time spent in `clone_state`
- time spent in decoder forward

Minimum synthetic setting:

```text
B=2 or 4
beam=10
max_len=100 or 200
bidirectional=true
use_tree_bias=true
```

## Acceptance criteria

The patch is done only when all conditions hold:

- No `torch.tensor(state.rel_list)` remains in beam-time `materialize_state()`.
- No Python `rel_list.copy()` remains in beam-time `clone_state()`.
- `_rate()` passes explicit `rel_ids` when tree bias is enabled.
- `beam_search(use_cache=True)` equals `beam_search(use_cache=False)` under deterministic settings.
- Cached state equals full builder for every prefix in required test cases.
- R2L escaped braces remain visible tokens, not layout delimiters.
- Benchmark shows reduced beam-time CPU overhead after replacing list state with tensor state.
- Training forward and `collate_fn` semantics are unchanged.

## Notes for performance

Do not implement Transformer KV-cache in this patch. It is a separate, harder optimization. R2L tree bias can resolve old token paths after later operators become visible, so KV-cache may change hidden-state semantics if implemented naively.

This patch should only optimize relation-id state storage, clone, materialization, and `_rate()` fallback.
