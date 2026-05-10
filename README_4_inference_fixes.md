# Strict Agent Guide — Finish the 4 Remaining CoMER Inference Bottlenecks

This README is a **narrow implementation contract** for the remaining inference/beam-search work in the CoMER PyTorch Lightning repo.

The repo already trains. Do **not** make broad unrelated refactors. Your job is to implement these 4 remaining items correctly and prove them with tests:

1. Implement a real KV-cache inference path.
2. Stop full-prefix decoder recomputation on every generated token.
3. Stop copying/gathering full encoder features on every beam step.
4. Remove CUDA scalar `.item()` use from boundary stripping after decode.

Do not claim completion unless every acceptance check in this file passes.

---

## 0. Hard scope rules

### You must not change

- Training objective.
- Validation policy.
- Scheduler monitor.
- AMP/BF16/mixed precision.
- deterministic/cuDNN benchmark settings.
- PyTorch/Lightning/CUDA versions.
- Vocabulary IDs or special-token semantics.
- Output file format.
- Image geometry semantics.

### Image task constraint

The task is image-to-sequence. Do not crop, distort, or add new resize logic. This file is about inference/beam-search performance, not image preprocessing.

### Files likely involved

Inspect these first:

```text
utils/generation_utils.py
models/decoder.py
models/comer.py
models/transformer/decoder_layer.py
models/transformer/attention.py
utils/utils.py
test_beam.py
configs/crohme_config.yaml
```

If this repo has renamed or moved these files, find the equivalent decoder, attention, generation, and beam-search implementations.

---

## 1. Baseline checks before editing

Run from repo root:

```bash
python -m compileall .
python test_beam.py || true

grep -R "def _beam_search\|def decode_step\|def transform" -n utils models . --exclude-dir=.git
grep -R "self.transform(src, src_mask, input_ids).*\\[:, -1" -n utils models . --exclude-dir=.git
grep -R "beam_src = \\[s\\[flat_indices\\]" -n utils models . --exclude-dir=.git
grep -R "\\.item()" -n utils/generation_utils.py utils/beam_search.py models/decoder.py models/transformer . --exclude-dir=.git
grep -R "\\.tolist()" -n utils/generation_utils.py utils/beam_search.py models/decoder.py models/transformer . --exclude-dir=.git
```

Write a short note before patching:

- Which generation path is active?
- Does `_beam_search()` call full-prefix `transform(...)` every step?
- Does `_beam_search()` reorder full encoder features every step?
- Where are boundary tokens stripped?
- Does a `decode_step()` exist, and is it a real cache implementation or a fallback?

---

# Phase A — Implement real KV-cache, not a fallback

## A1. Goal

Implement real incremental decoding for inference so generation no longer recomputes the entire decoded prefix each step.

Current bad pattern:

```python
next_token_logits = self.transform(src, src_mask, input_ids)[:, -1, :]
```

This recomputes all previous prefix tokens at every generated token. A real cache stores self-attention K/V for previous tokens and appends only the current token's K/V at each step.

## A2. Required design

Add an inference-only API, preferably in `models/decoder.py` and used by `utils/generation_utils.py`:

```python
def init_decode_cache(self, batch_beam_size: int, device, dtype=None):
    ...

def reorder_decode_cache(self, cache, new_order: torch.LongTensor):
    ...

def decode_step(
    self,
    src,
    src_mask,
    last_tokens: torch.LongTensor,
    cache,
    step: int,
    memory_batch_idx=None,
):
    # src: original encoder outputs, not repeatedly beam-copied every step.
    # src_mask: original encoder masks.
    # last_tokens: [B*beam] or [B*beam, 1], only the latest token.
    # cache: per-layer self-attention KV cache.
    # step: integer decode position.
    # memory_batch_idx: maps each active beam row to the original encoder batch row.
    #
    # Returns:
    #   logits: [B*beam, vocab_size] for the next token
    #   updated_cache
    ...
```

## A3. Cache structure

Use a clear cache structure. Example:

```python
cache = {
    "layers": [
        {
            "self_k": None,  # [B*beam, num_heads, T_cached, head_dim] or repo equivalent
            "self_v": None,
        }
        for _ in range(num_layers)
    ],
    "step": 0,
}
```

If the repo's attention stores K/V in another shape, document the shape in comments.

## A4. Decoder layer requirements

Each decoder layer must support an inference step path:

```python
def forward_step(self, x_t, encoder_out, encoder_mask, cache_layer, step, ...):
    ...
    return x_t_out, updated_cache_layer
```

The self-attention part must:

1. Project K/V only for the current token.
2. Concatenate or write the new K/V into the cache.
3. Attend current query against cached keys/values.
4. Return updated cache.

Training full-sequence forward must remain unchanged.

## A5. Positional encoding requirement

Full-prefix decoding usually applies positional encoding to all tokens. Incremental decoding must apply the correct position for the current `step`.

Do not always use position 0 for `last_tokens`.

Required:

```python
x_t = token_embedding(last_tokens)
x_t = add_position_for_step(x_t, step)
```

If the repo's positional encoding is sinusoidal/table-based, slice the row for `step`. If it is learned, index the learned table at `step`.

## A6. Beam cache reorder

After beam `topk`, selected beams change order. You must reorder the KV cache with the same `flat_indices` / selected beam indices used to reorder sequences and scores.

Required helper:

```python
cache = decoder.reorder_decode_cache(cache, selected_beam_rows)
```

This must reorder every cached layer's K and V along batch/beam dimension.

Do not reorder full encoder features here. Only reorder:

- token sequences,
- beam scores,
- done masks,
- small integer mapping tensors,
- cache tensors.

## A7. Acceptance checks for KV-cache

You must add tests that verify actual behavior.

### Test 1 — decode_step matches full-prefix logits

In eval mode, for a tiny model or a controlled dummy model:

```python
full_logits = model.transform(src, src_mask, prefix)[:, -1, :]
step_logits, cache = model.decode_step(...)

assert torch.allclose(step_logits, full_logits, atol=..., rtol=...)
```

Run for several prefix lengths, at least length 1, 2, and 3.

### Test 2 — cache length grows by one each step

After each `decode_step`, assert cached K/V length equals `step + 1`.

### Test 3 — cache reorder works

Create fake cache tensors with identifiable row values. Reorder using known indices. Assert all cached layers are reordered exactly.

### Completion rule

Do **not** write "KV-cache implemented" unless:

- `_beam_search()` actually calls `decode_step()` in the generation loop,
- `decode_step()` uses cache tensors,
- cache length grows,
- cache is reordered after beam selection,
- no full-prefix `transform(... input_ids)` call remains in the active beam loop.

If you only add the API but still use full-prefix internally, report it as **not complete**.

---

# Phase B — Stop full-prefix decode in active `_beam_search()`

## B1. Required change

In `utils/generation_utils.py`, the per-token loop must use only the latest token plus KV-cache:

Bad:

```python
for step in range(max_len):
    logits = self.transform(src, src_mask, input_ids)[:, -1, :]
```

Good:

```python
cache = self.decoder.init_decode_cache(batch_size * beam_size, device=input_ids.device)
last_tokens = input_ids[:, -1]

for step in range(max_len):
    logits, cache = self.decoder.decode_step(
        src=src,
        src_mask=src_mask,
        last_tokens=last_tokens,
        cache=cache,
        step=step,
        memory_batch_idx=beam_origin,
    )
    ...
    cache = self.decoder.reorder_decode_cache(cache, selected_beam_rows)
    last_tokens = next_tokens
```

Use the correct object names for this repo. If `DecodeModel` owns the decoder method, wire through that object cleanly.

## B2. Keep full-prefix path only for tests/debug

It is okay to keep full-prefix `transform()` for:

- training,
- teacher-forcing loss,
- `_rate()` candidate scoring,
- test comparison against `decode_step`,
- explicit debug fallback behind a config flag.

It is **not** okay for active `_beam_search()` to use full-prefix decoding after this phase.

## B3. Grep acceptance

After implementation:

```bash
grep -R "self.transform(src, src_mask, input_ids).*\\[:, -1" -n utils/generation_utils.py models . --exclude-dir=.git
grep -R "transform(src, src_mask, input_ids).*\\[:, -1" -n utils/generation_utils.py models . --exclude-dir=.git
```

Any remaining match must be outside active `_beam_search()` or behind a clearly named debug/test fallback.

## B4. Runtime acceptance

Update `test_beam.py` so it fails if active `_beam_search()` calls full-prefix `transform()` each step. One way:

- Make dummy `transform()` raise if called inside `_beam_search()`.
- Implement dummy `decode_step()` that returns scripted logits.
- Assert `_beam_search()` succeeds without `transform()`.

For the real model, add a separate comparison test where `decode_step()` matches `transform()` on tiny prefixes.

---

# Phase C — Stop copying/gathering full encoder features each beam step

## C1. Problem

This pattern is still not acceptable in the active per-token beam loop:

```python
beam_src = [s[flat_indices] for s in beam_src]
beam_src_mask = [sm[flat_indices] for sm in beam_src_mask]
```

It reorders/copies full encoder features every generated token. That defeats the purpose of reducing beam memory movement.

## C2. Required design

Keep original encoder outputs unchanged:

```python
src: List[Tensor]       # original [B, ...]
src_mask: List[Tensor]  # original [B, ...]
```

Maintain a small integer mapping:

```python
beam_origin: LongTensor[B*beam]
```

`beam_origin[row]` tells which original input sample owns this active beam row.

At initialization:

```python
beam_origin = torch.arange(batch_size, device=device).repeat_interleave(beam_size)
```

After beam selection:

```python
beam_origin = beam_origin[selected_beam_rows]
```

Do not reorder full `src` tensors.

## C3. How decoder should use `beam_origin`

Preferred approach:

- Pass `memory_batch_idx=beam_origin` into `decode_step()`.
- Inside cross-attention, gather/select encoder memory for the active beam rows only where needed.
- If cross-attention currently requires `[B*beam, ...]` memory, create an aligned view/copy **inside a small helper** and document that it is unavoidable for now.
- If copying is unavoidable, make sure it does not happen more than once per step and do not also keep reordering `beam_src`.

Better approach:

- Refactor cross-attention so query is `[B*beam, 1, D]` and memory is referenced by `beam_origin` without physically rebuilding all source feature maps each step.
- This can be done by gathering only K/V projections or by batching attention per original sample if necessary.

## C4. Forbidden active-loop patterns

These must not occur inside the token loop:

```python
beam_src = [s[flat_indices] for s in beam_src]
beam_src_mask = [sm[flat_indices] for sm in beam_src_mask]
src = [s[flat_indices] for s in src]
src_mask = [sm[flat_indices] for sm in src_mask]
```

Also avoid repeated:

```python
s.unsqueeze(1).expand(...).reshape(...)
```

inside per-step decode if it materializes full encoder memory.

## C5. Acceptance tests

Add a test or debug assertion that instruments source alignment calls.

Example:

```python
model.source_alignment_call_count = 0
...
assert model.source_alignment_call_count <= max_len_allowed_threshold
```

Better:

- assert no source alignment helper is called inside each token step,
- or prove only integer `beam_origin` is reordered each step.

Static check:

```bash
grep -R "beam_src = \\[s\\[flat_indices\\]" -n utils models . --exclude-dir=.git
grep -R "beam_src_mask = \\[sm\\[flat_indices\\]" -n utils models . --exclude-dir=.git
```

Expected: no active-path matches.

## C6. Important correctness requirement

When beams reorder, each beam must still attend to the correct original encoder output. Add a deterministic test with two different input samples and scripted logits to ensure beams from sample 0 never attend to sample 1's encoder memory after reorder.

---

# Phase D — Boundary stripping without CUDA `.item()`

## D1. Problem

Boundary stripping currently may do this on CUDA tensors:

```python
if seq[0].item() in boundary_ids:
    ...
if seq[end - 1].item() in boundary_ids:
    ...
```

Even though this happens after decode rather than inside the token loop, it still creates scattered CPU-GPU syncs over hypotheses.

## D2. Required change

Before boundary stripping, move final sequences and scores to CPU once:

```python
final_sequences_cpu = final_sequences.detach().cpu()
final_scores_cpu = final_scores.detach().cpu()
```

Then strip boundaries on CPU tensors or Python lists.

Boundary helper should either:

1. Require CPU input and assert it:

```python
def _strip_generated_boundaries_cpu(seq: torch.Tensor, direction: str) -> torch.Tensor:
    assert seq.device.type == "cpu"
    ...
```

or

2. Use tensor operations without `.item()` on CUDA, then convert once.

Preferred: CPU-only helper after one `.cpu()` transfer.

## D3. Boundary semantics

Must preserve original output semantics:

- Remove generated start boundary if present at sequence start.
- Remove generated terminal boundary only at sequence end.
- For l2r:
  - input raw might be `[SOS, token1, token2, EOS, PAD]`;
  - output must be `[token1, token2]`.
- For r2l:
  - ensure final comparable hypothesis is in the same orientation as ground-truth labels;
  - remove generated boundary tokens before returning hypothesis.
- Do not remove internal token IDs if they appear in the middle.

## D4. Forbidden patterns in boundary code

After this phase, active boundary stripping code must not contain:

```python
seq[...].item()
cuda_tensor.item()
```

Static check:

```bash
grep -R "_strip_generated_boundaries\\|\\.item()" -n utils/generation_utils.py
```

Any `.item()` in this file must be outside active generation/boundary code and justified.

## D5. Acceptance tests

Add tests:

1. l2r CPU boundary strip.
2. r2l CPU boundary strip.
3. no-terminal max-length case.
4. immediate-terminal empty output case.
5. GPU generation path returns outputs, then CPU boundary stripping occurs once.

If CUDA is available, add a test that wraps boundary helper with a CUDA input and expects either:

- it first converts the entire batch to CPU once, or
- it raises a clear error saying CPU input required.

---

# Phase E — Final verification

Run all:

```bash
python -m compileall .
python test_beam.py
```

Static checks:

```bash
grep -R "self.transform(src, src_mask, input_ids).*\\[:, -1" -n utils/generation_utils.py models . --exclude-dir=.git
grep -R "transform(src, src_mask, input_ids).*\\[:, -1" -n utils/generation_utils.py models . --exclude-dir=.git
grep -R "beam_src = \\[s\\[flat_indices\\]" -n utils models . --exclude-dir=.git
grep -R "beam_src_mask = \\[sm\\[flat_indices\\]" -n utils models . --exclude-dir=.git
grep -R "_strip_generated_boundaries\\|\\.item()" -n utils/generation_utils.py
grep -R "\\.tolist()" -n utils/generation_utils.py
```

Expected:

- No active `_beam_search()` full-prefix decode.
- No active per-step full encoder feature reorder/copy.
- No `.item()` in boundary stripping on CUDA tensors.
- `.tolist()` only after `.detach().cpu()` and outside hot loops.
- `test_beam.py` calls actual active `_beam_search()` and passes.

If dataset/checkpoint is available, run one-batch generation:

```bash
python - <<'PY'
from omegaconf import OmegaConf
from datamodule import CROHMEDatamodule
from lit_comer import LitCoMER
import torch

config = OmegaConf.load("configs/crohme_config.yaml")
dm = CROHMEDatamodule(config)
dm.setup("fit")
batch = next(iter(dm.val_dataloader()))

model = LitCoMER(config, vocab_info=dm.vocab.get_info())
model.eval()
with torch.inference_mode():
    hyps = model.approximate_joint_search(batch.imgs, batch.mask)
print("num_hyps", len(hyps))
print("first_seq", hyps[0].seq[:20] if hyps else None)
PY
```

If CUDA is available, run a memory/time smoke test:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
# run one generation call
torch.cuda.synchronize()
print(torch.cuda.max_memory_allocated())
print(torch.cuda.max_memory_reserved())
```

Report whether peak memory decreased compared with previous full-prefix/copy-per-step implementation if baseline numbers are available.

---

# Short prompt for Cursor / Agent

```text
Read `README.md` fully and implement only the 4 remaining CoMER inference fixes in this file.

The repo already trains. Do not do broad refactors and do not change training policy, validation policy, scheduler monitor, AMP/BF16, deterministic settings, framework versions, image geometry semantics, vocabulary IDs, metrics, or output format.

Required:
1. Implement a real KV-cache inference path. `decode_step()` must use cached self-attention K/V, cache length must grow each step, and cache must reorder after beam topk. Do not claim KV-cache if it still falls back to full-prefix `transform()`.
2. Remove full-prefix decoding from active `_beam_search()`. The beam loop must call `decode_step()` with only the latest token.
3. Stop copying/gathering full encoder features each beam step. Keep original encoder outputs stable and reorder only small integer beam-origin mapping plus token/cache state. No `beam_src = [s[flat_indices] ...]` in the active loop.
4. Remove CUDA scalar `.item()` from boundary stripping. Move final sequences/scores to CPU once, then strip SOS/EOS boundary tokens on CPU. Preserve l2r/r2l output semantics.

Update or write tests. `python test_beam.py` must call the actual active `_beam_search()` and fail if full-prefix decode, per-step encoder copy, or boundary-token leakage returns. Run all README acceptance checks and report files changed, commands run, what passed, and any incomplete items with exact reasons.
```

---

# Completion checklist

The task is complete only if all are true:

- Active `_beam_search()` does not call full-prefix `transform(..., input_ids)[:, -1]`.
- Active `_beam_search()` calls real cache-backed `decode_step()`.
- Cache contains per-layer self-attention K/V and grows by one token per step.
- Cache is reordered after beam top-k selection.
- Full encoder features are not copied/gathered every beam step.
- Only small integer beam-origin/index tensors are reordered every step.
- Boundary stripping does not call `.item()` on CUDA tensors.
- Boundary stripping removes only structural leading/trailing SOS/EOS and preserves normal tokens.
- `python test_beam.py` calls actual `_beam_search()` and passes.
- Static grep checks in Phase E pass or remaining matches are explicitly documented as non-active debug/test code.
