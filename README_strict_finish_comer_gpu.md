# Strict Completion Guide — Finish CoMER GPU Optimization Against Original README

This guide is for a coding agent working on the current CoMER repository after the repo already trains successfully. The goal is not merely "make it run"; the goal is to finish the remaining incomplete items from the original GPU-utilization README as tightly as possible while preserving task semantics.

Use this file as the implementation contract. Do not mark an item as done unless its acceptance checks pass.

---

## 0. Scope, guardrails, and non-negotiables

### Main goal

Finish the remaining performance/correctness work that is still incomplete relative to the original README:

1. Correct active beam-search output semantics.
2. Test the actual beam-search implementation, not a duplicated toy function.
3. Avoid hidden CPU-GPU syncs in generation hot loops.
4. Reduce repeated encoder-feature materialization during beam decoding.
5. Add a real incremental/KV-cache inference path, or if too large for one pass, implement the interface and a verified first step toward it.
6. Vectorize target construction more tightly.
7. Make `BucketedBatchSampler` DDP-safe and epoch-seeded.
8. Clean up data/cache correctness issues.
9. Tighten mask caching by device/dtype.
10. Add verification scripts that prove the actual code paths are exercised.

### Explicitly out of scope

Do not implement these unless the user explicitly asks later:

- AMP / BF16 / mixed precision.
- Deterministic or cuDNN benchmark policy changes.
- PyTorch / Lightning / CUDA environment upgrades.
- Changing validation policy, such as removing validation beam-search ExpRate.
- Changing scheduler monitor behavior or scheduler metric selection.
- Changing model objective, vocabulary semantics, metric semantics, or output artifact format.

### Image-task constraint

This is an image-to-sequence problem. Preserve image H/W semantics.

Allowed:
- Padding upward.
- Bucket/static padding.
- Padding to multiples such as 16 or 32.
- Keeping masks correct after padding.

Forbidden:
- Cropping content.
- Stretching/distorting images.
- Adding new rescale behavior beyond the existing transform policy.
- Replacing failed images with random noise silently.

### Agent workflow rules

Before editing each phase:

1. Inspect the referenced files.
2. Write a short patch plan.
3. Make the smallest safe diff.
4. Run the phase-specific checks.
5. Report files changed and verification results.

Never claim completion based only on `compileall`. A "done" item must run the actual relevant code path or have an explicit reason why runtime verification was impossible.

---

## 1. Current known status

The repo now trains, but these items remain incomplete or need strict verification:

| Area | Current status | Required completion |
|---|---|---|
| Beam output semantics | likely still risky | Strip generated terminal tokens correctly |
| Beam tests | toy test passes | Test actual `DecodeModel._beam_search()` |
| Beam feature memory | still likely materializes source per step | Avoid repeated per-step encoder feature expansion/copy |
| KV-cache | not implemented | Add incremental decode path or staged verified interface |
| Target construction | moved out of `training_step`, but still Python-loop-heavy in collate | Vectorize from padded label tensor |
| DDP sampler | bucket sampler exists but likely not epoch/rank deterministic enough | Add `set_epoch`, local RNG, DDP-safe slicing |
| Image load failure | random-noise fallback may exist | Fail loudly or skip deterministically, never train on random image |
| YAML numeric config | `128e4` may exist | Use integer `1280000` |
| Causal mask cache | single cache may exist | Cache by `(device, dtype, length)` or safe key |

---

## 2. Required baseline checks

Run these first from the repository root:

```bash
python -m compileall .
python test_beam.py || true

grep -R "empty_cache" -n . --exclude-dir=.git
grep -R "max_pixels_per_batch\|gpu_max_memory" -n configs datamodule . --exclude-dir=.git
grep -R "to_bi_tgt_out(" -n lit_comer.py datamodule utils . --exclude-dir=.git
grep -R "\.item()" -n utils models lit_comer.py --exclude-dir=.git
grep -R "\.tolist()" -n utils models lit_comer.py --exclude-dir=.git
grep -R "done_mask.all()" -n . --exclude-dir=.git
grep -R "shared_vocab\|CROHMEDatamodule.shared_vocab" -n . --exclude-dir=.git
grep -R "torch.equal" -n models . --exclude-dir=.git
```

Inspect:

- `utils/generation_utils.py`
- `models/decoder.py`
- `models/comer.py`
- `datamodule/utils.py`
- `datamodule/datamodule.py`
- `datamodule/dataset.py`
- `utils/utils.py`
- `test_beam.py`
- `configs/crohme_config.yaml`

Create a short note before patching that says which grep results are expected legacy-only and which are still active-path problems.

---

# Phase A — Beam correctness and actual beam tests

## A1. Fix generated terminal-token stripping

### Problem

The active `_beam_search()` likely returns sequences after removing the initial start token but may still include the generated terminal token:

- l2r generated terminal: `<eos>`
- r2l generated terminal: `<sos>`

If terminal tokens remain inside `Hypothesis.seq`, then:

- ExpRate may be wrong.
- `_rate()` may add terminal tokens again.
- Candidate scoring and output formatting may differ from original semantics.

### Required change

In `utils/generation_utils.py`, add a helper:

```python
def _strip_generated_boundaries(seq: torch.Tensor, direction: str) -> torch.Tensor:
    ...
```

Rules:

- Input should be a 1D tensor containing generated non-pad ids after decoding.
- Remove the leading start token if still present.
- Remove the generated terminal token if present:
  - for `direction == "l2r"`, remove trailing `eos_id`;
  - for `direction == "r2l"`, remove trailing `sos_id` before or after reverse handling, depending on current implementation.
- Do not remove normal vocabulary tokens.
- Do not remove internal `sos/eos` if they appear as data errors; only strip boundary positions.
- Return tokens in the same direction/format that the evaluator expects: comparable to `batch.indices` truth labels, without boundary tokens.

### Acceptance tests

Add or update tests so these cases pass:

1. l2r: `[SOS, a, b, EOS, PAD] -> [a, b]`
2. l2r no terminal due to max length: `[SOS, a, b] -> [a, b]`
3. r2l generated raw path returns final comparable sequence without boundary `SOS/EOS`.
4. Empty or immediate terminal output does not crash.
5. Hypothesis sequences passed to `ExpRateRecorder.update()` do not contain terminal boundary tokens.

Do not mark this done without a test that checks terminal-token stripping explicitly.

---

## A2. Replace toy `test_beam.py` with actual active-path tests

### Problem

A test that implements its own beam-search function does not validate the repository's actual `_beam_search()` code.

### Required change

Rewrite `test_beam.py` or add `tests/test_beam_actual.py` so it directly imports and calls the actual generation path.

Minimum acceptable design:

1. Create a tiny dummy subclass/object that uses the actual `DecodeModel._beam_search()` implementation.
2. Override `transform()` to return deterministic logits.
3. Use small fake encoder features and masks.
4. Run on CPU by default.
5. Run on CUDA if available, but CUDA should be optional.
6. Test batch size > 1 and beam size > 1.
7. Test both l2r and r2l if both are supported.
8. Verify output shape/type and boundary-token semantics.

Pseudo-structure:

```python
class DummyDecodeModel(DecodeModel):
    def __init__(self, vocab_info, scripted_logits):
        ...
    def transform(self, src, src_mask, input_ids):
        # Return [B_or_Bbeam, T, V] logits deterministically.
        ...
```

### Acceptance checks

Run:

```bash
python test_beam.py
```

Expected:

- Calls actual `DecodeModel._beam_search()`.
- Fails if `_beam_search()` shape logic is broken.
- Fails if terminal tokens are not stripped.
- Does not require dataset or checkpoint.

---

# Phase B — Remove hidden sync and repeated source copy in generation

## B1. No CUDA bool in generation hot loop

### Problem

Generation must not use CUDA tensor bools in Python control flow inside the per-token loop.

Forbidden in hot loop:

```python
if done_mask.all():
    break
if some_cuda_tensor:
    ...
while cuda_tensor:
    ...
```

### Required change

Inspect `utils/generation_utils.py`. Keep done state as GPU tensors and use masks. The simplest safe version is:

```python
for step in range(max_len):
    ...
```

Early stopping is allowed only if:

- it is intentionally CPU-based,
- it is outside per-beam inner loops,
- it is documented,
- and it is not performed every beam/token.

### Acceptance checks

```bash
grep -R "done_mask.all()" -n utils/generation_utils.py
grep -R "if .*\.all()" -n utils/generation_utils.py
```

Any match must be outside CUDA hot path and explicitly justified.

---

## B2. Avoid repeated encoder-feature materialization during each decode step

### Problem

Patterns like this inside `Decoder.transform()` can materialize encoder features per decode step:

```python
s = s.unsqueeze(1).expand(...).reshape(...)
sm = sm.unsqueeze(1).expand(...).reshape(...)
```

Even if `expand()` is a view, `reshape()` after expanding a stride-0 dimension may allocate/copy. If this occurs inside `transform()`, and `_beam_search()` calls `transform()` every token step, VRAM and runtime remain poor.

### Required target

Choose one safe design.

#### Option 1 — Beam-align encoder features once before the loop

Before entering the token loop:

```python
beam_src = align_src_once(src, beam_size)
beam_src_mask = align_mask_once(src_mask, beam_size)
```

Then `transform()` must not re-expand source on every call.

This may still copy once, but it must not copy every step.

#### Option 2 — Index-based source mapping

Keep original source at `[B, ...]` and pass `beam_to_batch_idx` into the decoder. Gather only where necessary. This is preferred but may require more refactor.

#### Option 3 — Documented fallback

If a full refactor is too risky, isolate the copy in one helper and add a TODO. But you must prove the copy is not repeated inside the token loop.

### Required checks

```bash
grep -R "expand.*beam\|repeat_interleave.*beam\|contiguous().view\|reshape" -n models/decoder.py utils/generation_utils.py
grep -R "torch.cat((src\[i\], src\[i\])" -n utils/generation_utils.py models . --exclude-dir=.git
```

Add a code comment near the final design explaining:

- whether encoder features are copied,
- how many times,
- why it is safe/necessary.

### Acceptance checks

- No per-token `transform()` call re-expands original `[B, ...]` source to `[B*beam, ...]`.
- At worst, encoder features are aligned once per beam search, not once per token.
- Behavior remains equivalent on the actual beam test.

---

# Phase C — Incremental decoding / KV-cache path

## C1. Add a real staged KV-cache implementation plan inside code

### Problem

The current inference path likely calls full-prefix decoding each step:

```python
self.transform(src, src_mask, input_ids)[:, -1, :]
```

This recomputes all previous prefix tokens every step. This is the largest remaining reason inference/validation may not maximize GPU.

### Required change

Implement a staged solution. If full KV-cache is too large for one patch, do not fake it. Add the interface and implement at least a verified first step.

Minimum Stage 1:

1. Add an inference-only method:

```python
def decode_step(self, src, src_mask, last_tokens, cache, step, beam_state=None):
    ...
```

2. Make it produce the same next-token logits as the full-prefix path on a tiny deterministic input.
3. Add `cache` as a structured object/dict, even if Stage 1 internally falls back to full-prefix.
4. Add TODO comments showing exactly where layer K/V will be cached.
5. Ensure training forward path remains unchanged.

Preferred Stage 2:

- Cache per-layer self-attention K/V.
- Reorder cache after beam selection.
- Reuse encoder cross-attention projections where feasible.
- Generation calls `decode_step()` instead of `transform(..., input_ids)` for every step.

### Acceptance tests

Add a small test:

```python
# For a short prefix, compare:
full_logits = model.transform(src, src_mask, prefix)[:, -1, :]
step_logits = model.decode_step(...)[0]
assert close_enough(full_logits, step_logits)
```

Run in `eval()` mode with reasonable tolerance.

### Completion rule

Do not claim "KV-cache implemented" unless generation actually uses cached K/V and avoids full-prefix decoder recomputation.

If only Stage 1 is implemented, report: "KV-cache interface added, generation still falls back to full-prefix decode."

---

# Phase D — Target construction and data correctness

## D1. Vectorize target construction from padded labels

### Problem

Target construction has moved out of `training_step`, but it may still use Python lists and per-sample loops in `collate_fn`/`to_bi_tgt_out()`.

### Required target design

Collate should create:

```python
labels: LongTensor[B, L]
lengths: LongTensor[B]
tgt: LongTensor[2B, L + 1]
out: LongTensor[2B, L + 1]
```

from padded tensors with minimal Python work.

### Implementation guide

1. Tokenize labels earlier if possible:
   - ideally in `extract_data()` or dataset record creation;
   - not repeatedly in collate every epoch if labels are static.
2. In collate:
   - use a single padded `labels` tensor;
   - use `lengths` to build masks;
   - use tensor slicing/indexing to fill l2r target/output.
3. For r2l:
   - create reversed labels using tensor operations with length masks;
   - avoid per-sample Python `reversed(list)` loops if feasible.
4. Keep a compatibility helper if needed, but `training_step` must not call it.

### Acceptance checks

```bash
grep -R "to_bi_tgt_out(" -n lit_comer.py
```

Expected: no match in `lit_comer.py`.

Add or update a test that compares old list-based target construction and new tensorized construction on several examples:

- different sequence lengths,
- length 1,
- max length,
- PAD positions,
- l2r/r2l semantics.

---

## D2. Remove random image fallback

### Problem

If image loading fails, the dataset must not silently replace the image with random noise. That corrupts training and hides data bugs.

### Required change

Find code like:

```python
np.random.randint(...)
```

inside image loading fallback and remove it.

Preferred behavior:

- Fail loudly with a clear error containing the image path, or
- skip invalid sample during dataset construction with a logged warning and deterministic count.

Do not silently train on random images.

### Acceptance checks

```bash
grep -R "random.randint\|np.random\|randn\|random image" -n datamodule dataset utils . --exclude-dir=.git
```

Any match in data loading must be justified and not used as silent fallback.

---

## D3. Make `BucketedBatchSampler` DDP-safe and epoch-seeded

### Problem

A bucket sampler using global `random.shuffle` can be nondeterministic across DDP ranks and epochs.

### Required change

Implement:

```python
def set_epoch(self, epoch: int):
    self.epoch = epoch
```

Use local RNG:

```python
rng = random.Random(self.seed + self.epoch)
rng.shuffle(...)
```

For DDP compatibility:

- detect `rank` and `world_size` if available,
- or accept them from constructor,
- slice batches deterministically:

```python
batches = batches[rank::world_size]
```

Do not rely on each rank having independent random state.

### Acceptance tests

Add a CPU-only test:

1. Create fake records.
2. Instantiate sampler for `rank=0, world_size=2` and `rank=1, world_size=2`.
3. Call `set_epoch(0)` on both.
4. Verify no overlapping batch indices between ranks for that epoch.
5. Call `set_epoch(1)` and verify order changes deterministically.

---

## D4. Use numeric integer for pixel budget

### Problem

YAML value `128e4` may be parsed inconsistently.

### Required change

In config, use:

```yaml
max_pixels_per_batch: 1280000
```

Keep backward compatibility:

```python
max_pixels = cfg.get("max_pixels_per_batch", cfg.get("gpu_max_memory", 1280000))
max_pixels = int(max_pixels)
```

### Acceptance checks

- Config loads and `max_pixels_per_batch` is an `int`.
- Old `gpu_max_memory` config still works as fallback.

---

# Phase E — Mask cache and model hot-path cleanup

## E1. Cache causal masks by device and dtype

### Problem

A single cached mask can break or reallocate if model moves between CPU/GPU or dtype changes.

### Required change

In `models/decoder.py`, use a dictionary cache:

```python
self._causal_mask_cache = {}

def _get_causal_mask(self, length, device, dtype):
    key = (device.type, device.index, str(dtype))
    ...
```

The cached tensor should be at least `length x length`. Return a slice.

### Acceptance checks

- CPU forward then CUDA forward does not reuse wrong-device mask.
- Multiple lengths reuse the same larger cached tensor.
- No repeated full mask allocation for same device/dtype.

---

## E2. Keep ARM vectorized norm behind config and test it

### Problem

`arm_norm_impl` may still default to `legacy`, which is acceptable for accuracy but not maximum performance.

### Required change

Do not force default change unless tested. But add a smoke test for:

```yaml
arm_norm_impl: "masked_vectorized"
```

Test must verify:

- forward output shape,
- finite output values,
- backward pass works,
- no obvious mask shape error.

### Acceptance

- The repo has a clear way to benchmark `legacy` vs `masked_vectorized`.
- The default remains accuracy-safe unless user changes it.

---

## E3. Measure training-time feature duplication

### Problem

`models/comer.py` may still duplicate encoder features for bidirectional training:

```python
feature = torch.cat((feature, feature), dim=0)
mask = torch.cat((mask, mask), dim=0)
```

The original README did not require risky semantic changes here, but it requested review/measurement.

### Required change

Add a small profiling utility or debug flag to measure memory before/after encoder and after bidirectional expansion.

Do not change this path unless equivalence is tested.

### Acceptance

- There is a way to measure how much memory this copy costs.
- No semantic-changing refactor is done without tests.

---

# Phase F — Final verification suite

## F1. Static checks

Run:

```bash
python -m compileall .

grep -R "empty_cache" -n . --exclude-dir=.git
grep -R "to_bi_tgt_out(" -n lit_comer.py --exclude-dir=.git
grep -R "done_mask.all()" -n utils/generation_utils.py --exclude-dir=.git
grep -R "torch.equal" -n models/transformer/attention.py --exclude-dir=.git
grep -R "assert len(batch) == 1" -n datamodule --exclude-dir=.git
grep -R "np.random\|random image" -n datamodule dataset utils . --exclude-dir=.git
grep -R "CROHMEDatamodule.shared_vocab" -n . --exclude-dir=.git
grep -R "\.item()" -n utils/generation_utils.py utils/beam_search.py models lit_comer.py --exclude-dir=.git
grep -R "\.tolist()" -n utils/generation_utils.py utils/beam_search.py models lit_comer.py --exclude-dir=.git
```

Allowed:

- `.item()` once in debug-only code.
- `.tolist()` only after `.detach().cpu()` and outside generation hot loops.
- Deprecated legacy `utils/beam_search.py` only if clearly not imported by active path.

Not allowed:

- `.item()` inside active generation per-token loop.
- `.tolist()` inside active generation per-token loop.
- CUDA tensor bool in Python `if`/`while`.

## F2. Runtime smoke tests

Run at minimum:

```bash
python test_beam.py
```

Then run a one-batch train smoke if dataset is available:

```bash
python - <<'PY'
from omegaconf import OmegaConf
from datamodule import CROHMEDatamodule
from lit_comer import LitCoMER

config = OmegaConf.load("configs/crohme_config.yaml")
dm = CROHMEDatamodule(config)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))

model = LitCoMER(config, vocab_info=dm.vocab.get_info())
out = model(batch.imgs, batch.mask, batch.tgt)
print("forward:", out.shape)
print("batch:", batch.imgs.shape, batch.mask.shape, batch.tgt.shape, batch.out.shape)
print("optim:", type(model.configure_optimizers()))
PY
```

Run a one-batch generation smoke if possible:

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
print("hyps:", len(hyps), hyps[0].seq[:10] if hyps else None)
PY
```

If checkpoint/data are unavailable, explain exactly which runtime checks could not run.

---

# Short prompt for coding agent

```text
Read `README.md` fully and implement ONLY the remaining strict completion items in this file.

The repo already trains, so do not do broad refactors. Your job is to finish what is still incomplete relative to the original GPU README:
1. Fix active beam-search output semantics by stripping generated terminal SOS/EOS tokens correctly.
2. Replace the current beam test with a test that calls the actual `DecodeModel._beam_search()`.
3. Ensure active generation hot loops have no `.item()`, `.tolist()`, or CUDA bool used in Python `if/while`.
4. Stop repeated encoder-feature expansion/copy inside per-token decode. At worst, align/copy source once before the loop and document it.
5. Add an incremental `decode_step` / KV-cache interface and implement real cache use if feasible; do not falsely claim KV-cache if generation still full-prefix decodes.
6. Vectorize target construction from padded labels more tightly and test l2r/r2l semantics.
7. Make `BucketedBatchSampler` DDP-safe with `set_epoch`, local RNG, and deterministic rank/world_size splitting.
8. Remove random-image fallback in data loading.
9. Change `max_pixels_per_batch: 128e4` to integer `1280000` and cast config values to int.
10. Cache causal masks by device/dtype.
11. Add smoke tests for actual generation, target construction, sampler DDP splitting, and optional ARM vectorized norm.

Do not implement AMP/BF16, deterministic/benchmark changes, framework upgrades, validation policy changes, or scheduler monitor changes.
Preserve image H/W semantics: do not crop, distort, or add new resize behavior; only pad/bucket.
After each phase, run the README acceptance checks and report files changed, commands run, completed items, and incomplete items with reasons.
```

---

# Final completion criteria

The agent may only claim this work is complete if all are true:

- `python -m compileall .` passes.
- `python test_beam.py` calls actual `_beam_search()` and passes.
- Actual beam outputs do not contain generated boundary tokens.
- `training_step` does not call `to_bi_tgt_out`.
- Target construction has tests proving l2r/r2l semantics.
- Active generation hot loop has no CUDA scalar sync.
- Encoder feature beam alignment is not repeated every token step.
- KV-cache is either truly implemented and used, or clearly reported as staged-only.
- Bucket sampler has deterministic `set_epoch` and rank split.
- No random image fallback remains.
- Pixel budget config is numeric integer.
- Causal mask cache is device/dtype safe.
- No excluded policy/framework changes were made.
