# README.md

## Objective

Restore the original behavior from `DSP391m_Group1_FALL25_CoMER_old` for padding, inference, and beam search in the main repo `DSP391m_Group1_FALL25_CoMER`, while preserving safe training refactor improvements where they do not conflict with those three restoration targets. The old repo is the behavioral source of truth for this task. 

## Repositories

* `DSP391m_Group1_FALL25_CoMER_old` is the behavioral reference. Do not edit it.
* `DSP391m_Group1_FALL25_CoMER` is the target repo to edit.
* Copy behavior from the old repo selectively. Do not revert the whole target repo unless a file is entirely inference/cache-specific and reverting it is clearly the safest minimal change.

## Scope of Changes

The coding agent must make exactly these three restoration changes:

1. Remove / disable KV-cache inference.
2. Restore original padding behavior.
3. Restore original inference and beam search behavior.

## Non-Goals

Do **not** do any of the following:

* Do not rewrite the whole repo.
* Do not change the model architecture.
* Do not change training loss semantics.
* Do not remove safe training cleanup improvements unless they directly conflict with the restoration targets.
* Do not keep the new beam search if it changes old semantics.
* Do not keep active ARM incremental cache behavior.
* Do not optimize further before restoring correctness.
* Do not modify `DSP391m_Group1_FALL25_CoMER_old`.
* Do not treat the new README files in the target repo as the source of truth; they describe the refactor path, not the requested restoration.

## Part 1: Remove / Disable KV-cache Inference

The target repo added an active incremental/KV-cache inference path. Disable it so inference uses the old full-prefix decoding path.

Search for these names and remove them if safe, or leave them unused if removal is risky:

* `decode_step`
* `init_decode_cache`
* `reorder_decode_cache`
* `forward_step`
* `project_kv`
* `beam_origin`
* `cross_kv`
* `self_k`
* `self_v`
* ARM incremental cumsum/cache code
* cache-aware beam search logic

Concrete instructions:

1. In `utils/generation_utils.py`, restore the active `_beam_search()` loop to call:

   ```python
   self.transform(src, src_mask, input_ids)[:, -1, :]
   ```

   Do **not** call `decode_step()` from `_beam_search()`.

2. Remove or disable the active cache initialization in `_beam_search()`:

   ```python
   cache = self.init_decode_cache(...)
   cache = self.reorder_decode_cache(...)
   next_token_logits, cache = self.decode_step(...)
   ```

3. In `models/decoder.py`, remove or leave unused the cache-specific methods:

   * `init_decode_cache`
   * `reorder_decode_cache`
   * `decode_step`

   The active inference path must not call them.

4. In `models/transformer/transformer_decoder.py`, remove or leave unused the incremental methods:

   * `TransformerDecoder.forward_step`
   * `TransformerDecoderLayer.forward_step`

5. In `models/transformer/attention.py`, remove or leave unused:

   * `MultiheadAttention.project_kv`
   * `MultiheadAttention.forward_step`

6. In `models/transformer/arm.py`, remove or leave unused:

   * `AttentionRefinementModule.forward_step`

7. Ensure ARM during inference is evaluated through the original full-prefix `forward()` path, not through incremental coverage state.

8. Prefer minimal changes that make the active inference path match the old repo. It is acceptable for unused helper methods to remain temporarily only if static grep and tests prove no active inference call reaches them.

## Part 2: Restore Original Padding Behavior

The old repo pads each batch only to the maximum image height and width in that batch. The target repo added config-driven bucket/static padding and defaults to `pad_strategy: "bucket"` with `pad_to_multiple: 32`. Disable that behavior.

Restore old-style `batch_max` behavior:

```python
max_height_x = max(heights_x)
max_width_x = max(widths_x)

x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)

for idx, s_x in enumerate(images_x):
    x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x
    x_mask[idx, :heights_x[idx], :widths_x[idx]] = 0
```

Concrete instructions:

1. In `datamodule/datamodule.py`, remove or bypass the branch that rounds padding up for:

   * `pad_strategy == "bucket"`
   * `pad_strategy == "static"`
   * `pad_to_multiple`

2. In `configs/crohme_config.yaml`, remove these fields or make them harmless:

   ```yaml
   pad_strategy: "bucket"
   pad_to_multiple: 32
   ```

   Preferred: remove them entirely, or set:

   ```yaml
   pad_strategy: "batch_max"
   ```

   and ensure `collate_fn()` ignores all non-`batch_max` strategies.

3. Padding must not be rounded up to a multiple unless the old repo did that. The old repo did **not** round H/W to multiples.

4. Image masks must match old behavior:

   * `True` means padded / masked region.
   * `False` means valid image region.
   * Mask shape must be `[batch, batch_max_height, batch_max_width]`.

5. The new batch sampler may be kept if it only changes how samples are grouped and does not force new padding semantics.

6. Keep precomputed `tgt`, `out`, `labels`, and `lengths` in `Batch` if they remain correct. This is a safe training refactor and does not conflict with old image padding behavior.

## Part 3: Restore Original Inference and Beam Search

Use the old repo’s `utils/generation_utils.py` and `utils/beam_search.py` as the behavioral reference.

Restore the old behavior for:

* `BeamSearchScorer`
* `BeamHypotheses`
* `topk(2 * beam_size)`
* finished hypothesis handling
* `early_stopping`
* `length_penalty` / `alpha`
* EOS/SOS terminal token handling
* finalization behavior
* approximate joint bidirectional search behavior
* test/inference output formatting

Concrete instructions:

1. In `utils/generation_utils.py`, restore old `DecodeModel.beam_search()` structure:

   * Duplicate encoder features once for bidirectional decoding:

     ```python
     src[i] = torch.cat((src[i], src[i]), dim=0)
     src_mask[i] = torch.cat((src_mask[i], src_mask[i]), dim=0)
     ```

   * Initialize L2R with SOS and R2L with EOS.

   * Construct a `BeamSearchScorer`.

   * Call `_beam_search()` using full-prefix decoding.

   * Reverse the R2L half after first-pass beam search.

   * Build reverse-direction `tgt/out`.

   * Call `_rate()`.

   * Combine forward and reverse scores.

   * Select best hypothesis across L2R/R2L candidates.

2. Adapt old code to the target repo’s safer `VocabInfo` design instead of reintroducing unsafe global vocab imports.

   Use:

   ```python
   self.vocab_info.sos_id
   self.vocab_info.eos_id
   self.vocab_info.pad_id
   self.vocab_info.vocab_size
   self.vocab_info.words
   ```

   Do **not** reintroduce:

   ```python
   from datamodule.datamodule import CROHMEDatamodule
   vocab = CROHMEDatamodule.shared_vocab
   ```

3. In `utils/generation_utils.py`, restore old `_beam_search()` semantics:

   ```python
   while cur_len < max_len and not beam_scorer.is_done():
       next_token_logits = (
           self.transform(src, src_mask, input_ids)[:, -1, :] / temperature
       )
       next_token_scores = F.log_softmax(next_token_logits, dim=-1)
       next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

       reshape_size = next_token_scores.shape[0] // batch_size
       next_token_scores = rearrange(
           next_token_scores,
           "(b m) v -> b (m v)",
           m=reshape_size,
       )

       next_token_scores, next_tokens = torch.topk(
           next_token_scores, 2 * beam_size, dim=1
       )

       next_indices = next_tokens // vocab_size
       next_tokens = next_tokens % vocab_size

       if cur_len == 1:
           input_ids = repeat(input_ids, "b l -> (b m) l", m=beam_size)
           for i in range(len(src)):
               src[i] = repeat(src[i], "b ... -> (b m) ...", m=beam_size)
               src_mask[i] = repeat(src_mask[i], "b ... -> (b m) ...", m=beam_size)

       beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
           input_ids=input_ids,
           next_scores=next_token_scores,
           next_tokens=next_tokens,
           next_indices=next_indices,
       )

       input_ids = torch.cat(
           (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1
       )
       cur_len += 1

   return beam_scorer.finalize(input_ids, beam_scores)
   ```

   Use `self.vocab_info.vocab_size` for `vocab_size`.

4. The old repo uses `torch.topk(..., 2 * beam_size, dim=1)`. Restore exactly this. `torch.topk` returns the requested `k` largest entries along a dimension, so the candidate count must be exactly `2 * beam_size` to match the old beam scorer’s finished-hypothesis logic. ([PyTorch Documentation][1])

5. Do not use the new target repo’s simplified beam loop that:

   * pre-expands all beams immediately,
   * uses `done_mask`,
   * calls `topk(..., beam_size)`,
   * manually pads done beams,
   * ignores `early_stopping`,
   * strips generated boundaries through `_strip_generated_boundaries_cpu`,
   * scores final beams outside `BeamSearchScorer`.

6. In `utils/beam_search.py`, make `BeamSearchScorer` active again.

   The target repo currently marks it as deprecated. Remove or update that warning.

   Preserve the old logic for:

   * `process()`
   * `finalize()`
   * `BeamHypotheses.add()`
   * `BeamHypotheses.is_done()`

   But keep the safe improvement of passing vocab explicitly if possible.

   Suggested target constructor:

   ```python
   class BeamSearchScorer:
       def __init__(
           self,
           batch_size: int,
           beam_size: int,
           alpha: float,
           do_early_stopping: bool,
           device: torch.device,
           vocab,
       ):
           ...
           self.vocab = vocab
   ```

   Then replace old `vocab.PAD_IDX`, `vocab.SOS_IDX`, `vocab.EOS_IDX` references with `self.vocab.PAD_IDX`, etc.

7. In `models/decoder.py`, restore `transform()` to the old simple full-prefix path:

   ```python
   def transform(self, src, src_mask, input_ids):
       assert len(src) == 1 and len(src_mask) == 1
       return self(src[0], src_mask[0], input_ids)
   ```

   Do not keep transform behavior that silently expands encoder features for new-cache `_rate()` unless a test proves it is required and behavior-equivalent.

8. Restore old `_rate()` behavior in `utils/generation_utils.py` unless memory constraints force chunking.

   Strict old behavior:

   ```python
   b = tgt.shape[0]
   out_hat = self.transform(src, src_mask, tgt) / temperature
   loss = ce_loss(out_hat, out, ignore_idx=self.vocab_info.pad_id, reduction="none")
   loss = rearrange(loss, "(b l) -> b l", b=b)
   mask = tgt == self.vocab_info.pad_id
   penalty = (~mask).sum(dim=1) ** alpha
   loss = -torch.sum(loss, dim=1) / penalty
   return loss
   ```

   If retaining chunking, prove it is numerically identical to old full-batch `_rate()` on a fixed tiny batch.

9. Restore `test_step()` and `test_epoch_end()` behavior in `lit_comer.py`, or provide an equivalent test path whose output format exactly matches the old repo.

   Old behavior writes `result.zip` entries as:

   ```text
   %<img_base>
   $<prediction>$
   ```

   Each file is named:

   ```text
   <img_base>.txt
   ```

10. The new standalone `scripts/test/test.py` may stay, but it must produce the same output format as old `LitCoMER.test_epoch_end()` and must use the restored beam search behavior.

## Safe Training Refactor Pieces to Preserve

Preserve these safe improvements unless they directly conflict with the three restoration targets:

* Precomputing `tgt/out` in `collate_fn`.
* Removing unnecessary `torch.cuda.empty_cache()` calls from hot training/validation hooks.
* Disabling heavy `wandb.watch(log="all")` by default.
* Disabling expensive grad norm logging by default.
* Creating the optimizer in `configure_optimizers`.
* Using `VocabInfo` instead of unsafe global shared vocab reads.
* Keeping the new `Batch` fields: `tgt`, `out`, `labels`, `lengths`.
* Keeping `Batch.pin_memory()` and non-blocking `.to()`.
* Keeping the new batch sampler if deterministic and padding semantics are restored.
* Caching transforms only if it does not change data semantics. It is safe only when augmentation is disabled; do not cache augmented training samples.

## Batch Sampler Notes

The new `BucketedBatchSampler` may be kept, but verify all of the following:

* It uses the configured seed, not the current hard-coded `seed=42`.
* Pass `config.seed_everything` into the sampler.
* It reshuffles per epoch, e.g. via `set_epoch(epoch)`.
* If PyTorch Lightning does not automatically call `set_epoch()` for this custom sampler, add a small hook or wrapper to do so.
* It does not force `pad_to_multiple` or any other new padding semantics.
* It does not break deterministic multi-seed research runs.
* It preserves the old filtering logic for `maxlen` and max image size.

Current target repo issue:

```python
class BucketedBatchSampler(..., seed: int = 42)
```

Change this to use the configured seed:

```python
seed=self.config.seed_everything
```

when constructing train/val/test samplers.

## Files Likely to Edit

| File                                        | Required Change                                                                                                                                                                                                                     | Reference File in Old Repo                  | Notes                                                                                                                                           |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs/crohme_config.yaml`                | Remove or neutralize `pad_strategy: "bucket"` and `pad_to_multiple: 32`. Keep `max_pixels_per_batch` if desired, or map it clearly to old `gpu_max_memory`.                                                                         | `configs/crohme_config.yaml`                | Old config has `gpu_max_memory: 128e4` and no padding strategy fields.                                                                          |
| `datamodule/datamodule.py`                  | Restore image padding to per-batch max H/W. Remove bucket/static rounding. Keep precomputed `tgt/out` if correct. Pass configured seed to `BucketedBatchSampler`.                                                                   | `datamodule/datamodule.py`                  | Old `collate_fn()` computes `max_height_x = max(heights_x)` and `max_width_x = max(widths_x)`.                                                  |
| `datamodule/utils.py`                       | Keep `BucketedBatchSampler` only if it remains deterministic and does not change padding semantics. Replace hard-coded `seed=42` usage with configured seed.                                                                        | `datamodule/utils.py`                       | Old `data_iterator()` sorted by image area and batched under max image memory. New sampler can emulate this, but must not control padding size. |
| `datamodule/dataset.py`                     | Usually keep the new single-sample dataset style and transform caching if safe. Do not revert unless sampler/collate behavior cannot be made correct.                                                                               | `datamodule/dataset.py`                     | Old dataset returned pre-batched tuples; new dataset returns single samples. This can remain if collate and sampler match behavior.             |
| `utils/generation_utils.py`                 | Restore old full-prefix `beam_search()`, `_beam_search()`, and `_rate()` semantics. Remove active `decode_step`/cache calls. Reintroduce active `BeamSearchScorer`. Use `VocabInfo` token IDs instead of global vocab.              | `utils/generation_utils.py`                 | This is the most important file. Current target `_beam_search()` is not old-compatible.                                                         |
| `utils/beam_search.py`                      | Make `BeamSearchScorer` active again. Restore old `process()` and `finalize()` behavior. Keep explicit vocab injection instead of global shared vocab. Remove “DEPRECATED / NOT USED” comments once active.                         | `utils/beam_search.py`                      | Old scorer handles EOS/SOS terminal logic, `early_stopping`, length penalty, and finalization.                                                  |
| `utils/__init__.py`                         | Optionally export `BeamSearchScorer` and `BeamHypotheses` again after removing unsafe global-vocab behavior.                                                                                                                        | `utils/__init__.py`                         | New repo stopped exporting them because old code read global vocab. With explicit vocab, export is safe.                                        |
| `models/decoder.py`                         | Disable/remove `init_decode_cache`, `reorder_decode_cache`, and `decode_step`. Restore active `transform()` to full-prefix path. Keep `VocabInfo`, causal mask cache, and `arm_norm_impl="legacy"` if they do not change semantics. | `models/decoder.py`                         | Old decoder has no KV-cache path.                                                                                                               |
| `models/transformer/transformer_decoder.py` | Disable/remove `TransformerDecoder.forward_step()` and `TransformerDecoderLayer.forward_step()` if no longer used. Keep normal `forward()` unchanged.                                                                               | `models/transformer/transformer_decoder.py` | Old file has only full-sequence decoder forward.                                                                                                |
| `models/transformer/attention.py`           | Disable/remove `project_kv()` and `forward_step()` if no longer used. Prefer old full-attention behavior for inference.                                                                                                             | `models/transformer/attention.py`           | Current target adds cache-specific methods.                                                                                                     |
| `models/transformer/arm.py`                 | Disable/remove `AttentionRefinementModule.forward_step()`. Ensure active ARM uses old full-prefix `forward()`. Keep `norm_impl="legacy"` default if it preserves old behavior.                                                      | `models/transformer/arm.py`                 | Old ARM computes coverage from full attention tensors.                                                                                          |
| `models/comer.py`                           | Usually only minor compatibility edits. Keep `VocabInfo` construction path. Ensure `beam_search()` still calls decoder’s restored full-prefix beam search.                                                                          | `models/comer.py`                           | Old `CoMER.beam_search()` encodes once then calls decoder beam search.                                                                          |
| `lit_comer.py`                              | Restore old test output behavior or an equivalent. Keep safe optimizer creation in `configure_optimizers()`. Keep no-op empty-cache hooks.                                                                                          | `lit_comer.py`                              | Target repo removed `test_step()` and `test_epoch_end()`; restore if using Lightning test path.                                                 |
| `scripts/test/test.py`                      | Ensure standalone inference writes the same `result.zip` format and uses restored beam search.                                                                                                                                      | `scripts/test/test.py`                      | New script can stay if behavior matches old output format.                                                                                      |
| `test_beam.py`                              | Rewrite or remove tests that require active KV-cache. Add tests for old full-prefix beam behavior instead.                                                                                                                          | No old equivalent                           | Current target tests include `decode_step` equivalence and cache tests; those conflict with this task.                                          |
| `train.py`                                  | Preserve safe training changes: explicit `vocab_info`, optional `wandb.watch`, optional grad norm logging, optimizer creation in `configure_optimizers()`. Ensure sampler seed integration if needed.                               | `train.py`                                  | Do not revert the whole file.                                                                                                                   |

## Verification Checklist

Before considering the task complete, verify:

* [ ] No active inference code path uses KV-cache.
* [ ] No active `_beam_search()` calls `decode_step()`.
* [ ] No active inference path calls `init_decode_cache()` or `reorder_decode_cache()`.
* [ ] No active ARM inference path uses incremental coverage cache.
* [ ] Padding uses old batch-max behavior.
* [ ] Padding is not rounded to multiples of 32 or any other multiple.
* [ ] Image masks match old behavior: `False` for valid image pixels, `True` for padded pixels.
* [ ] Beam search behavior matches old repo.
* [ ] `early_stopping` is honored through `BeamHypotheses.is_done()`.
* [ ] `topk(2 * beam_size)` behavior is restored.
* [ ] Length penalty / `alpha` handling matches old repo.
* [ ] EOS/SOS terminal handling matches old repo.
* [ ] `BeamSearchScorer.finalize()` behavior matches old repo.
* [ ] Approximate joint bidirectional search matches old repo.
* [ ] Test/inference output format matches old repo.
* [ ] Training still runs.
* [ ] Validation still runs.
* [ ] Inference/test still runs.
* [ ] Diff is minimal and localized.

## Suggested Tests

Run these from inside `DSP391m_Group1_FALL25_CoMER`.

### 1. Static grep checks for active KV-cache calls

```bash
grep -R "decode_step\|init_decode_cache\|reorder_decode_cache\|forward_step\|project_kv\|beam_origin\|cross_kv\|self_k\|self_v" \
  models utils lit_comer.py scripts test_beam.py \
  --exclude-dir=.git --exclude-dir=__pycache__
```

Acceptable result:

* No matches in active inference code, or
* matches only in dead/commented compatibility code that is not imported or called.

Also verify `_beam_search()` calls full-prefix `transform()`:

```bash
grep -R "self.transform(src, src_mask, input_ids).*\\[:, -1, :\\]" -n utils/generation_utils.py
grep -R "torch.topk" -n utils/generation_utils.py
grep -R "2 \\* beam_size" -n utils/generation_utils.py
```

### 2. Padding sanity test

Create a tiny synthetic batch with image shapes like:

* `[1, 17, 33]`
* `[1, 20, 31]`

Call `CROHMEDatamodule.collate_fn()` and assert:

```python
batch.imgs.shape == (2, 1, 20, 33)
batch.mask.shape == (2, 20, 33)
batch.mask[0, :17, :33].sum() == 0
batch.mask[0, 17:, :].all()
batch.mask[1, :20, :31].sum() == 0
batch.mask[1, :, 31:].all()
```

Also assert the output is **not** rounded to `(32, 64)`.

### 3. Compare padding against old behavior

Using the same two transformed tensors, compare the target repo `collate_fn()` output to the old repo’s padding rules.

Expected:

```python
new_batch.imgs.shape[-2:] == old_batch.imgs.shape[-2:]
torch.equal(new_batch.mask, old_batch.mask)
```

### 4. Small-batch forward pass test

Run one tiny forward pass through the model:

```bash
python - <<'PY'
from sconf import Config
from datamodule import CROHMEDatamodule
from lit_comer import LitCoMER

config = Config("configs/crohme_config.yaml")
dm = CROHMEDatamodule(config=config)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))

model = LitCoMER(
    config=config,
    beam_size=config.model.beam_size,
    max_len=5,
    alpha=config.model.alpha,
    early_stopping=config.model.early_stopping,
    temperature=config.model.temperature,
    vocab_info=dm.vocab.get_info(),
)

batch = batch.to(model.device)
loss = model.training_step(batch, 0)
print("loss:", float(loss.detach().cpu()))
PY
```

If CUDA/DDP setup is required in your environment, adapt device handling but keep the test small.

### 5. Beam search test on controlled dummy logits

Write a dummy `DecodeModel` subclass whose `transform()` returns scripted logits.

Test cases:

* Early EOS appears in top candidates.
* EOS outside top `beam_size` but inside `2 * beam_size`.
* `early_stopping=True` stops when enough hypotheses are complete.
* `early_stopping=False` continues according to length-penalty bound.
* L2R stops on EOS.
* R2L stops on SOS.
* Final hypotheses do not include the leading start token, matching old `finalize()`.

The test must fail if `_beam_search()` uses `topk(beam_size)` instead of `topk(2 * beam_size)`.

### 6. End-to-end inference output format test

Run inference on a tiny sample set and inspect `result.zip`:

```bash
python scripts/test/test.py --config configs/crohme_config.yaml --ckp path/to/checkpoint.ckpt --output result.zip
python - <<'PY'
import zipfile
with zipfile.ZipFile("result.zip") as z:
    names = z.namelist()
    print(names[:5])
    sample = z.read(names[0]).decode()
    print(sample)
    assert sample.startswith("%")
    assert "\n$" in sample
    assert sample.endswith("$")
PY
```

The zip entries must match the old format:

```text
%<img_base>
$<prediction>$
```

### 7. One short training smoke test

Run a minimal training smoke test, reducing epochs/batches if needed:

```bash
python train.py --config configs/crohme_config.yaml
```

If the full config is too heavy, temporarily override to a tiny dataset/subset or short epoch in a local test config. Do not commit unrelated config changes.

## Acceptance Criteria

The task is done when:

* Main repo trains successfully.
* Main repo validates successfully.
* Main repo inference uses old full-prefix beam search semantics.
* Main repo padding matches old batch-max behavior.
* KV-cache path is not used by active inference.
* ARM incremental cache behavior is not used by active inference.
* Beam search uses `BeamSearchScorer`/`BeamHypotheses` semantics from the old repo.
* `topk(2 * beam_size)` is restored.
* `early_stopping`, length penalty, EOS/SOS handling, and finalization match the old repo.
* Test/inference output format matches the old repo.
* Safe training cleanup improvements are preserved where compatible.
* No unrelated refactor is introduced.
* Git diff is minimal and easy to review.

## Final Notes for the Coding Agent

Restore behavior first, optimize later.

Use `DSP391m_Group1_FALL25_CoMER_old` as the source of truth. If uncertain, copy the old behavior rather than inventing a new one.

Do not keep the new cache-driven inference path active. Do not keep the new beam search if it changes old semantics. Do not keep padding rounded to bucket multiples.

Keep changes minimal, localized, and reviewable.