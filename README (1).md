# Anti Gravity Repair Guide: GPU Utilization & VRAM Stability for CoMER PyTorch Lightning Repo

This file is an instruction guide for Anti Gravity Agent. Place it at the repository root and ask Anti Gravity to follow it phase by phase. Do **not** let the agent make broad unrelated refactors. The goal is to fix CPU bottlenecks, CUDA synchronization points, memory allocator misuse, and unstable inference/train memory behavior while preserving the original CoMER task semantics.

## Scope

Fix the repo issues described in this README only.

### Explicitly excluded items

Do **not** implement or discuss these items in code changes unless the user explicitly asks later:

- Do not change the training validation policy around whether full beam-search ExpRate should run during validation.
- Do not change scheduler monitor behavior or scheduler metric selection.
- Do not add AMP/BF16/mixed precision work.
- Do not change `deterministic` / cuDNN benchmark behavior.
- Do not upgrade PyTorch, Lightning, CUDA, or the environment stack.

### Critical task constraint: preserve the image problem

This is an image-to-sequence training problem where image H/W is not expected to vary too much. Preserve the current task properties:

- Do not crop, rescale, stretch, or normalize H/W in a way that changes the mathematical task beyond the existing transform policy.
- If stable VRAM is needed, prefer **padding to fixed or bucketed H/W** rather than changing image geometry.
- Padding is allowed. Bucketed static padding is allowed. Resizing beyond current transform intent is not allowed unless the repo already does it.
- Keep image masks correct after padding.
- Keep label semantics exactly the same.

## Anti Gravity execution rules

When using Anti Gravity Agent:

1. Ask Anti Gravity to execute **one phase at a time**.
2. Before editing, Anti Gravity must inspect the referenced files and produce a short implementation plan.
3. Anti Gravity must make minimal diffs and avoid unrelated style rewrites.
4. Anti Gravity must run a smoke test after each phase where possible.
5. Anti Gravity must report:
   - files changed,
   - what bottleneck was fixed,
   - why semantics are preserved,
   - how to verify the fix.
6. Anti Gravity must not introduce CUDA tensors in DataLoader worker processes.
7. Anti Gravity must not add new external dependencies unless unavoidable.
8. Anti Gravity must not remove existing metrics or outputs unless the user asks.
9. Anti Gravity must not change model accuracy behavior intentionally. Performance changes should preserve numerical behavior as much as practical.

Useful Anti Gravity prompt pattern:

```text
Read README.md. Execute Phase <N> only. First inspect the referenced files, then propose a minimal patch plan. Do not modify excluded items. Preserve image H/W semantics; use padding/bucketing only if needed. After patching, run the smallest smoke test available and summarize changed files.
```

---

# Phase 0 — Baseline and safety checks

Before changing code, Anti Gravity should inspect and record the current behavior.

## 0.1 Confirm files and hot paths

Search for these symbols:

```bash
rg "empty_cache|training_step_end|validation_step_end|BeamSearchScorer|def generate|def _beam_search|to_bi_tgt_out|GradNormCallback|logger.watch|shared_vocab|torch.equal|MaskBatchNorm2d|gpu_max_memory|collate_fn|data_iterator"
```

Expected files include:

- `lit_comer.py`
- `train.py`
- `datamodule/datamodule.py`
- `datamodule/dataset.py`
- `datamodule/utils.py`
- `datamodule/transforms.py`
- `utils/utils.py`
- `utils/beam_search.py`
- `utils/generation_utils.py`
- `utils/callbacks/grad_norm_callback.py`
- `models/comer.py`
- `models/decoder.py`
- `models/transformer/attention.py`
- `models/transformer/arm.py`
- `configs/crohme_config.yaml`

## 0.2 Add or use a tiny smoke run

If the repo already has a small train/eval command, use it. If not, Anti Gravity may add a small local-only smoke script under `scripts/dev/` that verifies:

- datamodule can instantiate,
- one train batch can be loaded,
- one forward pass can run,
- loss can be computed,
- one inference call can run on a tiny batch if checkpoint/model weights are available.

Do not make this script required for production training.

## 0.3 Track memory correctly

Anti Gravity should not rely only on `nvidia-smi`. Add optional debug prints or a small utility that can report:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Interpretation:

- `allocated` changes with live tensors and can naturally vary with dynamic shapes.
- `reserved` should become more stable after removing allocator cache flushing.
- Fully fixed `allocated` VRAM is unrealistic unless shapes are bucketed/static-padded.

---

# Phase 1 — Low-risk fixes with immediate performance impact

## FIX-01 — Remove `torch.cuda.empty_cache()` from train/validation loop

### Problem

`lit_comer.py` calls `torch.cuda.empty_cache()` in step/epoch hooks. This destroys CUDA caching allocator reuse and makes VRAM appear to jump after every step.

### Files

- `lit_comer.py`

### Required change

Remove these methods entirely, or make them no-ops:

```python
def training_epoch_end(...):
    torch.cuda.empty_cache()

def validation_epoch_end(...):
    torch.cuda.empty_cache()

def training_step_end(...):
    torch.cuda.empty_cache()

def validation_step_end(...):
    torch.cuda.empty_cache()
```

### Do not do

- Do not replace them with `gc.collect()`.
- Do not call `empty_cache()` every N steps.
- Do not call `empty_cache()` in validation/test/inference loops.

### Acceptance checks

- `rg "empty_cache"` should return no calls in train/val/test hot paths.
- One train step still runs.
- `memory_reserved()` should stop dropping every step.

---

## FIX-02 — Disable per-step gradient norm sync overhead

### Problem

`GradNormCallback` computes grad norm by looping over parameters and calling `.item()` for each parameter. On CUDA this synchronizes GPU to CPU repeatedly after backward.

### Files

- `utils/callbacks/grad_norm_callback.py`
- `train.py`
- config files that register callbacks

### Required change

Make the callback opt-in and disabled by default.

Recommended implementation:

1. Add config flag such as:

```yaml
log_grad_norm: false
```

2. Only attach `GradNormCallback` if this flag is true.
3. If keeping the callback, add `log_every_n_steps` and compute less frequently.
4. Avoid `.item()` per parameter. If computing norm, accumulate tensor values on GPU and call `.item()` once at the end.

### Acceptance checks

- Default training does not instantiate `GradNormCallback`.
- `rg "GradNormCallback" train.py configs` shows it is gated behind config.
- No `.item()` inside a per-parameter loop in default training path.

---

## FIX-03 — Disable W&B `logger.watch(..., log="all")` by default

### Problem

`logger.watch(model, log="all", log_freq=100)` can log gradients/weights/histograms and introduce CPU/GPU sync plus I/O overhead.

### Files

- `train.py`
- config files

### Required change

Gate W&B watch behind config:

```yaml
wandb_watch: false
wandb_watch_log: "gradients"
wandb_watch_log_freq: 1000
```

Default should be disabled.

### Do not do

- Do not remove W&B logging entirely.
- Do not change metric names.
- Do not force offline mode.

### Acceptance checks

- Default run does not call `logger.watch`.
- Enabling config flag still works for debugging.

---

## FIX-04 — Audit distributed metric sync usage

### Problem

`sync_dist=True` is useful in DDP but can be expensive if used on many step-level logs. The current main issue is not this, but future patches must avoid adding unnecessary step syncs.

### Files

- `lit_comer.py`

### Required change

- Keep existing essential epoch-level syncs if needed.
- Do not add new `sync_dist=True` logs inside tight loops.
- If a metric is purely local/debug, log it without distributed sync.

### Acceptance checks

- No new step-level `sync_dist=True` logs are introduced.

---

# Phase 2 — Data pipeline and batch transfer refactor

## FIX-05 — Replace pre-batched dataset with sample-level dataset + bucketed batch sampler

### Problem

The current pipeline creates static pre-batches in `data_iterator()`. The DataLoader then uses default `batch_size=1`, and `collate_fn` asserts `len(batch) == 1`. This makes the dataset item itself a batch, which is hard to shuffle correctly, hard to balance in DDP, and hard to optimize.

### Files

- `datamodule/utils.py`
- `datamodule/dataset.py`
- `datamodule/datamodule.py`

### Required target design

1. Dataset returns **one sample**:

```python
{
    "fname": str,
    "image": image_or_tensor,
    "label_indices": List[int],
    "height": int,
    "width": int,
    "label_len": int,
}
```

2. A custom bucketed batch sampler returns lists of sample indices.

3. Collate receives a list of samples and pads into one tensor batch.

4. Config key `gpu_max_memory` should be treated as a pixel budget, not real VRAM.

### Implementation steps

#### Step 1 — Split data reading from batching

Refactor `data_iterator()` or create new functions:

- `load_records(...) -> List[Record]`
- `build_bucketed_batches(records, max_pixels_per_batch, max_batch_size, shuffle) -> List[List[int]]`

Do not let `Dataset.__getitem__()` return a pre-batch.

#### Step 2 — Add `BucketedBatchSampler`

The sampler should:

- group samples by similar H/W and possibly label length,
- keep batch pixel budget under `max_pixels_per_batch`,
- optionally pad batch H/W to fixed bucket sizes,
- shuffle samples or buckets each epoch,
- support `drop_last` if configured.

Pseudo-interface:

```python
class BucketedBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self, records, max_pixels_per_batch, max_batch_size, shuffle=True, drop_last=False, seed=0):
        ...

    def __iter__(self):
        ...

    def __len__(self):
        ...
```

If DDP is used, make the sampler rank-aware or compatible with distributed training. Do not change the DDP API in this phase.

#### Step 3 — Keep H/W semantics

Because this task's image H/W is nearly stable, do **not** introduce aggressive resizing. Use padding only:

- batch-max padding: pad to max H/W inside batch,
- bucket padding: pad to a fixed H/W bucket,
- global static padding: only if dataset dimensions are known and memory is acceptable.

Recommended safe default:

```yaml
pad_strategy: "bucket"   # "batch_max" | "bucket" | "static"
pad_to_multiple: 32
```

Pad H/W upward only. Never crop.

### Acceptance checks

- `collate_fn` no longer asserts `len(batch) == 1`.
- DataLoader receives `batch_sampler=...` or equivalent.
- Dataset item is a sample, not a pre-batch.
- Labels and masks remain correct.
- One train batch shape is sensible and H/W is padded, not distorted.

---

## FIX-06 — Add custom `Batch.pin_memory()` and non-blocking device transfer

### Problem

`pin_memory=True` does not automatically pin tensors inside an unknown custom dataclass unless the dataclass implements `pin_memory()`.

### Files

- `datamodule/utils.py`
- `datamodule/datamodule.py`
- `lit_comer.py` if overriding Lightning transfer hook

### Required change

Update `Batch` so all tensor fields can be pinned and moved non-blockingly.

Target shape:

```python
@dataclass
class Batch:
    img_bases: List[str]
    imgs: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor
    tgt: torch.Tensor | None = None
    out: torch.Tensor | None = None

    def pin_memory(self):
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.pin_memory(),
            mask=self.mask.pin_memory(),
            labels=self.labels.pin_memory(),
            lengths=self.lengths.pin_memory(),
            tgt=None if self.tgt is None else self.tgt.pin_memory(),
            out=None if self.out is None else self.out.pin_memory(),
        )

    def to(self, device, non_blocking=True):
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device, non_blocking=non_blocking),
            mask=self.mask.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            tgt=None if self.tgt is None else self.tgt.to(device, non_blocking=non_blocking),
            out=None if self.out is None else self.out.to(device, non_blocking=non_blocking),
        )
```

If using older Python without `|` type syntax, use `Optional[torch.Tensor]`.

### Do not do

- Do not create CUDA tensors inside DataLoader workers.
- Do not call `.cuda()` in `collate_fn` or `Dataset.__getitem__()`.

### Acceptance checks

- DataLoader still has `pin_memory=True`.
- `Batch.pin_memory()` exists and pins every tensor field.
- `Batch.to(..., non_blocking=True)` exists.
- One batch transfer to GPU works.

---

## FIX-07 — Move target construction out of `training_step`

### Problem

`training_step` calls `to_bi_tgt_out(batch.indices, self.device)`. That function takes Python lists, creates many small tensors, loops per sample, and writes into GPU tensors. This is CPU-heavy and creates small host-to-device copies.

### Files

- `lit_comer.py`
- `utils/utils.py`
- `datamodule/datamodule.py`
- `datamodule/utils.py`

### Required target design

Collate should return padded label tensors and target/output tensors on CPU. They will be pinned and transferred as one batch.

Recommended fields:

```python
labels: LongTensor[B, L]
lengths: LongTensor[B]
tgt: LongTensor[2B, L + 1]
out: LongTensor[2B, L + 1]
```

Maintain existing semantics:

- l2r target starts with `<sos>` and predicts tokens then `<eos>`.
- r2l target uses reversed sequence with the proper start/end tokens.
- Padding index remains unchanged.
- Final shape must match what `comer_model` currently expects.

### Implementation steps

1. Tokenize labels once during dataset record creation if possible.
2. In collate, pad labels with `PAD_IDX`.
3. Build l2r/r2l `tgt` and `out` using vectorized CPU tensor operations.
4. Store these tensors in `Batch`.
5. In `training_step`, use `batch.tgt` and `batch.out` directly.
6. Keep old `to_bi_tgt_out` only for backward compatibility or tests; do not use it in the hot training path.

### Acceptance checks

- `training_step` no longer calls `to_bi_tgt_out`.
- `batch.indices` is removed or no longer used in training hot path.
- Labels are tensors, not Python lists, in `Batch`.
- Loss before/after refactor is numerically plausible on one batch.

---

## FIX-08 — Cache deterministic image preprocessing when `scale_aug=false`

### Problem

When `scale_aug=false`, image resize/normalization/to-tensor work is deterministic but still repeated every epoch.

### Files

- `datamodule/dataset.py`
- `datamodule/transforms.py`
- `datamodule/datamodule.py`

### Required change

Add optional caching for transformed image tensors when augmentation is disabled.

Safe options:

1. In-memory cache per dataset instance.
2. Disk cache under a configured cache directory.
3. Preprocessed `.pt` shard files.

Recommended default:

```yaml
cache_transforms: true
cache_dir: null
```

If `cache_dir` is null, use in-memory cache only for train/val/test dataset objects.

### Preserve H/W semantics

- Apply the same existing transform output as before.
- Do not introduce new resizing behavior.
- Padding still belongs in collate, not transform, unless using explicit static-padding cache.

### Acceptance checks

- With `scale_aug=false`, repeated access to same sample avoids repeated `cv2.resize`/ToTensor work.
- With `scale_aug=true`, caching is disabled or keyed by augmentation parameters so semantics are not frozen accidentally.

---

## FIX-09 — Prevent CPU thread oversubscription from OpenCV workers

### Problem

DataLoader may use multiple workers and OpenCV may use internal threads per worker. This can oversubscribe CPU cores.

### Files

- `datamodule/datamodule.py`
- `datamodule/transforms.py`

### Required change

Add a worker init function or setup hook:

```python
def worker_init_fn(worker_id):
    try:
        import cv2
        cv2.setNumThreads(0)  # or 1, benchmark both
    except Exception:
        pass
```

Pass it to DataLoader.

Make this configurable:

```yaml
opencv_num_threads_per_worker: 0
```

### Acceptance checks

- DataLoader still works with multiple workers.
- CPU thread count is lower/more stable during training.

---

## FIX-10 — Rework `lazy_load=false` memory behavior

### Problem

When `lazy_load=false`, the dataset may hold many PIL/NumPy objects in parent process memory and then worker processes may duplicate memory depending on multiprocessing behavior.

### Files

- `datamodule/utils.py`
- `datamodule/dataset.py`
- `datamodule/datamodule.py`

### Required change

Choose one safe mode:

1. Path-based lazy loading with transform cache.
2. Preprocessed tensor cache/shards.
3. Small-dataset RAM cache with explicit memory accounting.

Do not leave an ambiguous mode where large Python object graphs are silently copied into workers.

### Acceptance checks

- Config clearly states whether data is path-loaded, RAM-cached, or disk-cached.
- Worker memory usage is predictable.

---

## FIX-11 — Rename and tune `gpu_max_memory`

### Problem

`gpu_max_memory` is not GPU memory. It is used as a pixel budget:

```python
batch_image_size = biggest_image_size * (i + 1)
```

### Files

- `configs/crohme_config.yaml`
- `datamodule/utils.py`
- `datamodule/datamodule.py`

### Required change

Rename config key:

```yaml
max_pixels_per_batch: 1280000
```

Support backward compatibility for old configs:

```python
max_pixels = cfg.get("max_pixels_per_batch", cfg.get("gpu_max_memory"))
```

### Acceptance checks

- Old config still works.
- New config name is used in code and comments.
- Documentation explains it is a pixel budget.

---

## FIX-12 — Use bucket/static padding to stabilize VRAM without changing image semantics

### Problem

Even after removing `empty_cache()`, `memory_allocated()` may vary because H/W and sequence lengths vary. Since this task has mostly stable H/W, padding can make shapes more stable without changing image content.

### Files

- `datamodule/datamodule.py`
- `datamodule/utils.py`
- config files

### Required change

Add controlled padding strategy:

```yaml
pad_strategy: "bucket"      # "batch_max" | "bucket" | "static"
pad_to_multiple: 32
static_pad_height: null
static_pad_width: null
max_label_length_bucket: null
```

Implementation rules:

- `batch_max`: current behavior, pad to max H/W in batch.
- `bucket`: pad H/W up to nearest configured multiple or bucket boundary.
- `static`: pad all batches to fixed H/W from config or dataset max.

Important:

- Only pad upward.
- Do not crop.
- Do not rescale differently.
- Ensure `mask` marks padded pixels correctly.

### Acceptance checks

- With `pad_strategy=bucket`, batch H/W has fewer unique shapes.
- With `pad_strategy=static`, batch H/W is fixed.
- Model output remains valid and masks are correct.

---

# Phase 3 — Beam search and inference implementation refactor

## FIX-13 — Remove `.item()` and CUDA bool from beam-search inner loop

### Problem

`utils/beam_search.py` uses `.item()` and CUDA tensor booleans in Python `if`/`while`. This forces CPU-GPU synchronization during every decode step.

### Files

- `utils/beam_search.py`
- `utils/generation_utils.py`

### Required target design

Beam state should be represented by tensors:

```python
beam_scores: FloatTensor[B, beam]
beam_tokens: LongTensor[B, beam, T]
done: BoolTensor[B, beam]
```

Per step:

1. Compute logits for all active beams.
2. Mask finished beams.
3. Apply EOS handling with tensor masks.
4. Use `topk` to select next beams.
5. Use `gather` to update sequences and scores.

Avoid:

- `.item()` inside the decode loop,
- `.tolist()` inside the decode loop,
- Python list of per-beam hypotheses,
- `if cuda_tensor:` or `while not cuda_tensor:`.

Use a fixed Python loop:

```python
for step in range(max_len):
    ...
```

Optional early stopping may check a CPU bool only every K steps, not every beam/token. The simplest safe version can decode until `max_len` and rely on EOS masks.

### Acceptance checks

- `rg "\.item\(" utils/beam_search.py utils/generation_utils.py` returns no inner-loop sync calls.
- `rg "is_done|_done"` shows no CUDA bool used directly in Python control flow.
- Beam output shape and token semantics match old implementation on a tiny case.

---

## FIX-14 — Avoid physical repeat/copy of encoder features for beam search

### Problem

Inference duplicates encoder features for bidirectional decoding and again for beam size. This causes dynamic VRAM spikes.

### Files

- `utils/generation_utils.py`
- `models/comer.py`
- `models/decoder.py`

### Required change

Replace physical `repeat` of `src` and `src_mask` with index-based beam mapping.

Preferred design:

```python
encoder_out: original [B, ...]
beam_origin: LongTensor[B, beam]
```

When decoder needs beam-aligned encoder features, use controlled indexing/gather or expand views where safe. Avoid materializing `[B * beam, ...]` copies unless absolutely required by an existing module.

### Acceptance checks

- `repeat(src[i], "b ... -> (b m) ...")` no longer appears in inference hot path.
- Peak inference VRAM decreases for same batch/beam size.
- Outputs remain equivalent on a small deterministic input.

---

## FIX-15 — Add incremental decoder KV-cache for inference

### Problem

Current inference calls decoder on the full prefix each token step. Step T recomputes tokens `1..T` from scratch.

### Files

- `models/decoder.py`
- `models/transformer/decoder_layer.py` or equivalent decoder layer files
- `utils/generation_utils.py`

### Required design

Add an inference-only decode path:

```python
def decode_step(self, encoder_out, encoder_mask, last_token, cache, step):
    # returns next logits and updated cache
```

Cache should contain per-layer self-attention K/V. Cross-attention encoder K/V may also be cached or reused without recomputing projections if feasible.

Implementation requirements:

- Training forward path must remain unchanged.
- Existing full-sequence decoder remains available for training/loss.
- KV-cache path is used only by generation/inference.
- Beam reordering must reorder cache tensors using selected beam indices.

### Acceptance checks

- Generation no longer calls full decoder on the entire `input_ids` prefix each step.
- `decode_step` produces logits compatible with full decoder on a tiny example.
- Beam cache reorder works after `topk/gather` beam selection.

---

## FIX-16 — Optimize bidirectional candidate re-scoring `_rate()` without changing scoring semantics

### Problem

After beam search, `_rate()` forwards all candidate sequences again for reverse scoring. This can be expensive and memory-heavy.

### Files

- `utils/generation_utils.py`

### Required change

Keep the same scoring semantics, but make implementation less memory-heavy:

- Do not repeat encoder features physically for all candidates if avoidable.
- Batch candidates in chunks when memory would spike.
- Use tensorized target creation, not Python list loops.
- Move CPU conversion to the end only.

### Acceptance checks

- Same candidate scoring formula as before.
- Lower peak memory for same beam size.
- No `.item()`/`.tolist()` inside hot candidate scoring loops.

---

## FIX-17 — Convert beam outputs to CPU once at the end

### Problem

`Hypothesis.__init__()` calls `.tolist()` on CUDA tensors, and some code uses CUDA scalar tensors to index Python lists. This causes CPU-GPU sync at scattered points.

### Files

- `utils/utils.py`
- `utils/generation_utils.py`

### Required change

- Keep beam outputs as tensors during generation.
- At the very end, call `.detach().cpu()` once for final sequences and scores.
- Construct `Hypothesis` objects from CPU lists only after generation is complete.

### Acceptance checks

- `Hypothesis` no longer receives CUDA tensors.
- `.tolist()` appears only after `.cpu()` and outside hot loops.

---

## FIX-18 — Cache causal masks and repeated decode masks

### Problem

Decoder creates causal masks repeatedly, including during autoregressive decode.

### Files

- `models/decoder.py`

### Required change

Cache masks per device/dtype/max_len:

```python
self._causal_mask_cache = {}
```

Return slices instead of allocating a new mask every call.

### Acceptance checks

- Mask creation allocation count is reduced.
- Training and inference shapes still match.

---

# Phase 4 — Model hot-path cleanup

## FIX-19 — Remove `torch.equal()` from attention branch decisions

### Problem

`torch.equal(query, key)` on CUDA tensors can force CPU-visible boolean decisions in the hot path.

### Files

- `models/transformer/attention.py`

### Required change

Replace equality checks with explicit control flow.

Preferred options:

1. Pass a flag such as `is_self_attention=True` from caller.
2. Use object identity only when safe: `query is key is value`.
3. Split self-attention and cross-attention call sites.

Do not compare full tensor values to decide execution path.

### Acceptance checks

- `rg "torch.equal" models/transformer/attention.py` returns no hot-path usage.
- Self-attention and cross-attention still route correctly.

---

## FIX-20 — Replace ARM boolean indexing/scatter with a more GPU-friendly path

### Problem

`MaskBatchNorm2d` uses boolean indexing and writeback:

```python
flat_x = x[not_mask, :]
flat_x = self.bn(flat_x)
x[not_mask, :] = flat_x
```

This creates gather/scatter patterns that are inefficient on GPU.

### Files

- `models/transformer/arm.py`

### Required change

Implement a vectorized masked normalization or replace with a semantically acceptable normalization layer.

Options:

1. Masked mean/variance reductions over valid pixels.
2. GroupNorm/LayerNorm if accuracy remains acceptable.
3. Keep BatchNorm but avoid boolean scatter in the forward hot path.

### Safety requirement

This can affect accuracy. Anti Gravity must add an easy config toggle:

```yaml
arm_norm_impl: "legacy"  # "legacy" | "masked_vectorized" | "groupnorm"
```

Default can remain `legacy` until tested. Implement new path but do not force it unless the user chooses.

### Acceptance checks

- Legacy path still works.
- New path runs on one forward pass.
- No semantic-breaking change is forced by default.

---

## FIX-21 — Review training-time feature duplication for bidirectional decoding

### Problem

Training forward duplicates encoder features:

```python
feature = torch.cat((feature, feature), dim=0)
mask = torch.cat((mask, mask), dim=0)
```

This is semantically used for bidirectional training but creates physical copies.

### Files

- `models/comer.py`
- `models/decoder.py`

### Required change

First, measure memory impact. Then only optimize if safe.

Possible safe approaches:

1. Keep current behavior if decoder requires contiguous `[2B, ...]` tensors and optimization is risky.
2. Use `expand`/view only if downstream operations do not require writeable contiguous memory.
3. Run two directional passes if memory is more important than speed.
4. Refactor decoder to accept direction dimension without copying.

### Acceptance checks

- No change unless equivalence is verified on a tiny batch.
- If changed, output logits shape and loss computation remain identical.

---

# Phase 5 — Inference and metric output cleanup

## FIX-22 — Repair stale inference/test script to match current config API

### Problem

The test/inference script appears stale: it imports from a `comer` package path and instantiates `CROHMEDatamodule` with arguments that do not match the current `config` constructor.

### Files

- `scripts/test/test.py`
- `train.py`
- `datamodule/datamodule.py`
- `lit_comer.py`

### Required change

Create a current inference entrypoint that:

1. Loads the same config format as training.
2. Instantiates `CROHMEDatamodule(config)`.
3. Loads `LitCoMER` checkpoint correctly.
4. Calls `model.eval()`.
5. Uses `torch.no_grad()` or `torch.inference_mode()` for standalone inference.
6. Uses the optimized DataLoader/Batch path.
7. Writes predictions in the same expected format as before.

### Acceptance checks

- Script imports local modules consistently with training.
- Script does not depend on import-order side effects.
- One-batch inference runs.

---

## FIX-23 — Remove global vocab import-order dependency

### Problem

Several modules read `CROHMEDatamodule.shared_vocab` at import time. This relies on DataModule being initialized before model/utils import.

### Files

- `lit_comer.py`
- `models/decoder.py`
- `utils/generation_utils.py`
- `utils/utils.py`
- `datamodule/datamodule.py`

### Required change

Pass vocabulary/config explicitly.

Options:

1. Pass `vocab` into `LitCoMER` constructor.
2. Pass `vocab_size`, special token ids, and decode helpers into model/generation utilities.
3. Keep a small compatibility fallback but do not rely on class global state.

Recommended data object:

```python
@dataclass(frozen=True)
class VocabInfo:
    vocab_size: int
    sos_id: int
    eos_id: int
    pad_id: int
    words: Any
```

### Acceptance checks

- Importing `LitCoMER` before creating `CROHMEDatamodule` no longer crashes or creates invalid vocab.
- Training still uses the same vocabulary.
- Inference script can load config/checkpoint without import-order hacks.

---

## FIX-24 — Reduce Python overhead in ExpRate conversion

### Problem

`ExpRateRecorder.update()` converts predictions and labels to strings one by one in Python. This is small compared with beam search, but it is still validation/test CPU overhead.

### Files

- `utils/utils.py`

### Required change

Keep semantics, but avoid unnecessary repeated work:

- Compare token ids first when possible.
- Convert to label strings only when needed for reporting/debug.
- Batch CPU conversion after all GPU work is complete.

### Acceptance checks

- ExpRate result is unchanged on a small known set.
- No CUDA tensors are converted to Python lists inside generation hot loops.

---

## FIX-25 — Make test zip/output writing less blocking

### Problem

`test_epoch_end` writes output zip files at the end of test. This is not a training bottleneck but can block large inference runs.

### Files

- `lit_comer.py`

### Required change

Keep output format unchanged, but make writing clear and isolated:

- Move zip writing into a utility function.
- Ensure all tensors are already CPU data before writing.
- If outputs are large, write incrementally or chunked.

### Acceptance checks

- Output zip contents and names remain compatible.
- Test/inference still produces expected artifacts.

---

# Phase 6 — Optimizer/scheduler lifecycle cleanup without changing training policy

## FIX-26 — Lazy-create only the selected optimizer

### Problem

The module creates multiple optimizer objects in `__init__`. This is not the main GPU memory issue because most optimizer state is lazy-created after `.step()`, but it is confusing and unnecessary.

### Files

- `lit_comer.py`

### Required change

Replace pre-created optimizer dict with factory functions:

```python
def build_optimizer(self):
    name = self.optimizer_use
    if name == "AdamW":
        return torch.optim.AdamW(self.parameters(), **cfg)
    ...
```

`configure_optimizers()` should call this once.

### Acceptance checks

- Only the selected optimizer is instantiated.
- Checkpoint save/load still works.
- No training hyperparameters change.

---

## FIX-27 — Create scheduler from the selected optimizer in `configure_optimizers()`

### Problem

Scheduler creation currently happens early. It is cleaner to create it after the actual optimizer object exists.

### Files

- `lit_comer.py`

### Required change

Inside `configure_optimizers()`:

```python
optimizer = self.build_optimizer()
scheduler = self.build_scheduler(optimizer)
return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
```

Do not change scheduler metric/policy in this guide.

### Acceptance checks

- Scheduler receives the exact optimizer object returned to Lightning.
- Existing scheduler config values are preserved.

---

# Phase 7 — Final verification checklist

After completing all included phases, Anti Gravity should run this checklist.

## Static grep checks

```bash
rg "empty_cache"
rg "\.item\(" utils/beam_search.py utils/generation_utils.py
rg "torch.equal" models/transformer/attention.py
rg "assert len\(batch\) == 1" datamodule
rg "to_bi_tgt_out\(" lit_comer.py
rg "logger.watch" train.py
rg "shared_vocab"
```

Expected:

- No `empty_cache()` in hot paths.
- No `.item()` in beam inner loop.
- No `torch.equal()` in attention hot path.
- No `assert len(batch) == 1` collate anti-pattern.
- `training_step` does not call `to_bi_tgt_out`.
- `logger.watch` is config-gated.
- `shared_vocab` is removed or only used as a compatibility fallback.

## Runtime checks

Run the smallest available smoke commands:

1. Load datamodule.
2. Fetch one train batch.
3. Transfer batch to GPU.
4. Run one forward/loss.
5. Run one backward if possible.
6. Run one generation/inference call if checkpoint/model setup allows.
7. Check masks and label tensors manually on one batch.

## Memory checks

Record before/after:

```python
torch.cuda.reset_peak_memory_stats()
# run one or a few steps
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())
print(torch.cuda.max_memory_allocated())
print(torch.cuda.max_memory_reserved())
```

Expected:

- `memory_reserved` should be more stable after removing `empty_cache()`.
- `memory_allocated` may still vary unless bucket/static padding is enabled.
- Inference peak VRAM should reduce after removing physical beam repeats.

## Semantic checks

Anti Gravity must confirm:

- No image content is cropped.
- No H/W-changing resize is added beyond existing transforms.
- Padding masks correctly mark padded regions.
- Special token IDs are unchanged.
- Output sequence format is unchanged.
- Checkpoint loading still works.

---

# Recommended order for Anti Gravity PRs

Use small PR-sized changes:

1. **PR 1:** remove `empty_cache`, gate grad norm, gate W&B watch.
2. **PR 2:** refactor `Batch` with `pin_memory()` and non-blocking transfer.
3. **PR 3:** move label/target tensor construction into collate.
4. **PR 4:** replace pre-batched dataset with sample dataset + bucketed batch sampler.
5. **PR 5:** add transform cache and OpenCV worker thread control.
6. **PR 6:** add bucket/static padding options for stable VRAM.
7. **PR 7:** tensorize beam search and remove sync points.
8. **PR 8:** remove physical encoder feature repeat in inference.
9. **PR 9:** add KV-cache inference path.
10. **PR 10:** clean attention/ARM/model hot paths with toggles.
11. **PR 11:** fix inference script and remove global vocab dependency.
12. **PR 12:** optimizer/scheduler lifecycle cleanup.

Do not combine all changes into one giant patch.

---

# Summary for Anti Gravity Agent

Main goal: keep GPU fed, stop CPU/GPU sync, stop allocator cache flushing, and reduce unnecessary memory copies.

Do first:

- Remove `empty_cache()` from loops.
- Disable heavy debug logging by default.
- Make custom batches pinnable and transferable with `non_blocking=True`.
- Move labels/targets to tensorized collate.
- Replace pre-batched dataset with sample dataset + bucketed sampler.

Do later:

- Rewrite beam search tensorized.
- Avoid physical encoder feature repeat.
- Add KV-cache decode.
- Clean attention/ARM hot paths safely.
- Fix inference script and vocab dependency.

Never do in this guide:

- Do not change validation policy.
- Do not change scheduler monitor.
- Do not add AMP/BF16.
- Do not change deterministic settings.
- Do not upgrade framework versions.
