# CoMER Refactor Fix Guide for Coding Agents

This document is the only implementation guide for the remaining correctness fixes in the refactored CoMER repository.

## Repository roles

- `DSP391m_Group1_FALL25_COMER`: refactored repository. Edit this one only.
- `DSP391m_Group1_FALL25_COMER old`: original repository. Use it only as the behavior reference.

## Goal

Keep the refactor speedups, but make the training and inference logic match the original repository as closely as possible.

The required final state is:

1. ARM normalization is hard-coded to the original legacy behavior.
2. The `arm_norm_impl` config option and all alternative ARM normalization implementations are removed.
3. Train batch shuffling changes across epochs when `BucketedBatchSampler(shuffle=True)` is used.
4. Checkpoint inference/evaluation always receives explicit `vocab_info`.
5. Official evaluation is not affected by DDP duplicate padding.
6. Existing correct refactor logic stays unchanged.

## Do not change

Do not rewrite the model architecture.
Do not change `to_bi_tgt_out_from_padded` unless a test proves it is wrong.
Do not change beam search scoring, beam size logic, reverse-direction reranking, or EOS/SOS handling.
Do not restore global vocab imports in model or generation code.
Do not edit the old repository.

---

## Fix 1 - Remove `arm_norm_impl` and hard-code legacy ARM normalization

The project no longer needs multiple ARM normalization modes. The only allowed behavior is the original legacy BatchNorm-on-valid-pixels behavior.

### 1.1 Edit `configs/crohme_config.yaml`

Remove this line completely:

```yaml
arm_norm_impl: "legacy"
```

Do not replace it with another config key.

### 1.2 Edit `models/comer.py`

Find this block:

```python
arm_norm_impl = mcfg.get("arm_norm_impl", "legacy")
...
self.decoder = Decoder(
    ...
    vocab_info=vocab_info,
    arm_norm_impl=arm_norm_impl,
)
```

Change it to:

```python
self.decoder = Decoder(
    d_model=d_model,
    nhead=nhead,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    dc=dc,
    cross_coverage=cross_coverage,
    self_coverage=self_coverage,
    vocab_info=vocab_info,
)
```

Requirements:

- Delete the local variable `arm_norm_impl`.
- Delete the `arm_norm_impl=...` argument.
- Keep every other argument unchanged.

### 1.3 Edit `models/decoder.py`

Remove `arm_norm_impl` from `_build_transformer_decoder`.

Before:

```python
def _build_transformer_decoder(..., self_coverage: bool, arm_norm_impl: str = "legacy"):
    ...
    arm = AttentionRefinementModule(..., norm_impl=arm_norm_impl)
```

After:

```python
def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
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

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder
```

Remove `arm_norm_impl` from `Decoder.__init__`.

Before:

```python
class Decoder(DecodeModel):
    def __init__(..., vocab_info: VocabInfo, arm_norm_impl: str = "legacy"):
```

After:

```python
class Decoder(DecodeModel):
    def __init__(..., vocab_info: VocabInfo):
```

Also remove `arm_norm_impl=arm_norm_impl` from the call to `_build_transformer_decoder`.

### 1.4 Edit `models/transformer/arm.py`

Hard-code `MaskBatchNorm2d` to the original legacy implementation.

The final `MaskBatchNorm2d` must have no `impl` argument, no `GroupNorm`, no `masked_vectorized`, and no mode branch.

Use this final shape:

```python
class MaskBatchNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor
            [b, d, h, w]
        mask: Tensor
            [b, 1, h, w], True means padded/invalid.

        Returns
        -------
        Tensor
            [b, d, h, w]
        """
        x = rearrange(x, "b d h w -> b h w d")
        mask = mask.squeeze(1)
        not_mask = ~mask

        flat_x = x[not_mask, :]
        flat_x = self.bn(flat_x)
        x[not_mask, :] = flat_x

        x = rearrange(x, "b h w d -> b d h w")
        return x
```

Then change `AttentionRefinementModule.__init__`.

Before:

```python
def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool, norm_impl: str = "legacy"):
    ...
    self.post_norm = MaskBatchNorm2d(nhead, impl=norm_impl)
```

After:

```python
def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool):
    ...
    self.post_norm = MaskBatchNorm2d(nhead)
```

### 1.5 Remove all dead ARM-normalization code

After the edits, this command must return no matches:

```bash
grep -R "arm_norm_impl\|norm_impl\|masked_vectorized\|groupnorm" -n configs models utils lit_comer.py train.py
```

If it still returns matches, remove them unless the match is inside this README.

---

## Fix 2 - Make epoch shuffling actually change between train epochs

Current risk: `LitCoMER.on_train_epoch_start()` may not access the real DataLoader instance under PyTorch Lightning 1.4.9. If the real sampler is not found, `BucketedBatchSampler.set_epoch()` is never called, and train batches are shuffled with the same seed every epoch.

### 2.1 Store the train sampler in `CROHMEDatamodule`

Edit `datamodule/datamodule.py`.

In `CROHMEDatamodule.__init__`, add:

```python
self.train_batch_sampler = None
self.val_batch_sampler = None
self.test_batch_sampler = None
```

In `train_dataloader`, replace the local-only sampler with an instance attribute.

Before:

```python
batch_sampler = BucketedBatchSampler(...)
return DataLoader(
    dataset=self.train_dataset,
    batch_sampler=batch_sampler,
    ...
)
```

After:

```python
self.train_batch_sampler = BucketedBatchSampler(
    data=self.train_dataset.dataset,
    max_pixels_per_batch=self.max_pixels_per_batch,
    max_batch_size=self.train_batch_size,
    shuffle=True,
    maxlen=self.maxlen,
    max_image_size=self.max_pixels_per_batch,
    seed=self.config.seed_everything,
)
return DataLoader(
    dataset=self.train_dataset,
    batch_sampler=self.train_batch_sampler,
    num_workers=self.num_workers,
    collate_fn=self.collate_fn,
    pin_memory=self.pin_memory,
    persistent_workers=self.persistent_workers,
    worker_init_fn=self._get_worker_init_fn(),
)
```

Do the same for validation and test if useful for debugging, but only train needs `set_epoch`:

```python
self.val_batch_sampler = BucketedBatchSampler(... shuffle=False ...)
self.test_batch_sampler = BucketedBatchSampler(... shuffle=False ...)
```

### 2.2 Replace `LitCoMER.on_train_epoch_start`

Edit `lit_comer.py`.

Replace the current method with this fail-fast version:

```python
def on_train_epoch_start(self):
    sampler = None

    datamodule = getattr(self.trainer, "datamodule", None)
    if datamodule is not None:
        sampler = getattr(datamodule, "train_batch_sampler", None)

    if sampler is None:
        loaders = getattr(self.trainer, "train_dataloaders", None)
        if loaders is None:
            loaders = getattr(self.trainer, "train_dataloader", None)
        if loaders is not None and not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        if loaders:
            for loader in loaders:
                candidate = getattr(loader, "batch_sampler", None)
                if hasattr(candidate, "set_epoch"):
                    sampler = candidate
                    break

    if not hasattr(sampler, "set_epoch"):
        raise RuntimeError(
            "Could not find BucketedBatchSampler in on_train_epoch_start; "
            "epoch-dependent shuffling would be frozen."
        )

    sampler.set_epoch(int(self.current_epoch))
```

Do not silently ignore failures here. If the sampler is not found, training should stop instead of silently using the same shuffle order forever.

### 2.3 Add a sampler regression test

Create `tests/test_sampler_epoch.py` or add this to an existing test file:

```python
from datamodule.utils import BucketedBatchSampler


def test_bucketed_batch_sampler_changes_order_between_epochs():
    data = []
    for i in range(20):
        # (fname, (width, height), label_tokens)
        data.append((f"img_{i}", (10 + i, 20 + i), ["a", "b"]))

    sampler = BucketedBatchSampler(
        data=data,
        max_pixels_per_batch=10_000,
        max_batch_size=2,
        shuffle=True,
        maxlen=200,
        max_image_size=10_000,
        seed=7,
    )

    sampler.set_epoch(0)
    order0 = list(iter(sampler))

    sampler.set_epoch(1)
    order1 = list(iter(sampler))

    assert order0 != order1
```

---

## Fix 3 - Avoid incorrect official validation/test metrics from DDP duplicate padding

`BucketedBatchSampler` pads batches across DDP ranks when the number of batches is not divisible by `world_size`. This is acceptable for training stability, but it can duplicate samples during validation/test and slightly bias ExpRate.

Required policy:

- Training may still run with DDP.
- Official validation/test metrics must be produced with a single process.
- If a standalone eval/test entrypoint exists, force it to use one GPU/process or document the command clearly.

### 3.1 Add an explicit warning in code

In `datamodule/datamodule.py`, add a comment above `val_dataloader` and `test_dataloader`:

```python
# NOTE: In DDP, BucketedBatchSampler may pad by repeating batches so each rank
# has the same number of steps. This is fine for training-time validation used
# as a rough signal, but official ExpRate should be computed with a single
# process to avoid counting duplicated samples.
```

### 3.2 Add an official single-process evaluation note

Add or update the project README/eval instructions with this command pattern:

```bash
# Official metric run: single process only.
# Adjust the checkpoint/config arguments to the project's eval script.
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/crohme_config.yaml --trainer.gpus 1 --trainer.accelerator null
```

If the project has a separate test/eval script, use that script instead of `train.py`. The important part is: official ExpRate must not be reported from multi-process DDP validation/test unless duplicate samples are explicitly removed.

### 3.3 Optional stronger implementation

Only do this if exact DDP validation/test metrics are required.

Implement a metric path that carries `img_bases` into the recorder and skips duplicate `img_base` values. This needs careful distributed handling. Do not add a half-fix that only deduplicates per process while still allowing cross-rank duplicates.

Accepted minimal fix: official eval is single-process.

---

## Fix 4 - Ensure checkpoint inference/evaluation always passes `vocab_info`

The refactor correctly removed global vocab imports from model code. Keep it that way.

### 4.1 Audit all checkpoint loading calls

Run:

```bash
grep -R "load_from_checkpoint" -n .
```

Every `LitCoMER.load_from_checkpoint(...)` call must pass `vocab_info=data_module.vocab.get_info()` or an equivalent `Vocab(...).get_info()`.

Correct pattern:

```python
from datamodule import CROHMEDatamodule
from lit_comer import LitCoMER


data_module = CROHMEDatamodule(config=config)
model = LitCoMER.load_from_checkpoint(
    checkpoint_path,
    vocab_info=data_module.vocab.get_info(),
)
model.eval()
```

Also keep the new-training path as:

```python
model = LitCoMER(
    config=config,
    beam_size=config.model.beam_size,
    max_len=config.model.max_len,
    alpha=config.model.alpha,
    early_stopping=config.model.early_stopping,
    temperature=config.model.temperature,
    vocab_info=data_module.vocab.get_info(),
)
```

### 4.2 Do not reintroduce global vocab dependencies

These patterns must not exist in model or generation files:

```python
from datamodule.datamodule import CROHMEDatamodule
vocab = CROHMEDatamodule.shared_vocab
vocab_size = len(vocab)
```

Run:

```bash
grep -R "shared_vocab\|vocab_size = len(vocab)\|CROHMEDatamodule.shared_vocab" -n models utils lit_comer.py
```

Allowed exception: `datamodule/datamodule.py` may keep `CROHMEDatamodule.shared_vocab` as an internal compatibility cache.

---

## Fix 5 - Keep known-correct refactor behavior unchanged

The following refactor pieces are considered correct. Do not change them unless a test fails and the failure proves a correctness bug.

- `to_bi_tgt_out_from_padded`: must keep the same target/output logic as the original `to_bi_tgt_out`.
- `collate_fn`: may continue precomputing `batch.tgt` and `batch.out` on CPU.
- `Batch.pin_memory()` and `Batch.to(...)`: keep them unless a concrete DataLoader issue appears.
- Attention self-detection using identity checks is acceptable in this architecture because self-attention calls use `tgt, tgt, tgt`, and cross-attention calls use `memory, memory` for key/value.
- Optimizer and scheduler lazy creation in `configure_optimizers` is correct.

---

## Required verification checklist

Run these checks before finishing.

### Static checks

```bash
# 1. No configurable ARM normalization remains.
grep -R "arm_norm_impl\|norm_impl\|masked_vectorized\|groupnorm" -n configs models utils lit_comer.py train.py

# Expected: no output.

# 2. No global model/generation vocab import regression.
grep -R "shared_vocab\|vocab_size = len(vocab)\|CROHMEDatamodule.shared_vocab" -n models utils lit_comer.py

# Expected: no output outside datamodule internals.

# 3. All checkpoint loads pass vocab_info.
grep -R "load_from_checkpoint" -n .
```

### Python syntax check

```bash
python -m py_compile \
  datamodule/datamodule.py \
  datamodule/utils.py \
  lit_comer.py \
  models/comer.py \
  models/decoder.py \
  models/transformer/arm.py \
  models/transformer/attention.py \
  models/transformer/transformer_decoder.py \
  utils/generation_utils.py \
  utils/utils.py \
  train.py
```

### Unit/smoke tests

```bash
python test_beam.py
pytest -q tests/test_sampler_epoch.py
```

If `pytest` is not available, run the sampler test as a simple script or add it to `test_beam.py` temporarily.

### Manual sanity checks

1. Start a short training run for 2 epochs.
2. Confirm `on_train_epoch_start` does not raise.
3. Print or log the first 3 train batches for epoch 0 and epoch 1.
4. Confirm the order differs when `shuffle=True`.
5. Run a single-process eval/inference smoke test with a checkpoint and `vocab_info`.
6. Confirm generated predictions can be converted with `vocab_info.words.indices2label(...)`.

---

## Definition of done

The task is complete only when all of these are true:

- `arm_norm_impl` is removed from config and code.
- `MaskBatchNorm2d` has exactly one behavior: legacy valid-pixel BatchNorm1d.
- `AttentionRefinementModule` has no `norm_impl` argument.
- `CoMER` and `Decoder` do not pass or accept ARM normalization mode arguments.
- `BucketedBatchSampler.set_epoch()` is definitely called for train epochs.
- A regression test proves shuffle order changes between epoch 0 and epoch 1.
- All `load_from_checkpoint` calls pass `vocab_info`.
- Official ExpRate is documented or enforced as single-process unless a complete distributed dedup implementation is added.
- `python test_beam.py` still passes.
- Syntax checks pass.

