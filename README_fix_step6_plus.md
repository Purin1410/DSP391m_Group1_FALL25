# Anti Gravity Repair Instructions — CoMER GPU Pipeline Fixes From Step 6 Onward

This file is an execution guide for Anti Gravity Agent. It focuses only on the remaining fixes **from step 6 onward** after the user manually fixes the first 5 critical blockers.

## Repository layout and hard boundaries

There are two project folders in the workspace:

- `DSP391m_Group1_FALL25_CoMER`  
  This is the **current working repo**. All edits must happen only here.

- `DSP391m_Group1_FALL25_CoMER_old`  
  This is the **original untouched repo**. Do **not** modify, delete, format, rename, move, or write files inside this folder.  
  You may inspect it read-only only when you need to compare original behavior, but every patch must target `DSP391m_Group1_FALL25_CoMER`.

If you are unsure which folder you are editing, stop and run:

```bash
pwd
git status
```

Expected working directory for edits:

```bash
DSP391m_Group1_FALL25_CoMER
```

## Out of scope

The user will manually fix the first 5 blockers. Do not spend time editing them unless the user explicitly asks.

Assume these are already handled or being handled by the user:

1. Restore `extract_data()` and `_resolve_image_path()`.
2. Fix `datamodule/datamodule.py setup()` builder signatures.
3. Replace stale `self.gpu_max_memory` references with `self.max_pixels_per_batch` or remove them from builder calls.
4. Fix `to_bi_tgt_out()` / target construction signature mismatch.
5. Restore scheduler config fields in `LitCoMER.__init__`.

Also do **not** implement these broad changes in this task:

- AMP / mixed precision changes.
- Deterministic / cuDNN benchmark changes.
- PyTorch / Lightning version upgrades.
- Validation policy changes such as removing validation beam search entirely.
- Scheduler monitor policy changes unless needed only to keep code from crashing.

## Core goal

Make the current repo usable and closer to the README performance-fix plan without changing the mathematical task.

The task trains on image data where image H/W should remain effectively stable. Do not crop, distort, or destructively resize images beyond existing task behavior. If VRAM stabilization is needed, prefer:

- padding to a fixed or bucketed H/W,
- padding to multiples such as 16 or 32,
- controlled bucketing by H/W and label length.

Do not alter image semantics.

## Start here: Step 6 — mandatory smoke checks

Before doing any additional optimization work, verify that the user's first 5 fixes are truly working.

Run from `DSP391m_Group1_FALL25_CoMER`:

```bash
python -m compileall .
```

Then run grep checks:

```bash
grep -R "empty_cache" -n . --exclude-dir=.git
grep -R "self.gpu_max_memory" -n . --exclude-dir=.git
grep -R "extract_data" -n datamodule utils . --exclude-dir=.git
grep -R "scheduler_use\|scheduler_cfg\|scheduler_interval\|scheduler_monitor" -n lit_comer.py configs . --exclude-dir=.git
```

Create a minimal Python smoke test script, or run equivalent inline checks, to validate these items:

```python
from omegaconf import OmegaConf
from datamodule import CROHMEDatamodule
from lit_comer import LitCoMER

config = OmegaConf.load("configs/crohme_config.yaml")

dm = CROHMEDatamodule(config)
dm.setup("fit")

train_loader = dm.train_dataloader()
batch = next(iter(train_loader))

print(type(batch))
print(batch.imgs.shape, batch.mask.shape)
print(batch.tgt.shape, batch.out.shape)
print(batch.labels.shape, batch.lengths.shape)

model = LitCoMER(config)
optim_config = model.configure_optimizers()
print(type(optim_config))
```

If dataset files are unavailable in the environment, do not fake success. Instead:

1. Run all static checks that do not need the dataset.
2. Explain clearly that dataset-dependent smoke checks could not run.
3. Continue only with code-level fixes that are safe without dataset access.

Acceptance for Step 6:

- `python -m compileall .` passes.
- No hot-path `torch.cuda.empty_cache()` calls remain.
- No stale `self.gpu_max_memory` references remain in live code.
- `dm.setup("fit")` can run when data is present.
- `next(iter(dm.train_dataloader()))` returns a batch with `imgs`, `mask`, `labels`, `lengths`, `tgt`, and `out`.
- `model.configure_optimizers()` runs without `AttributeError`.

## Step 7 — clean up stale `utils/beam_search.py`

Problem: the previous review found that `utils/beam_search.py` still contains old Python-driven beam-search code with `.item()` and CUDA bool sync. Even if new generation code no longer uses it, stale exported code is dangerous.

Actions:

1. Find all imports/usages:

```bash
grep -R "BeamSearchScorer\|from utils.beam_search\|import .*beam_search" -n . --exclude-dir=.git
```

2. If `utils/beam_search.py` is unused:
   - remove it from `utils/__init__.py` exports if present,
   - keep the file only as deprecated compatibility if necessary,
   - add a clear warning comment at the top saying it is not used by the optimized generation path,
   - ensure no training/inference code imports it.

3. If it is used:
   - remove `.item()` calls from inner loops,
   - remove Python `if` or `while` decisions based on CUDA tensors,
   - replace per-beam Python state transitions with tensorized score/state operations where feasible,
   - delay CPU conversion until after decoding finishes.

Forbidden in hot beam-search loops:

```python
cuda_tensor.item()
bool(cuda_tensor)
if cuda_tensor:
while cuda_tensor:
cuda_tensor.tolist()
```

Allowed only after decoding is finished:

```python
final_ids = ids.detach().cpu().tolist()
final_scores = scores.detach().cpu().tolist()
```

Acceptance:

```bash
grep -R "\.item()" -n utils/beam_search.py utils/generation_utils.py models lit_comer.py
grep -R "\.tolist()" -n utils/beam_search.py utils/generation_utils.py models lit_comer.py
```

Remaining `.item()` / `.tolist()` calls must be either outside generation hot loops or explicitly justified in comments.

## Step 8 — remove CUDA bool sync in generation loop

Problem: the current generation implementation may still have:

```python
if done_mask.all():
    break
```

If `done_mask` is a CUDA tensor, this forces CPU-GPU synchronization every decode step.

Actions:

1. Inspect `utils/generation_utils.py`.
2. Find any Python branch that depends on a CUDA tensor.
3. Remove per-step early break based on CUDA scalar if needed.
4. Prefer a fixed loop:

```python
for step in range(max_len):
    ...
```

5. Keep `done_mask` on GPU and use it only for tensor masking:

```python
scores = torch.where(done_mask.unsqueeze(-1), finished_scores, scores)
```

6. If early stopping is retained, make it explicit and low-frequency, with a comment explaining the sync tradeoff.

Acceptance:

- No `if done_mask.all():` inside the per-token decode loop.
- No Python `if`/`while` depends on CUDA tensors in generation hot path.
- Finished beams are handled with tensor masks, not Python branching per beam.

## Step 9 — avoid materialized encoder feature copies during beam expansion

Problem: patterns like this still physically copy encoder features:

```python
s = s.unsqueeze(1).expand(...).contiguous().view(...)
```

`expand()` is cheap, but `.contiguous()` materializes memory. Similarly, this copies bidirectional features:

```python
src[i] = torch.cat((src[i], src[i]), dim=0)
```

Actions:

1. Inspect:
   - `models/decoder.py`
   - `utils/generation_utils.py`
   - `models/comer.py`

2. Find beam expansion logic for encoder features and masks.

3. Avoid copying encoder features per beam when possible. Prefer:
   - keeping encoder features as `[B, ...]`,
   - keeping `beam_idx` / `batch_idx` mapping tensors,
   - gathering features only when required,
   - using views/expand without `.contiguous()` if downstream operations support it.

4. If a copy is temporarily unavoidable:
   - isolate it in one helper function,
   - add a TODO explaining why,
   - make sure it is not done repeatedly inside the decode loop.

5. Do not change image H/W semantics.

Acceptance:

- No repeated encoder feature materialization inside the token decode loop.
- Beam feature expansion, if still present, happens once and is clearly documented.
- `torch.cat((src[i], src[i]), dim=0)` is removed or isolated with justification.
- Memory behavior is documented in code comments.

## Step 10 — fix or remove `test_beam.py`

Problem: the previous review found that `python test_beam.py` failed with a tensor shape mismatch. A broken beam test is worse than no test because it gives false confidence.

Actions:

1. Run:

```bash
python test_beam.py
```

2. If the test is meant to validate the new beam path:
   - fix shape assumptions,
   - make it small and deterministic,
   - run on CPU by default,
   - optionally run on CUDA when available.

3. If it is only scratch/debug code:
   - rename it to something clearly non-test, or
   - move it under a debug folder, or
   - remove it if it is misleading.

Recommended test properties:

- No dataset dependency.
- Uses fake logits or a tiny dummy decoder.
- Verifies output shape.
- Verifies EOS handling.
- Verifies no crash on batch size > 1 and beam size > 1.

Acceptance:

```bash
python test_beam.py
```

passes, or the file is removed/renamed so it is no longer presented as a valid test.

## Step 11 — review ARM masked normalization path

Problem: `models/transformer/arm.py` may still default to legacy boolean indexing/scatter:

```python
flat_x = x[not_mask, :]
flat_x = self.bn(flat_x)
x[not_mask, :] = flat_x
```

This is GPU-unfriendly. However, changing normalization behavior can affect accuracy, so handle carefully.

Actions:

1. Inspect `models/transformer/arm.py`.
2. Check whether a vectorized implementation exists.
3. Check config value such as:

```yaml
arm_norm_impl: "legacy"
```

4. Do not blindly switch default behavior if numerical correctness is untested.
5. Add a safe config path:
   - `legacy` remains available,
   - `masked_vectorized` can be enabled,
   - the code path is clearly documented.

6. If switching default to `masked_vectorized`, add a quick numerical smoke test comparing output shape, finite values, and backward pass.

Acceptance:

- ARM normalization implementation choice is explicit and documented.
- Legacy path is not accidentally removed unless proven safe.
- If vectorized path is enabled by default, there is a smoke test for forward/backward.

## Step 12 — remove remaining global `shared_vocab` dependencies where practical

Problem: the old repo depended on:

```python
CROHMEDatamodule.shared_vocab
```

at import time. This is fragile for inference scripts and tests.

Actions:

1. Search:

```bash
grep -R "shared_vocab\|CROHMEDatamodule.shared_vocab" -n . --exclude-dir=.git
```

2. In model/generation/evaluation code, prefer passing explicit `vocab_info`, `sos_id`, `eos_id`, `pad_id`, or `vocab` objects.

3. If `shared_vocab` remains for backward compatibility:
   - keep it only inside the datamodule,
   - avoid reading it at import time in utility modules,
   - add a comment that it is a compatibility fallback.

Acceptance:

- No utility module reads `CROHMEDatamodule.shared_vocab` at import time.
- Inference/test scripts can instantiate config/datamodule/model in a predictable order.
- Token IDs used by generation are passed explicitly.

## Step 13 — re-check inference/test script

Problem: the updated test script may look better but still depends on DataModule and generation paths that were previously broken.

Actions:

1. Inspect:
   - `scripts/test/test.py`
   - any CLI or README command that runs inference.

2. Ensure it:
   - loads config consistently with training,
   - calls `dm.setup("test")`,
   - uses `model.eval()`,
   - uses `torch.inference_mode()` for inference-only execution,
   - does not import stale `comer.*` paths if the repo structure does not support them,
   - does not depend on `shared_vocab` import side effects.

3. Run an import-level smoke check if dataset/checkpoint is missing:

```bash
python - <<'PY'
import importlib
import scripts.test.test
print("test script import ok")
PY
```

Acceptance:

- Test script imports without stale package path errors.
- If dataset/checkpoint is present, test script can start and reach DataModule setup.
- It does not touch `DSP391m_Group1_FALL25_CoMER_old`.

## Step 14 — final verification pass

Run from `DSP391m_Group1_FALL25_CoMER`:

```bash
python -m compileall .
```

Run static search:

```bash
grep -R "empty_cache" -n . --exclude-dir=.git
grep -R "self.gpu_max_memory" -n . --exclude-dir=.git
grep -R "CROHMEDatamodule.shared_vocab" -n . --exclude-dir=.git
grep -R "\.item()" -n utils models lit_comer.py --exclude-dir=.git
grep -R "\.tolist()" -n utils models lit_comer.py --exclude-dir=.git
grep -R "done_mask.all()" -n . --exclude-dir=.git
grep -R "torch.cat((src\[i\], src\[i\])" -n . --exclude-dir=.git
```

If data is available, run:

```bash
python - <<'PY'
from omegaconf import OmegaConf
from datamodule import CROHMEDatamodule
from lit_comer import LitCoMER

config = OmegaConf.load("configs/crohme_config.yaml")
dm = CROHMEDatamodule(config)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
print("batch ok:", batch.imgs.shape, batch.mask.shape, batch.tgt.shape, batch.out.shape)

model = LitCoMER(config)
model.train()
out = model(batch.imgs, batch.mask, batch.tgt)
print("forward ok:", out.shape)
print("optim ok:", type(model.configure_optimizers()))
PY
```

If CUDA is available, add a one-batch GPU smoke test:

```python
batch = batch.to("cuda", non_blocking=True)
model = model.cuda()
out = model(batch.imgs, batch.mask, batch.tgt)
loss = out.float().mean()
loss.backward()
print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
```

Do not claim GPU optimization is complete unless CUDA smoke test runs.

## Final report required from Anti Gravity

After editing, return a concise report with:

1. Files changed.
2. Items completed from this README.
3. Commands run and results.
4. Items not completed and exact reason.
5. Any behavior changes that need user review.
6. Confirmation that `DSP391m_Group1_FALL25_CoMER_old` was not modified.

## Short Anti Gravity prompt

Copy this into Anti Gravity Agent:

```text
Read the instruction file `README.md` in this workspace and apply it only to `DSP391m_Group1_FALL25_CoMER`.

Important:
- `DSP391m_Group1_FALL25_CoMER` is the only folder you may edit.
- `DSP391m_Group1_FALL25_CoMER_old` is read-only reference only. Do not modify it.
- The user already fixed the first 5 blockers, so start from Step 6 smoke checks and continue through the remaining items.
- Do not implement AMP, deterministic/benchmark changes, framework upgrades, validation policy changes, or scheduler monitor policy changes.
- Preserve the image task semantics. Do not crop/distort/resize images beyond existing behavior; use padding/bucketing for VRAM stability.
- Work phase by phase, run the checks listed in README.md, and report files changed, commands run, incomplete items, and confirmation that the old folder was untouched.

Start by `cd DSP391m_Group1_FALL25_CoMER`, reading README.md fully, then run Step 6 checks before editing.
```
