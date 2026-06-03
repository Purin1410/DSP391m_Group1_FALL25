# Test matrix for KV-cache implementation

Run small CPU tests first. Do not require checkpoint/data for logic tests.

## Unit tests to add
1. `TreeRelativeBias`
   - Square `(B,L,L)` output exactly matches old implementation.
   - Row `(B,1,S)` returns `(B*H,1,S)` when `flatten=True`.

2. `DecoderKVCache`
   - `expand_beam_()` duplicates self K/V and ARM sums but not cross K/V.
   - `reorder_()` handles duplicate parent indices, e.g. `beam_idx=[2,0,1,2]`.
   - `beam_to_batch_idx` remains correct after expand/reorder.

3. Cross K/V projection
   - `project_static_kv(memory)` matches the K/V projection used by full MHA.
   - Key padding mask keeps padded source positions ignored.

4. Cached self-attention
   - Without ARM/tree-bias, incremental output at every t equals full-prefix last hidden/logits.
   - With tree-bias row, incremental output equals full-prefix last hidden/logits.

5. ARM sums
   - `forward_from_sums(...)` equals `full_arm(... )[:, -1:, :]`.
   - SOS position uses zero coverage and updates sums after cross attention.
   - fp32 sums are used even if model tensors are half precision.

## Integration tests
1. `Decoder.transform_step` exactness
   - Build a small random decoder in eval mode.
   - For t=1..L, compare `decoder.transform(... ids[:,:t])[:, -1, :]` vs incremental logits.
   - Cover ARM on/off and tree-bias on/off when feasible.

2. Beam exactness
   - `use_kv_cache=False` is oracle.
   - `use_kv_cache=True` returns identical sequences and close scores for `beam_size=1`, `beam_size=3`, and configured `beam_size=10` if runtime allows.
   - Test L2R first; bidirectional after L2R passes.

3. Fallback guard
   - In training mode or with grad enabled, cached path must not activate.

## Suggested commands
```bash
python agent_harness/scripts/kv_cache_static_guard.py --repo .
python -m pytest -q tests/test_kv_cache.py
python -m pytest -q tests/test_kv_cache.py -k "tree or cache or arm"
```

## Acceptance gate
- No behavior change for training/full-prefix path.
- Cached path exactness passes against fallback.
- Beam output exactness passes on lightweight CPU fixtures.
- No dependency on PyTorch 2.x APIs.
