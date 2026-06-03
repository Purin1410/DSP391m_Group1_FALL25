# CoMER KV-cache implementation rules

You are implementing inference-only KV-cache for this repository. Treat `agent_harness/context/02_implementation_contract.md` as the task contract and `agent_harness/testing/TEST_MATRIX.md` as the required validation loop.

## Non-negotiable constraints
- Do not change training, validation loss, target builders, or full-prefix `Decoder.forward(...)` semantics.
- Keep `_rate()` and bidirectional rescoring full-prefix unless explicitly required by the contract.
- Do not disable ARM, tree-bias, cross_coverage, self_coverage, or bidirectional support to make tests pass.
- Keep `MultiheadAttention.forward(...)` backward-compatible; add cached helper methods instead.
- KV-cache is active only under eval/no-grad inference guard and must have fallback `use_kv_cache=False`.
- Use fp32 running sums for ARM coverage even when model dtype is fp16.
- Do not use PyTorch 2.x SDPA in the main implementation because the repo is pinned to PyTorch 1.8.1 and ARM needs attention weights.

## Work loop
1. Inspect the real repo files before editing.
2. Implement one vertical slice at a time.
3. Add/adjust small CPU tests before or with each slice.
4. After every slice, run targeted tests and `python agent_harness/scripts/kv_cache_static_guard.py --repo .`.
5. Preserve exact outputs: cached logits/beam hypotheses must match fallback within tight tolerance.
6. Prefer small, reviewable changes over a large rewrite.

## Primary target files
- `models/transformer/tree_bias.py`
- `models/transformer/kv_cache.py` (new)
- `models/transformer/attention.py`
- `models/transformer/arm.py`
- `models/transformer/transformer_decoder.py`
- `models/decoder.py`
- `utils/generation_utils.py`
- tests / benchmark script as described in the harness.
