# Agent harness: CoMER KV-cache inference

This harness is designed for an agentic coding session on a laptop/CPU-first environment. It compresses the repo/task context so the agent can implement the KV-cache feature without rereading a huge consolidated repo dump.

## Read order
1. `CLAUDE.md`
2. `agent_harness/context/00_task_brief.md`
3. `agent_harness/context/01_repo_map.md`
4. `agent_harness/context/02_implementation_contract.md`
5. `agent_harness/testing/TEST_MATRIX.md`
6. `agent_harness/context/04_target_snippets.md` only when editing the target files.
7. `agent_harness/reference/final_kv_cache_implementation_plan_comer.md` only when the contract seems ambiguous.

## Agent operating mode
- Optimize for correctness and behavior preservation first, speed second.
- Use small CPU tests and static guards because the machine is not a training server.
- Do not run heavy training or full CROHME evaluation unless the user explicitly provides compute/checkpoints.
- Keep old inference available through `use_kv_cache=False` and use it as the oracle.

## Deliverables expected from the coding agent
- New `DecoderKVCache` object with `expand_beam_()` and `reorder_()`.
- Incremental `transform_step` / decoder `forward_step` path.
- Cached self-attention and cached cross-attention helpers.
- ARM `forward_from_sums()` with fp32 sums and exact update order.
- Beam-search integration with SOS/EOS warm-up, beam expand/reorder, tree-state update order, and fallback.
- Unit/integration tests that compare cached path against full-prefix fallback.
- Optional benchmark script after exactness passes.
