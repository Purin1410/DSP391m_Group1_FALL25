# 03 External research notes

These notes are distilled for implementation decisions, not for citation-heavy reading during coding.

- Context engineering for agents: keep the loaded context small, explicit, and task-specific. Put stable rules in `CLAUDE.md`; put deeper task details in small linked files. Avoid making the agent read a huge repo dump unless a specific ambiguity requires it.
- Claude Code best-practice implication: use repository instructions, concrete commands, a phased plan, and small verification loops. Context window is a scarce resource; do not dump verbose command output back into the prompt.
- Claude memory implication: `CLAUDE.md` is project-level context, not hard enforcement. Tests and static guard scripts are the enforcement layer.
- KV-cache implementation implication: static cache fits this repo because `max_len` is fixed. Cross K/V can be precomputed once per decoder layer; self K/V grows one token per step.
- PyTorch implication: the repo is pinned to PyTorch 1.8.1. Newer `scaled_dot_product_attention`/fastpath docs are useful background only, not a dependency. Also, ARM needs attention weights, so any optimized path that hides weights is not a drop-in replacement.

Useful public references:
- Anthropic Claude Code best practices: https://code.claude.com/docs/en/best-practices
- Anthropic context engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Claude project memory / CLAUDE.md: https://code.claude.com/docs/en/memory
- Hugging Face KV cache strategies: https://huggingface.co/docs/transformers/kv_cache
- PyTorch MultiheadAttention docs: https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html
