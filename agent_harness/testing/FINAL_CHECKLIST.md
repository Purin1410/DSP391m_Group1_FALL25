# Final checklist before PR/merge

- [ ] Training loss path unchanged.
- [ ] Validation loss path unchanged.
- [ ] Full-prefix `Decoder.forward(...)` output unchanged.
- [ ] `_rate()` bidirectional rescoring remains full-prefix.
- [ ] `use_kv_cache=False` exactly reproduces old inference.
- [ ] `use_kv_cache=True` beam output matches fallback.
- [ ] ARM enabled exactness passes.
- [ ] Tree-bias enabled exactness passes.
- [ ] Bidirectional mode exactness passes or is explicitly marked as pending with reason.
- [ ] Beam duplicate-parent reorder test passes.
- [ ] SOS/EOS warm-up test passes.
- [ ] fp16 path keeps ARM sums in fp32.
- [ ] No PyTorch 2.x SDPA dependency.
- [ ] Config/kwargs flags documented.
- [ ] Benchmark reports latency and exactness after tests pass.
- [ ] `python agent_harness/scripts/kv_cache_static_guard.py --repo . --final` passes.
