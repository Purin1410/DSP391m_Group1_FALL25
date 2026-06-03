# Light test loop for a laptop

Use this loop after every vertical slice:

1. Static sanity
```bash
python agent_harness/scripts/kv_cache_static_guard.py --repo .
```

2. Narrow unit tests
```bash
python -m pytest -q tests/test_kv_cache.py -k "tree or DecoderKVCache or reorder"
```

3. Incremental exactness tests
```bash
python -m pytest -q tests/test_kv_cache.py -k "transform_step or self_attention or arm"
```

4. Beam exactness smoke
```bash
python -m pytest -q tests/test_kv_cache.py -k "beam and not slow"
```

5. Optional benchmark only after exactness
```bash
python scripts/benchmark_kv_cache.py --check-exact --batch-size 1 --beam-size 3 --max-len 40 --warmup 1 --iters 3
```

Keep stdout small. If a failure occurs, inspect the exact mismatching tensor position and update the smallest relevant slice.

Final static gate before PR:
```bash
python agent_harness/scripts/kv_cache_static_guard.py --repo . --final
```
