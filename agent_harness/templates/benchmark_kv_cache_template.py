\
#!/usr/bin/env python3
"""Template for a lightweight KV-cache benchmark.

Place/adapt as `scripts/benchmark_kv_cache.py` after exactness tests pass.
"""
import argparse
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-exact", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--max-len", type=int, default=40)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default="configs/crohme_config.yaml")
    args = p.parse_args()
    print("TODO: instantiate model/checkpoint or tiny fixture and compare:")
    print("mode latency_ms speedup peak_mem_MB exact_match")
    print("full_prefix ...")
    print("full_kv_cache_with_arm ...")


if __name__ == "__main__":
    main()
