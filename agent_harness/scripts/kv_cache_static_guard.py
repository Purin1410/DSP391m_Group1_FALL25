\
#!/usr/bin/env python3
"""Light static guard for the CoMER KV-cache task.

This script intentionally avoids importing torch or repo modules. It checks for
expected APIs and common forbidden shortcuts. Run from repo root:

    python agent_harness/scripts/kv_cache_static_guard.py --repo .
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REQUIRED_FILES = [
    "models/decoder.py",
    "models/transformer/attention.py",
    "models/transformer/arm.py",
    "models/transformer/transformer_decoder.py",
    "models/transformer/tree_bias.py",
    "utils/generation_utils.py",
]

REQUIRED_PATTERNS = {
    "models/transformer/kv_cache.py": [
        r"class\s+DecoderKVCache",
        r"def\s+expand_beam_\s*\(",
        r"def\s+reorder_\s*\(",
        r"beam_to_batch_idx",
        r"cross_pre_sum",
        r"cross_final_sum",
    ],
    "models/transformer/attention.py": [
        r"def\s+forward_cached_self\s*\(",
        r"def\s+forward_cached_cross\s*\(",
        r"def\s+project_static_kv\s*\(",
    ],
    "models/transformer/arm.py": [
        r"def\s+forward_from_sums\s*\(",
    ],
    "models/transformer/transformer_decoder.py": [
        r"def\s+forward_step\s*\(",
    ],
    "models/decoder.py": [
        r"def\s+init_decode_cache\s*\(",
        r"def\s+transform_step\s*\(",
    ],
    "utils/generation_utils.py": [
        r"use_kv_cache",
        r"use_tree_state_cache",
    ],
}

FORBIDDEN_PATTERNS = {
    "global": [
        r"scaled_dot_product_attention\s*\(",
        r"cross_coverage\s*=\s*False",
        r"self_coverage\s*=\s*False",
        r"use_tree_bias\s*=\s*False",
    ],
}

WARN_PATTERNS = {
    "utils/generation_utils.py": [
        (r"reorder_\s*\(\s*beam_idx\s*\)", "Expected KV-cache reorder by beam_idx after beam_scorer.process."),
    ],
    "models/transformer/arm.py": [
        (r"\.float\s*\(\s*\)", "Expected fp32 handling for ARM sums."),
    ],
}


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--final", action="store_true", help="enforce all final KV-cache API patterns")
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    errors = []
    warnings = []

    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            errors.append(f"Missing expected repo file: {rel}")

    if args.final:
        for rel, patterns in REQUIRED_PATTERNS.items():
            text = read(root / rel)
            if not text:
                errors.append(f"Missing file required for KV-cache task: {rel}")
                continue
            for pattern in patterns:
                if not re.search(pattern, text):
                    errors.append(f"{rel}: missing pattern {pattern}")
    else:
        print("Info: development mode. Use --final to enforce all required KV-cache API patterns.")

    all_text = "\n".join(read(root / rel) for rel in REQUIRED_FILES if (root / rel).exists())
    all_text += "\n" + read(root / "models/transformer/kv_cache.py")
    for pattern in FORBIDDEN_PATTERNS["global"]:
        if re.search(pattern, all_text):
            errors.append(f"Forbidden shortcut/API detected: {pattern}")

    for rel, checks in WARN_PATTERNS.items():
        text = read(root / rel)
        for pattern, message in checks:
            if text and not re.search(pattern, text):
                warnings.append(f"{rel}: {message}")

    for msg in warnings:
        print("WARN:", msg)
    for msg in errors:
        print("ERROR:", msg)

    if errors:
        print(f"Static guard failed with {len(errors)} error(s).")
        return 1
    print("Static guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
