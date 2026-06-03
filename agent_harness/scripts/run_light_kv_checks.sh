\
#!/usr/bin/env bash
set -euo pipefail
python agent_harness/scripts/kv_cache_static_guard.py --repo .
if [ -f tests/test_kv_cache.py ]; then
  python -m pytest -q tests/test_kv_cache.py
else
  echo "tests/test_kv_cache.py not found yet; copy/adapt template before full check."
fi
