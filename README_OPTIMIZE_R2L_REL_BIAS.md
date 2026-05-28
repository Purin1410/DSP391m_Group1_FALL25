# Optimize R2L tree relative bias for bidirectional CoMER

## Goal

Fix the CPU bottleneck caused by `CausalR2LTreeRelationBuilder` during validation sanity check and normal DataLoader collation.

Current code is logically close, but too slow because it:

1. reparses `tokens[:i+1]` from scratch for every row `i`,
2. builds `all_step_paths`,
3. allocates 4D CPU tensors `P_cpu/A_cpu` with shape `(B, L, L, D)`,
4. then computes relation ids from those tensors.

Replace it with a one-pass incremental causal builder that directly fills `rel_ids` row by row.

## Important dataset fact

The dictionary contains both visible braces and layout braces as different tokens:

```text
\{
\}
{
}
```

Therefore:

- Only exact `{` and `}` are layout delimiters.
- `\{` and `\}` are visible symbols and must be treated as normal content tokens.
- Do not use `\lbrace` or `\rbrace` as delimiters unless they are explicitly present in the target dictionary. For this dataset, they are not needed.

## Files to change

Primary file:

```text
models/transformer/tree_bias.py
```

Likely no required changes elsewhere if the public API stays the same:

```python
CausalR2LTreeRelationBuilder(...).build(tgt_ids) -> LongTensor[B, L, L]
```

Optional dev-only change:

```text
configs/crohme_config.yaml
train.py
```

Only add `num_sanity_val_steps: 0` as a temporary debugging knob. Do not use it as the real fix.

## Required design

Keep these constants:

```python
TYPE_ROOT = 0
TYPE_SUP  = 1
TYPE_SUB  = 2
TYPE_NUM  = 3
TYPE_DEN  = 4
TYPE_UNK  = 5
```

Keep shared bidirectional relation vocabulary with `type_size=6`.

Implement R2L causal build as:

```text
for each sequence:
  initialize parser state once
  initialize rel[L, L] = 0
  for row i in 0..L-1:
    process token i only
    materialize current paths visible at row i
    fill rel[i, 0:i+1]
  return rel
```

Never rebuild prefix `0..i` from scratch.
Never allocate `(B, L, L, D)` tensors.
Only allocate final `rel_ids` shaped `(B, L, L)`.

## One-pass state machine

Use mutable context nodes so resolving a previously unknown group updates its path for current and future rows, but previously computed rows stay frozen in `rel_ids`.

Suggested structures:

```python
@dataclass
class CtxNode:
    type: int

@dataclass
class Operand:
    kind: str                 # "atom" or "group"
    token_indices: List[int]  # for atom
    node: Optional[CtxNode]   # for group
    parent_nodes: List[CtxNode]

@dataclass
class Frame:
    node: Optional[CtxNode]
    operands: List[Operand]
```

Parser state:

```python
frames = [Frame(node=None, operands=[])]
path_refs = [[] for _ in range(L)]
rel = torch.zeros((L, L), dtype=torch.long)
```

Helpers:

```python
def current_path_nodes():
    return [fr.node for fr in frames if fr.node is not None]

def materialize(nodes):
    types = [n.type for n in nodes]
    return (TYPE_ROOT,) if len(types) == 0 else tuple(types)
```

Token handling:

```text
PAD:
  path_refs[i] = []
  do not push operand

SOS/EOS boundary tokens:
  path_refs[i] = current_path_nodes()
  do not push operand

"}":
  path_refs[i] = current_path_nodes()
  push new Frame(node=CtxNode(TYPE_UNK), operands=[])

"{":
  path_refs[i] = current_path_nodes()
  pop current frame if any group is open
  append Operand(kind="group", node=closed.node, parent_nodes=current_path_nodes()) to parent frame

"^" or "_":
  path_refs[i] = current_path_nodes()
  resolve last operand in current frame to SUP or SUB

"\\frac", "\\dfrac", "\\tfrac":
  path_refs[i] = current_path_nodes()
  resolve last operand as NUM
  resolve previous operand as DEN

normal token, including "\\{" and "\\}":
  path_refs[i] = current_path_nodes()
  append atom operand with token_indices=[i]
```

For `\frac { a } { b }`, R2L target content is:

```text
} b { } a { \frac
```

At `\frac`, the last operand is numerator `a`, previous operand is denominator `b`.

## Relation id computation

Use the same formula as `TreeRelationBuilder`, but scalar and cached:

```python
def pair_to_rel_id(pi, pj):
    pi = () if pi == (TYPE_ROOT,) else pi
    pj = () if pj == (TYPE_ROOT,) else pj
    lcp = longest_common_prefix(pi, pj)
    d = len(pi) + len(pj) - 2 * lcp
    db = min(max(d, 0), num_buckets - 1)
    ti = pi[lcp] if lcp < len(pi) else TYPE_ROOT
    tj = pj[lcp] if lcp < len(pj) else TYPE_ROOT

    # Apply rel_set remap exactly like existing code.
    # For R2L, keep UNK in script/fraction/core modes.

    if mode == "dist_only": return db
    if mode == "type_only": return ti * type_size + tj
    return db * (type_size * type_size) + ti * type_size + tj
```

Add a small dictionary cache inside each sequence build:

```python
rel_cache = {}
rid = rel_cache.get((pi, pj))
```

This avoids recomputing LCP for repeated path pairs.

## Batch build

Implement `build(self, tgt_ids)` as:

```python
B, L = tgt_ids.shape
tgt_cpu = tgt_ids.detach().cpu()
out = torch.zeros((B, L, L), dtype=torch.long)
for b in range(B):
    out[b] = self._build_one(tgt_cpu[b])
return out.to(tgt_ids.device, non_blocking=True) if needed
```

Keep causal upper triangle zero. Existing decoder mask will block it, but tests should still require `rel[i, j] == 0` for `j > i`.

## Boundary tokens

The builder currently receives only `pad_id`. Add robust boundary detection in `__init__`:

```python
self.boundary_ids = self._find_all({"<sos>", "<eos>"})
```

Then do not push SOS/EOS as operands. This prevents boundary tokens from being accidentally consumed by an unbraced `^`, `_`, or `\frac` fallback.

## Required tests

Create or extend `test_tree_bias.py`.

Path-level tests should use the real dictionary-like tokens:

```python
id2tok = {
  0: "<pad>", 1: "<sos>", 2: "<eos>",
  3: "x", 4: "^", 5: "{", 6: "}", 7: "2",
  8: "a", 9: "b", 10: "_", 11: "i", 12: "\\frac",
  13: "\\{", 14: "\\}"
}
```

Must pass these expected row paths:

```text
R2L x ^ { 2 }:
  tokens: } 2 { ^ x
  rows:
    R
    R U
    R U U
    R S S R
    R S S R R

R2L \frac { a } { b }:
  tokens: } b { } a { \frac
  rows:
    R
    R U
    R U U
    R U U R
    R U U R U
    R U U R U U
    R D D R N N R

R2L x ^ { a _ { i } }:
  after "_", token i must be U/B
  after "^", token i must be S/B

Escaped braces:
  tokens: \} b \{ ^ x
  \} and \{ must be normal atoms, not group delimiters
```

Relation-id tests:

```text
rel.shape == (B, L, L)
rel.max() < builder.num_relations
rel[:, i, j] == 0 for j > i
PAD rows/columns are zero
No row uses future marker resolution
```

Integration tests:

```text
use_bidirectional=true and use_tree_bias=true
collate_fn returns rel_ids with shape [2B, L, L]
first B rows are L2R relation ids
last B rows are R2L causal relation ids
Decoder forward passes with precomputed rel_ids
Decoder fallback also passes when rel_ids=None
```

## Performance gate

Add a microbenchmark test or script, but do not make it too hardware-specific.

Suggested script:

```python
import time, torch
from models.transformer.tree_bias import CausalR2LTreeRelationBuilder

B, L = 4, 200
# Build synthetic legal-ish ids containing braces/scripts/fracs and pads.
# Warm up once, then average 10 calls.
```

The gate is qualitative:

```text
The optimized builder must not allocate P_cpu/A_cpu with 4D shape.
The optimized builder must be clearly faster than the old prefix-reparse implementation.
The sanity validation should start within seconds, not minutes, on eval batch size 4.
```

Use Lightning or PyTorch profiler to confirm `collate_fn` or `CausalR2LTreeRelationBuilder.build` is no longer the dominant CPU bottleneck.

## Acceptance checklist

- [ ] No prefix reparse loop `for i: for pos in range(i + 1)` remains in R2L builder.
- [ ] No `P_cpu = torch.full((B, L, L, D), ...)` remains in R2L builder.
- [ ] Only final `rel_ids` shaped `(B, L, L)` is allocated.
- [ ] Exact `{` and `}` are the only layout delimiters.
- [ ] `\{` and `\}` are normal visible tokens.
- [ ] SOS/EOS/PAD are not operands.
- [ ] Expected path tests pass.
- [ ] Relation-id invariant tests pass.
- [ ] Bidirectional decoder forward passes.
- [ ] Validation sanity check no longer hangs on CPU relation building.
