# R2L Causal Tree Relative Bias Upgrade Guide

This README is a compact implementation plan for upgrading the repository so `use_bidirectional=true` and `use_tree_bias=true` can be enabled together without structural leakage.

## Goal

Support tree relative bias for both decoder directions:

- L2R branch: keep the existing tree relation behavior.
- R2L branch: build causal, prefix-safe tree relations where row `i` can only use the R2L prefix `tokens[0..i]`.
- Bidirectional training target layout remains `[L2R batch; R2L batch]`.
- `rel_ids` shape must match `tgt`: `[B, L, L]` in L2R mode and `[2B, L, L]` in bidirectional mode.

The key correctness rule is: never let an R2L row use a future marker such as `^`, `_`, or `\frac` before that marker is visible in the R2L prefix.

## Current repository facts

Relevant files:

| File | Current role | Required change |
|---|---|---|
| `configs/crohme_config.yaml` | Config says tree bias is L2R-only when bidirectional is enabled | Update comment and allow the combined mode after code is fixed |
| `datamodule/datamodule.py` | Builds `tgt,out`, then precomputes `rel_ids` in the DataLoader | Build L2R and R2L relation IDs separately when bidirectional |
| `models/comer.py` | Duplicates encoder features when `use_bidirectional=true` | Remove the guard that rejects tree bias with bidirectional |
| `models/decoder.py` | Creates `TreeRelationBuilder` and `TreeRelativeBias`; fallback builds rel_ids if dataloader did not pass them | Use direction-aware builders and relation-id space large enough for `UNK` |
| `models/transformer/tree_bias.py` | Contains current L2R builder and relation embedding | Add R2L causal builder and shared relation constants |
| `utils/utils.py` | Builds L2R and bidirectional targets | Do not change target layout unless tests prove it is necessary |
| `test_beam.py` or `tests/` | Existing shape sanity tests | Add deterministic unit tests for R2L tree bias |

Current target construction already places first `B` rows as L2R and last `B` rows as R2L. R2L uses `<eos>` as start token and reversed labels after it. Keep this contract.

## Core design decision

Add an explicit `UNK` context type for R2L only:

```python
TYPE_ROOT = 0
TYPE_SUP  = 1
TYPE_SUB  = 2
TYPE_NUM  = 3
TYPE_DEN  = 4
TYPE_UNK  = 5
TYPE_SIZE_R2L = 6
```

Important: when bidirectional tree bias is enabled, both L2R and R2L relation IDs must live in the same embedding table. Therefore instantiate L2R with `type_size=6` in bidirectional mode, even though L2R never emits `TYPE_UNK`. If L2R uses `type_size=5` and R2L uses `type_size=6`, `TreeRelativeBias.num_relations` can be too small or IDs can be inconsistent.

Safe rule:

```python
relation_type_size = 6 if use_bidirectional else 5
num_relations = num_buckets * relation_type_size * relation_type_size
```

For `dist_only`, `num_relations = num_buckets`. For `type_only`, `num_relations = relation_type_size * relation_type_size`.

## Token semantics

Use exact layout tokens only:

- Layout left brace: `{`
- Layout right brace: `}`
- Literal visible left brace: `\{` is a normal content token
- Literal visible right brace: `\}` is a normal content token

Do not treat `\{`, `\}`, `\lbrace`, or `\rbrace` as layout delimiters unless the dataset explicitly documents that they are structural delimiters. For this project, the user requirement is that escaped braces are visually real symbols and must not open or close layout groups.

The dataset is standardized to use grouped scripts such as:

```text
x ^ { 2 }
```

Do not optimize for unbraced `x ^ 2` as the main path. You may preserve fallback handling, but every QA test must prioritize braced ground-truth LaTeX.

## R2L causal parser specification

Input R2L example for `x ^ { 2 }`:

```text
["}", "2", "{", "^", "x"]
```

The parser must produce row-wise visible paths, not one fixed path per token. At row `i`, only `tokens[0..i]` are known.

Required row-wise labels:

```text
row0: R
row1: R U
row2: R U U
row3: R S S R
row4: R S S R R
```

Meaning: token `2` is `U` before `^` is visible, then becomes `S` when `^` is visible. Earlier rows must not be retroactively changed in `rel_ids`.

### Data structures

Implement as small internal dataclasses in `tree_bias.py`:

```python
@dataclass
class _R2LCtxNode:
    type: int

@dataclass
class _R2LOperand:
    kind: str  # "atom" or "group"
    token_indices: List[int]
    node: Optional[_R2LCtxNode]
    parent_nodes: List[_R2LCtxNode]

@dataclass
class _R2LFrame:
    node: Optional[_R2LCtxNode]
    operands: List[_R2LOperand]
```

The parser state is:

```python
frames = [_R2LFrame(node=None, operands=[])]
path_refs = [None] * L
row_paths = []
```

### R2L transition rules

For each token `tok` at position `i`:

1. `tok == "}"`
   - Assign current parent path to `path_refs[i]`.
   - Push a new frame with `node=UNK` because tokens after this are inside an unresolved group.

2. `tok == "{"`
   - Assign current path to `path_refs[i]` before popping.
   - Pop the current group frame and append it as a `group` operand to the parent frame.

3. `tok in {"^", "_"}`
   - Assign current parent path to marker.
   - Resolve the most recent operand in the current frame to `SUP` for `^` or `SUB` for `_`.

4. `tok in {"\\frac", "\\dfrac", "\\tfrac"}`
   - Assign current parent path to marker.
   - In R2L order for `\frac { a } { b }`, the nearest operand is numerator and the previous operand is denominator.
   - Pop nearest as `NUM`, then previous as `DEN`.

5. `tok == pad_id`
   - Mark as root or empty and do not create operands.

6. `tok in {sos_id, eos_id}`
   - Mark as current parent path, normally root at sequence start.
   - Do not create operands.

7. Otherwise
   - Assign current path.
   - Append as an `atom` operand.

After processing the token and any resolution, snapshot paths for `j <= i` only. This snapshot is what row `i` uses to build `rel_ids[i, j]`.

## Relation ID conversion

Use the same relation encoding as current L2R builder, but with `type_size=6` for bidirectional mode.

For a pair of paths at row `i`:

```python
pi = [] if path_i == [TYPE_ROOT] else path_i
pj = [] if path_j == [TYPE_ROOT] else path_j
lcp = longest_common_prefix(pi, pj)
d = len(pi) + len(pj) - 2 * lcp
db = min(max(d, 0), num_buckets - 1)
ti = pi[lcp] if lcp < len(pi) else TYPE_ROOT
tj = pj[lcp] if lcp < len(pj) else TYPE_ROOT
```

Then:

```python
if mode == "dist_only":
    rid = db
elif mode == "type_only":
    rid = ti * type_size + tj
else:
    rid = db * (type_size * type_size) + ti * type_size + tj
```

Set `rel_ids[i, j] = 0` for `j > i`. The causal mask will block those positions, but zeroing them makes debugging and tests deterministic.

## Implementation steps

### Step 1: Refactor constants in `tree_bias.py`

Add:

```python
TYPE_UNK = 5
TYPE_SIZE_L2R = 5
TYPE_SIZE_BIDIR = 6
```

Keep old IDs stable. Do not renumber existing relation types.

### Step 2: Add `CausalR2LTreeRelationBuilder`

Add a class next to `TreeRelationBuilder`:

```python
class CausalR2LTreeRelationBuilder(TreeRelationBuilder):
    def __init__(self, *args, **kwargs):
        kwargs["type_size"] = kwargs.get("type_size", TYPE_SIZE_BIDIR)
        super().__init__(*args, **kwargs)

    def build(self, tgt_ids: torch.LongTensor) -> torch.LongTensor:
        ...
```

Implementation notes:

- Reuse `self.id2tok`, `self.sup_ids`, `self.sub_ids`, `self.frac_ids`, `self.lbrace_ids`, `self.rbrace_ids`, `self.pad_id`, `self.num_buckets`, `self.mode`, and `self.rel_set`.
- Convert IDs to tokens through `self.id2tok[tid]` only after checking `tid` bounds.
- Keep the sequential parser on CPU like current L2R builder.
- Return tensor on the original `tgt_ids.device`.
- Mask PAD pairs to 0 exactly as current builder does.
- Do not treat escaped braces as delimiters.

### Step 3: Make decoder relation building direction-aware

In `models/decoder.py`, replace the single builder with helper methods:

```python
self._tree_builder_l2r = TreeRelationBuilder(..., type_size=relation_type_size)
self._tree_builder_r2l = CausalR2LTreeRelationBuilder(..., type_size=relation_type_size)
self._tree_rel_bias = TreeRelativeBias(num_heads=nhead, num_relations=self._tree_builder_l2r.num_relations)
```

Add:

```python
def _build_rel_ids_for_tgt(self, tgt):
    if not self.use_bidirectional:
        return self._tree_builder_l2r.build(tgt)
    if tgt.size(0) % 2 != 0:
        raise ValueError("Bidirectional tree bias expects even batch dimension [L2R; R2L].")
    half = tgt.size(0) // 2
    rel_l2r = self._tree_builder_l2r.build(tgt[:half])
    rel_r2l = self._tree_builder_r2l.build(tgt[half:])
    return torch.cat((rel_l2r, rel_r2l), dim=0)
```

Use this helper whenever `rel_ids is None` in `forward`. This is required for beam search and reverse scoring, because generation code calls decoder without dataloader-provided `rel_ids`.

### Step 4: Update `datamodule/datamodule.py`

Remove the guard that raises when both flags are true. Instantiate builders:

```python
if self.use_tree_bias:
    relation_type_size = 6 if self.use_bidirectional else 5
    self.tree_builder_l2r = TreeRelationBuilder(..., type_size=relation_type_size)
    self.tree_builder_r2l = CausalR2LTreeRelationBuilder(..., type_size=relation_type_size)
else:
    self.tree_builder_l2r = None
    self.tree_builder_r2l = None
```

In `collate_fn`:

```python
rel_ids = None
if self.use_tree_bias:
    if self.use_bidirectional:
        B = labels.size(0)
        rel_l2r = self.tree_builder_l2r.build(tgt[:B])
        rel_r2l = self.tree_builder_r2l.build(tgt[B:])
        rel_ids = torch.cat((rel_l2r, rel_r2l), dim=0)
    else:
        rel_ids = self.tree_builder_l2r.build(tgt)
```

Do not build R2L relations from `labels` directly unless you exactly mirror the `r2l_tgt` sequence including `<eos>` at position 0. Safer: split the final `tgt`.

### Step 5: Update `models/comer.py`

Remove the guard that rejects tree bias in bidirectional mode. Keep encoder duplication unchanged:

```python
if self.use_bidirectional:
    feature = torch.cat((feature, feature), dim=0)
    mask = torch.cat((mask, mask), dim=0)
```

### Step 6: Update config comments

Change comments from "tree bias is L2R-only" to:

```yaml
# Tree relative bias supports L2R and bidirectional modes.
# In bidirectional mode, the R2L branch uses causal prefix-safe relation IDs with UNK contexts.
```

## Required sanity cases

The tests must use a tiny fake vocab. Include normal layout braces and escaped visible braces as distinct tokens.

### Case A: superscript

Original:

```text
x ^ { 2 }
```

R2L target content, ignoring special start token:

```text
} 2 { ^ x
```

Expected visible paths:

```text
row0: R
row1: R U
row2: R U U
row3: R S S R
row4: R S S R R
```

### Case B: nested subscript inside superscript

Original:

```text
x ^ { a _ { i } }
```

R2L content:

```text
} } i { _ a { ^ x
```

Expected high-level behavior:

- First `}` is root-level boundary, so row0 is `R`.
- Second `}` is inside unresolved outer group, so it is `U`, not root.
- When `_` appears, the inner group resolves to `SUB` while the outer group remains `UNK`.
- When `^` appears, the outer group resolves to `SUP`, and the inner group path becomes `SUP/SUB`.

### Case C: fraction

Original:

```text
\frac { a } { b }
```

R2L content:

```text
} b { } a { \frac
```

Expected visible paths:

```text
row0: R
row1: R U
row2: R U U
row3: R U U R
row4: R U U R U
row5: R U U R U U
row6: R D D R N N R
```

### Case D: escaped braces

Original content tokens:

```text
x + \{ y \}
```

Expected:

- `\{` and `\}` are atoms.
- They must not push or pop a group frame.
- They can become operands only if a real future marker uses them.

### Case E: special tokens and PAD

For R2L `tgt`:

```text
<eos> } 2 { ^ x <pad> <pad>
```

Expected:

- `<eos>` gets root and creates no operand.
- PAD rows and PAD columns in `rel_ids` are 0.
- No PAD token affects grouping or operand stacks.

## QA and QC checklist

Before considering the task done, run or implement checks for all items below.

### Structural correctness

- [ ] `use_bidirectional=true` and `use_tree_bias=true` no longer raises in datamodule, CoMER, or Decoder.
- [ ] L2R-only mode still works with existing `TreeRelationBuilder` behavior.
- [ ] Bidirectional mode returns `rel_ids.shape == tgt.shape + (tgt.size(1),)`.
- [ ] `rel_ids[:B]` are L2R relations and `rel_ids[B:]` are R2L causal relations.
- [ ] R2L rows never use future marker resolution before the marker appears.
- [ ] `TYPE_UNK` appears only when causal ambiguity exists in R2L.
- [ ] Literal escaped braces are never treated as layout braces.

### Relation ID validity

- [ ] `rel_ids.min() >= 0`.
- [ ] `rel_ids.max() < TreeRelativeBias.num_relations` for every mode.
- [ ] `mode in {dist_only, type_only, full}` works.
- [ ] `rel_set in {script, fraction, core}` works with `TYPE_UNK`; for ablation remaps, keep or drop `UNK` intentionally and document the choice.

Recommended rel-set rule with `UNK`:

- `core`: keep `UNK`.
- `script`: keep `ROOT, SUP, SUB, UNK`; map `NUM, DEN` to `ROOT`.
- `fraction`: keep `ROOT, NUM, DEN, UNK`; map `SUP, SUB` to `ROOT`.

### Training and inference integration

- [ ] Dataloader precomputes CPU `rel_ids` and `Batch.to()` moves it to device.
- [ ] Decoder fallback builds direction-aware `rel_ids` when `rel_ids=None`.
- [ ] Beam search does not crash in bidirectional mode with tree bias enabled.
- [ ] `_rate()` reverse scoring works because decoder fallback can build bidirectional `rel_ids` from `tgt`.
- [ ] No device mismatch between `rel_ids`, relation embedding output, and attention logits.

### Regression safety

- [ ] Existing `test_beam.py` passes.
- [ ] New R2L tree-bias unit tests pass without requiring CROHME data.
- [ ] A one-batch forward pass passes for:
  - `use_bidirectional=false, use_tree_bias=true`
  - `use_bidirectional=true, use_tree_bias=false`
  - `use_bidirectional=true, use_tree_bias=true`
- [ ] If checkpoint compatibility matters, document that changing relation vocabulary from 5 to 6 in bidirectional mode changes tree-bias embedding shape.

## Suggested test file

Create `tests/test_tree_bias_r2l.py` or append to `test_beam.py` if the repo does not use pytest.

Minimum tests:

1. `test_r2l_x_sup_2_no_future_leakage()`
2. `test_r2l_nested_sup_sub_right_brace_not_always_root()`
3. `test_r2l_fraction_num_den_order()`
4. `test_escaped_braces_are_atoms()`
5. `test_bidirectional_rel_ids_shape_and_max_id()`
6. `test_decoder_fallback_rel_ids_bidir()`

A direct path-label helper is acceptable for tests, for example `_debug_build_visible_paths(tokens)`, but keep it private or test-only if you do not want it in production API.

## Expected patch summary

A good final patch should say:

```text
Implemented causal R2L tree relative bias with UNK contexts.
Bidirectional training now builds relation IDs as [L2R rel_ids; R2L rel_ids].
Decoder fallback also builds direction-aware rel_ids for beam search and reverse scoring.
Added unit tests for superscript, nested script, fraction, escaped braces, PAD, and ID bounds.
```

## Main risks

1. Relation vocabulary mismatch: L2R uses 5 types but R2L emits 6 types. Fix by using `type_size=6` for both builders in bidirectional mode.
2. Hidden inference breakage: dataloader precompute may pass training, but beam search calls decoder without `rel_ids`. Fix decoder fallback.
3. Escaped brace bug: `\{` and `\}` must be visible symbols, not layout delimiters.
4. Leakage bug: do not compute a final full R2L tree once and reuse it for all rows. Build row-wise snapshots.
5. Special-token operand bug: `<eos>` at the beginning of R2L target must not become an operand.
