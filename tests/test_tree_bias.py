import torch

from models.transformer.tree_bias import (
    TYPE_ROOT,
    TYPE_SUP,
    TreeRelationBuilder,
    TreeRelativeBias,
)

# Minimal vocab: id 0 is pad, rest are LaTeX-ish tokens for "x ^ { 2 }".
ID2TOK = {0: "<pad>", 1: "x", 2: "^", 3: "{", 4: "2", 5: "}"}
NUM_BUCKETS = 8
TYPE_SIZE = 5


def make_builder(**overrides):
    kwargs = dict(
        id2tok=ID2TOK,
        pad_id=0,
        num_buckets=NUM_BUCKETS,
        type_size=TYPE_SIZE,
        mode="full",
        rel_set="full",
    )
    kwargs.update(overrides)
    return TreeRelationBuilder(**kwargs)


def test_build_shape_and_range():
    builder = make_builder()
    tgt = torch.tensor([[1, 2, 3, 4, 5]])  # x ^ { 2 }
    rel_ids = builder.build(tgt)
    assert rel_ids.shape == (1, 5, 5)
    assert rel_ids.dtype == torch.long
    assert int(rel_ids.min()) >= 0
    assert int(rel_ids.max()) < builder.num_relations


def test_diagonal_is_always_zero_for_non_pad_tokens():
    # Any token's relation to itself has lcp == len(path), so distance == 0
    # and the "first differing type" lookup always lands on the ROOT sentinel.
    builder = make_builder()
    tgt = torch.tensor([[1, 2, 3, 4, 5]])
    rel_ids = builder.build(tgt)
    diag = torch.diagonal(rel_ids[0])
    assert torch.equal(diag, torch.zeros_like(diag))


def test_pad_pairs_are_zero():
    builder = make_builder()
    # "x ^ { 2 }" followed by two pad tokens.
    tgt = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
    rel_ids = builder.build(tgt)
    # Any row/col touching a pad position must be zero everywhere.
    assert torch.all(rel_ids[0, 5, :] == 0)
    assert torch.all(rel_ids[0, :, 5] == 0)
    assert torch.all(rel_ids[0, 6, :] == 0)
    assert torch.all(rel_ids[0, :, 6] == 0)


def test_relation_id_swaps_type_roles_when_transposed():
    """rel_id(i, j) and rel_id(j, i) share the same distance bucket but the
    (type_i, type_j) pair is swapped -- this is a directed encoding, not a
    symmetric one. Verified against a hand-traced example: token '2' sits
    inside a SUP context that token 'x' is not part of.
    """
    builder = make_builder()
    tgt = torch.tensor([[1, 2, 3, 4, 5]])  # x=0 ^=1 {=2 2=3 }=4
    rel_ids = builder.build(tgt)[0]

    x_idx, two_idx = 0, 3
    db = 1  # hand-traced: lcp=0, lens=(0,1) -> distance=1 -> bucket=1
    expected_two_to_x = db * (TYPE_SIZE * TYPE_SIZE) + TYPE_SUP * TYPE_SIZE + TYPE_ROOT
    expected_x_to_two = db * (TYPE_SIZE * TYPE_SIZE) + TYPE_ROOT * TYPE_SIZE + TYPE_SUP

    assert int(rel_ids[two_idx, x_idx]) == expected_two_to_x
    assert int(rel_ids[x_idx, two_idx]) == expected_x_to_two
    assert expected_two_to_x != expected_x_to_two


def test_build_is_batch_independent():
    """Relation ids for one sequence must not depend on what else is in the
    batch (padding to a common length must not leak across rows)."""
    builder = make_builder()
    solo = builder.build(torch.tensor([[1, 2, 3, 4, 5]]))

    batched = builder.build(
        torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [1, 1, 1, 1, 1],  # different, longer-looking content
            ]
        )
    )
    assert torch.equal(solo[0], batched[0])


def test_tree_relative_bias_zero_init_and_shape():
    num_heads, num_relations = 4, 8 * TYPE_SIZE * TYPE_SIZE
    bias_module = TreeRelativeBias(num_heads=num_heads, num_relations=num_relations)

    assert torch.equal(bias_module.emb.weight, torch.zeros_like(bias_module.emb.weight))

    rel_ids = torch.randint(0, num_relations, (2, 5, 5))
    out = bias_module(rel_ids, flatten=False)
    assert out.shape == (2, num_heads, 5, 5)

    out_flat = bias_module(rel_ids, flatten=True)
    assert out_flat.shape == (2 * num_heads, 5, 5)
    # zero-init means both forms must be exactly zero regardless of rel_ids.
    assert torch.equal(out_flat, torch.zeros_like(out_flat))


def test_tree_relative_bias_flatten_matches_manual_reshape():
    num_heads, num_relations = 3, 10
    bias_module = TreeRelativeBias(num_heads=num_heads, num_relations=num_relations)
    # Give the embedding non-zero, distinguishable values.
    with torch.no_grad():
        bias_module.emb.weight.copy_(
            torch.arange(num_relations * num_heads, dtype=torch.float32).view(
                num_relations, num_heads
            )
        )

    rel_ids = torch.randint(0, num_relations, (2, 4, 4))
    unflat = bias_module(rel_ids, flatten=False)  # (B, H, T, S)
    flat = bias_module(rel_ids, flatten=True)  # (B*H, T, S)

    manual = unflat.contiguous().view(2 * num_heads, 4, 4)
    assert torch.equal(flat, manual)
