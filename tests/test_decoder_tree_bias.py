import torch

from models.decoder import Decoder
from utils.bidirectional import BidirectionalLayout
from utils.vocab_info import VocabInfo

ID2TOK = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "x", 4: "^", 5: "{", 6: "2", 7: "}"}
VOCAB_SIZE = len(ID2TOK)


class _FakeWords:
    def __init__(self, idx2word):
        self.idx2word = idx2word


def _vocab_info() -> VocabInfo:
    return VocabInfo(
        vocab_size=VOCAB_SIZE,
        sos_id=1,
        eos_id=2,
        pad_id=0,
        words=_FakeWords(ID2TOK),
    )


def _make_decoder(use_tree_bias: bool, seed: int = 0, **overrides) -> Decoder:
    torch.manual_seed(seed)
    kwargs = dict(
        d_model=8,
        nhead=2,
        num_decoder_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        dc=4,
        cross_coverage=False,
        self_coverage=False,
        vocab_info=_vocab_info(),
        use_tree_bias=use_tree_bias,
        tree_bias_num_buckets=8,
        tree_bias_mode="full",
        tree_bias_layers="all",
        tree_bias_rel_set="full",
    )
    kwargs.update(overrides)
    return Decoder(**kwargs)


def _make_inputs(total_batch=4, l=5, h=3, w=3, d=8):
    src = torch.randn(total_batch, h, w, d)
    src_mask = torch.zeros(total_batch, h, w, dtype=torch.bool)
    # Content tokens only (avoid pad/sos/eos) so the parser exercises SUP context.
    tgt = torch.randint(3, VOCAB_SIZE, (total_batch, l))
    return src, src_mask, tgt


def test_decoder_builds_and_forwards_with_tree_bias_enabled():
    dec = _make_decoder(use_tree_bias=True)
    dec.eval()
    src, src_mask, tgt = _make_inputs()
    out = dec(src, src_mask, tgt)
    assert out.shape == (4, 5, VOCAB_SIZE)
    assert torch.isfinite(out).all()


def test_decoder_use_tree_bias_false_matches_true_zero_init():
    """Zero-initialized tree bias must be a complete no-op: a Decoder built
    with use_tree_bias=True (fresh, untrained) must produce byte-identical
    logits to one built with use_tree_bias=False, given identical seeds for
    every *shared* parameter (the tree-bias embedding is the only thing
    constructed after all shared modules, so resetting the seed before each
    build keeps every shared weight draw in lockstep)."""
    src, src_mask, tgt = _make_inputs()

    dec_false = _make_decoder(use_tree_bias=False, seed=42)
    dec_false.eval()
    out_false = dec_false(src.clone(), src_mask.clone(), tgt.clone())

    dec_true = _make_decoder(use_tree_bias=True, seed=42)
    dec_true.eval()
    out_true = dec_true(src.clone(), src_mask.clone(), tgt.clone())

    assert torch.allclose(out_false, out_true, atol=1e-6)


def test_r2l_bias_is_exactly_zero_even_with_nonzero_embedding():
    """The R2L half of rel_bias must be structurally zero -- not zero because
    the embedding happens to be zero-initialized. Overwrite the embedding
    with values that are nonzero at every relation id (including id 0) and
    confirm the R2L block is still an exact zero tensor."""
    dec = _make_decoder(use_tree_bias=True, seed=0)
    with torch.no_grad():
        dec._tree_rel_bias.emb.weight.copy_(
            torch.randn_like(dec._tree_rel_bias.emb.weight) + 1.0  # guaranteed != 0
        )

    _, _, tgt = _make_inputs()
    rel_bias = dec._build_rel_bias_for_tgt(tgt)
    assert rel_bias is not None

    total = tgt.shape[0]
    l2r_slice, _ = BidirectionalLayout.split(total)
    num_heads = dec._tree_rel_bias.num_heads
    l2r_bias = rel_bias[: l2r_slice.stop * num_heads]
    r2l_bias = rel_bias[l2r_slice.stop * num_heads :]

    assert torch.equal(r2l_bias, torch.zeros_like(r2l_bias))
    assert not torch.equal(l2r_bias, torch.zeros_like(l2r_bias))


def test_gradient_flows_only_from_l2r_rows_into_tree_bias_embedding():
    dec = _make_decoder(use_tree_bias=True, seed=0)
    with torch.no_grad():
        dec._tree_rel_bias.emb.weight.copy_(
            torch.randn_like(dec._tree_rel_bias.emb.weight) + 1.0
        )

    src, src_mask, tgt = _make_inputs()
    total = tgt.shape[0]
    l2r_slice, r2l_slice = BidirectionalLayout.split(total)

    # R2L-only loss must not move the bias embedding's gradient at all.
    out_r2l = dec(src, src_mask, tgt)
    out_r2l[r2l_slice].sum().backward()
    grad_from_r2l = dec._tree_rel_bias.emb.weight.grad
    assert grad_from_r2l is None or torch.equal(grad_from_r2l, torch.zeros_like(grad_from_r2l))

    dec.zero_grad(set_to_none=True)

    # Positive control: L2R-only loss must produce a nonzero gradient,
    # otherwise the isolation check above would be passing vacuously
    # (e.g. because the whole module got disconnected from the graph).
    out_l2r = dec(src, src_mask, tgt)
    out_l2r[l2r_slice].sum().backward()
    grad_from_l2r = dec._tree_rel_bias.emb.weight.grad
    assert grad_from_l2r is not None
    assert not torch.equal(grad_from_l2r, torch.zeros_like(grad_from_l2r))


def test_use_tree_bias_false_disables_builder_entirely():
    dec = _make_decoder(use_tree_bias=False)
    assert dec._tree_builder is None
    assert dec._tree_rel_bias is None
    tgt = torch.randint(3, VOCAB_SIZE, (4, 5))
    assert dec._build_rel_bias_for_tgt(tgt) is None
