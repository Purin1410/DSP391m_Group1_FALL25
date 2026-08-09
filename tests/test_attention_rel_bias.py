import pytest
import torch

from models.transformer.attention import MultiheadAttention
from models.transformer.transformer_decoder import TransformerDecoder, TransformerDecoderLayer


def test_rel_bias_zero_is_noop_for_multihead_attention():
    torch.manual_seed(0)
    mha = MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0)
    mha.eval()
    q = torch.randn(4, 3, 8)  # tgt_len, bsz, embed_dim
    out_none, _ = mha(q, q, q)
    zero_bias = torch.zeros(3 * 2, 4, 4)  # bsz*num_heads, tgt_len, src_len
    out_zero, _ = mha(q, q, q, rel_bias=zero_bias)
    assert torch.allclose(out_none, out_zero, atol=1e-6)


def test_rel_bias_nonzero_changes_output():
    torch.manual_seed(0)
    mha = MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0)
    mha.eval()
    q = torch.randn(4, 3, 8)
    out_none, _ = mha(q, q, q)
    bias = torch.randn(3 * 2, 4, 4)
    out_biased, _ = mha(q, q, q, rel_bias=bias)
    assert not torch.allclose(out_none, out_biased, atol=1e-6)


def test_rel_bias_broadcasts_from_bsz_shape():
    """rel_bias with first dim == bsz (not bsz*num_heads) must broadcast
    identically across heads, matching the (B, T, S) full-batch shape
    TreeRelativeBias would never actually produce (it emits (B*H, T, S)),
    but attention.py documents support for it -- verify it still works."""
    torch.manual_seed(0)
    mha = MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0)
    mha.eval()
    q = torch.randn(4, 3, 8)
    per_head_bias = torch.randn(2, 4, 4).repeat(3, 1, 1)  # same per batch item...
    # Build an explicit (bsz, T, S) bias and its manually-broadcast (bsz*H, T, S) equivalent.
    bsz_bias = torch.randn(3, 4, 4)
    expanded = bsz_bias.unsqueeze(1).expand(3, 2, 4, 4).reshape(3 * 2, 4, 4)
    out_bsz, _ = mha(q, q, q, rel_bias=bsz_bias)
    out_expanded, _ = mha(q, q, q, rel_bias=expanded)
    assert torch.allclose(out_bsz, out_expanded, atol=1e-6)


def test_rel_bias_shape_mismatch_raises():
    mha = MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0)
    q = torch.randn(4, 3, 8)
    bad_bias = torch.randn(3 * 2, 5, 5)  # wrong tgt_len/src_len
    with pytest.raises(RuntimeError):
        mha(q, q, q, rel_bias=bad_bias)


def test_rel_bias_only_affects_self_attention_not_cross_attention():
    torch.manual_seed(0)
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    layer.eval()

    cross_attn_calls = []
    original_cross_forward = layer.multihead_attn.forward

    def spy_forward(*args, **kwargs):
        cross_attn_calls.append(kwargs.get("rel_bias", "MISSING"))
        return original_cross_forward(*args, **kwargs)

    layer.multihead_attn.forward = spy_forward

    tgt = torch.randn(4, 3, 8)
    memory = torch.randn(5, 3, 8)
    bias = torch.randn(3 * 2, 4, 4)

    layer(tgt, memory, arm=None, rel_bias=bias)

    assert len(cross_attn_calls) == 1
    # The layer never forwards rel_bias to cross-attention at all, so the
    # kwarg is simply absent (not explicitly None) from that call.
    assert cross_attn_calls[0] == "MISSING"


def test_rel_bias_none_vs_zero_identical_through_full_layer():
    torch.manual_seed(0)
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    layer.eval()
    tgt = torch.randn(4, 3, 8)
    memory = torch.randn(5, 3, 8)

    out_none, attn_none, self_attn_none = layer(tgt, memory, arm=None, rel_bias=None)
    zero_bias = torch.zeros(3 * 2, 4, 4)
    out_zero, attn_zero, self_attn_zero = layer(tgt, memory, arm=None, rel_bias=zero_bias)

    assert torch.allclose(out_none, out_zero, atol=1e-6)


def test_transformer_decoder_last1_only_biases_final_layer():
    torch.manual_seed(0)
    num_layers = 3
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    decoder = TransformerDecoder(layer, num_layers, arm=None, tree_bias_layers="last1")
    decoder.eval()

    seen_rel_bias = []
    for i, mod in enumerate(decoder.layers):
        original_self_forward = mod.self_attn.forward

        def make_spy(idx, orig):
            def spy(*args, **kwargs):
                seen_rel_bias.append((idx, kwargs.get("rel_bias")))
                return orig(*args, **kwargs)

            return spy

        mod.self_attn.forward = make_spy(i, original_self_forward)

    tgt = torch.randn(4, 3, 8)
    memory = torch.randn(5, 3, 8)
    bias = torch.randn(3 * 2, 4, 4)
    decoder(tgt, memory, height=1, rel_bias=bias)

    assert len(seen_rel_bias) == num_layers
    for idx, rb in seen_rel_bias:
        if idx == num_layers - 1:
            assert rb is not None
        else:
            assert rb is None


def test_transformer_decoder_all_layers_biased_by_default():
    torch.manual_seed(0)
    num_layers = 3
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    decoder = TransformerDecoder(layer, num_layers, arm=None)  # tree_bias_layers="all" default
    decoder.eval()

    seen_rel_bias = []
    for i, mod in enumerate(decoder.layers):
        original_self_forward = mod.self_attn.forward

        def make_spy(idx, orig):
            def spy(*args, **kwargs):
                seen_rel_bias.append((idx, kwargs.get("rel_bias")))
                return orig(*args, **kwargs)

            return spy

        mod.self_attn.forward = make_spy(i, original_self_forward)

    tgt = torch.randn(4, 3, 8)
    memory = torch.randn(5, 3, 8)
    bias = torch.randn(3 * 2, 4, 4)
    decoder(tgt, memory, height=1, rel_bias=bias)

    assert len(seen_rel_bias) == num_layers
    for _, rb in seen_rel_bias:
        assert rb is not None
