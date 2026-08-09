import torch

from models.decoder import Decoder
from utils.vocab_info import VocabInfo

ID2TOK = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "x", 4: "^", 5: "{", 6: "2", 7: "}"}
VOCAB_SIZE = len(ID2TOK)


class _FakeWords:
    def __init__(self, idx2word):
        self.idx2word = idx2word


def _vocab_info() -> VocabInfo:
    return VocabInfo(
        vocab_size=VOCAB_SIZE, sos_id=1, eos_id=2, pad_id=0, words=_FakeWords(ID2TOK)
    )


def _make_decoder(use_tree_bias: bool, seed: int = 0) -> Decoder:
    torch.manual_seed(seed)
    return Decoder(
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


def _make_feature(real_batch=2, h=3, w=3, d=8, seed=123):
    g = torch.Generator().manual_seed(seed)
    feature = torch.randn(real_batch, h, w, d, generator=g)
    mask = torch.zeros(real_batch, h, w, dtype=torch.bool)
    return feature, mask


def test_beam_search_calls_tree_builder_with_l2r_only_batch_each_step():
    """The tree builder must only ever see the L2R half of the (post-beam-
    expand) batch -- never the full 2*B or 2*B*beam_size bidirectional
    batch. This is the property that lets R2L skip all bias computation
    entirely, not just skip the *result*."""
    dec = _make_decoder(use_tree_bias=True)
    dec.eval()

    real_batch, beam_size, max_len = 2, 3, 6
    feature, mask = _make_feature(real_batch=real_batch)

    call_shapes = []
    original_build = dec._tree_builder.build

    def spy_build(tgt_ids):
        call_shapes.append(tuple(tgt_ids.shape))
        return original_build(tgt_ids)

    dec._tree_builder.build = spy_build

    with torch.inference_mode():
        hyps = dec.beam_search(
            [feature],
            [mask],
            beam_size=beam_size,
            max_len=max_len,
            alpha=1.0,
            early_stopping=True,
            temperature=1.0,
        )

    assert len(hyps) == real_batch
    assert len(call_shapes) > 0
    for shape in call_shapes:
        # Pre-expand (cur_len==1): L2R half == real_batch rows.
        # Post-expand: L2R half == real_batch * beam_size rows.
        # Never 2x either of those -- that would mean R2L leaked in.
        assert shape[0] in (real_batch, real_batch * beam_size), shape


def test_beam_search_parity_zero_init_tree_bias_vs_disabled():
    """With tree bias freshly zero-initialized, decoding must be byte-for-
    byte identical to a decoder with tree bias disabled entirely -- the
    bias term is provably inert until trained."""
    real_batch, beam_size, max_len = 2, 3, 6

    dec_false = _make_decoder(use_tree_bias=False, seed=7)
    dec_false.eval()
    feature_a, mask_a = _make_feature(real_batch=real_batch, seed=999)
    with torch.inference_mode():
        hyps_false = dec_false.beam_search(
            [feature_a.clone()],
            [mask_a.clone()],
            beam_size=beam_size,
            max_len=max_len,
            alpha=1.0,
            early_stopping=True,
            temperature=1.0,
        )

    dec_true = _make_decoder(use_tree_bias=True, seed=7)
    dec_true.eval()
    feature_b, mask_b = _make_feature(real_batch=real_batch, seed=999)
    with torch.inference_mode():
        hyps_true = dec_true.beam_search(
            [feature_b.clone()],
            [mask_b.clone()],
            beam_size=beam_size,
            max_len=max_len,
            alpha=1.0,
            early_stopping=True,
            temperature=1.0,
        )

    assert len(hyps_false) == len(hyps_true) == real_batch
    for hf, ht in zip(hyps_false, hyps_true):
        assert hf.seq == ht.seq
        assert abs(hf.score - ht.score) < 1e-5
