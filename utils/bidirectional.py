"""Single source of truth for the L2R/R2L batch layout used by CoMER's
bidirectional decoding.

CoMER trains and decodes L2R and R2L simultaneously by stacking both
directions into one batch: rows [0, B) are L2R, rows [B, 2B) are R2L, where
B is the original (single-direction) batch size. This convention is set up
in `utils.utils.to_bi_tgt_out` / `to_bi_tgt_out_from_padded` and preserved
through `models/comer.py` and `utils/generation_utils.py::DecodeModel.beam_search`
(including every later beam-expand step, since beam expansion repeats each
half in place and never interleaves L2R with R2L rows).

The old DSP391m_Group1_FALL25/LiSRB repo re-derived `batch_size // 2` by hand
in half a dozen places (models/decoder.py, utils/generation_utils.py, tests).
Any of those call sites drifting out of sync with the others was exactly the
kind of bug that made that repo's KV-cache/tree-bias interaction hard to
trust. Here, only `models/decoder.py::Decoder._build_rel_bias_for_tgt` calls
into this class -- `attention.py` and `transformer_decoder.py` never learn
about L2R/R2L at all, they just add whatever `rel_bias` tensor they're given.
"""

from typing import Tuple


class BidirectionalLayout:
    """The one place that knows rows [0, B) = L2R, rows [B, 2B) = R2L."""

    @staticmethod
    def assert_even(total: int) -> int:
        assert total % 2 == 0, f"bidirectional batch size must be even, got {total}"
        return total // 2

    @classmethod
    def split(cls, total: int) -> Tuple[slice, slice]:
        """Return (l2r_slice, r2l_slice) for a bidirectional batch of size `total`."""
        half = cls.assert_even(total)
        return slice(0, half), slice(half, total)
