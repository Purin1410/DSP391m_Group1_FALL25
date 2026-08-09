import pytest

from utils.bidirectional import BidirectionalLayout


def test_split_even_batch():
    l2r, r2l = BidirectionalLayout.split(8)
    assert l2r == slice(0, 4)
    assert r2l == slice(4, 8)


def test_split_slices_are_contiguous_and_cover_whole_batch():
    total = 12
    l2r, r2l = BidirectionalLayout.split(total)
    indices = list(range(*l2r.indices(total))) + list(range(*r2l.indices(total)))
    assert indices == list(range(total))


def test_split_zero_batch():
    l2r, r2l = BidirectionalLayout.split(0)
    assert l2r == slice(0, 0)
    assert r2l == slice(0, 0)


def test_odd_batch_raises():
    with pytest.raises(AssertionError):
        BidirectionalLayout.split(7)


def test_assert_even_returns_half():
    assert BidirectionalLayout.assert_even(10) == 5
