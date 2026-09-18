"""Target for day 2. These should pass when TokenSequence is done.

Write the day-1 and day-3/4/5 tests yourself — writing the test first is
often the fastest way to discover what the API should be.
"""

from miniserve.sequence import TokenSequence


def test_len_and_index() -> None:
    s = TokenSequence([1, 2, 3])
    assert len(s) == 3
    assert s[0] == 1
    assert s[-1] == 3


def test_slice_returns_same_type() -> None:
    s = TokenSequence([1, 2, 3])
    assert s[1:] == TokenSequence([2, 3])


def test_iteration_and_truthiness() -> None:
    assert list(TokenSequence([1, 2])) == [1, 2]
    assert not TokenSequence([])
    assert TokenSequence([0])          # non-empty but contains 0 — still True


def test_properties() -> None:
    s = TokenSequence([7, 8])
    assert s.num_tokens == 2
    assert s.last_token == 8
    assert TokenSequence([]).last_token is None
