"""
Test timed
"""

import time

import pytest

from miniserve.instrument import BlockPool, allocate, retry, timed


@timed
def time_consuming_add(a: int, b: int) -> int:
    time.sleep(0.1)
    return a + b


def test_timed(capsys: pytest.CaptureFixture[str]) -> None:
    assert time_consuming_add(1, 2) == 3
    assert time_consuming_add(2, 7) == 9
    assert time_consuming_add.__name__ == "time_consuming_add"
    logs = capsys.readouterr().out
    assert logs.__contains__("Time taken")
    assert "ms" in logs


def retry_fail():
    cnt = 0

    @retry(times=2)
    def counter():
        nonlocal cnt
        cnt += 1
        if cnt == 3:
            return
        raise ValueError("Less than 3 retries")

    counter()


def test_retry_succeed_eventually():
    cnt = 0

    @retry(times=3)
    def counter():
        nonlocal cnt
        cnt += 1
        if cnt == 3:
            return
        raise Exception("Retry failed")

    counter()


def test_retry() -> None:
    with pytest.raises(ValueError):
        retry_fail()


def test_pool_releases_on_exception() -> None:
    pool = BlockPool(8)
    with pytest.raises(RuntimeError), pool.allocate(4):
        raise RuntimeError("client died")
    assert sorted(pool.get_pool()) == list(range(8))


def test_generator_allocate_releases_on_success() -> None:
    pool = BlockPool(8)
    with allocate(pool, 4):
        pass
    assert sorted(pool.get_pool()) == list(range(8))


def test_counts_drop_while_borrowed() -> None:
    pool = BlockPool(8)
    with pool.allocate(4) as blocks:
        assert len(blocks) == 4
        assert pool.available() == 4


def test_too_many_raises_and_pool_untouched() -> None:
    pool = BlockPool(8)
    with pytest.raises(ValueError), allocate(pool, 99):
        pass
    assert sorted(pool.get_pool()) == list[int](range(8))
