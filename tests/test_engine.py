"""Day 5 — engine tests: laziness, stops, disconnect, batching, admission."""
import pytest

from miniserve.engine import Engine
from miniserve.sampler import CharTokenizer, GreedySampler, TopKSampler
from miniserve.types import Output, Request, SamplingParams


def make_engine(pool_size: int = 64) -> Engine:
    return Engine(CharTokenizer(), GreedySampler(), pool_size=pool_size)

def make_request(rid: str = "r1", max_tokens: int = 8) -> Request:
    return Request(rid, "hello", SamplingParams(max_tokens=max_tokens), 0.0)


def test_generate_is_lazy() -> None:
    e = make_engine(4)
    g = e.generate(make_request())
    assert e.get_pool_size() == 4        # nothing taken yet
    next(g)
    assert e.get_pool_size() < 4         # first step claimed blocks

def test_full_generation_stops_at_length() -> None:
    outs = list(make_engine().generate(make_request(max_tokens=8)))
    assert len(outs) == 8
    assert outs[-1].finished and outs[-1].finish_reason == "length"
    assert all(not o.finished for o in outs[:-1])
    assert "".join(o.text for o in outs).isalpha()

def test_stop_string_ends_early() -> None:
    # greedy + fixed model = deterministic text; steal a substring from a dry run
    text = "".join(o.text for o in make_engine().generate(make_request(max_tokens=8)))
    r = Request("r1", "hello", SamplingParams(max_tokens=8, stop=(text[:3],)), 0.0)
    outs = list(make_engine().generate(r))
    assert outs[-1].finish_reason == "stop"
    assert len(outs) < 8

def test_close_releases_blocks() -> None:          # the week's thesis
    e = make_engine(4)
    g = e.generate(make_request(max_tokens=20))
    next(g)
    next(g)                                # generation in flight
    del g                                       # client disconnects
    assert e.get_pool_size() == 4            # nothing leaked

def test_batch_interleaves() -> None:
    e = make_engine()
    outs = list(e.generate_batch([make_request("r1", 4), make_request("r2", 4)]))
    assert len(outs) == 8
    assert [o.request_id for o in outs][:4] == ["r1", "r2", "r1", "r2"]

def test_batch_defers_until_capacity() -> None:
    e = make_engine(4)                              # each request needs 2 blocks
    reqs = [make_request(f"r{i}", 20) for i in (1, 2, 3)]
    order = [o.request_id for o in e.generate_batch(reqs)]
    assert len(order) == 60
    i3 = order.index("r3")
    assert order[:i3].count("r1") == 20             # r1 fully done before r3 admitted
    assert e.get_pool_size() == 4

def test_impossible_request_rejected() -> None:
    e = make_engine(2)
    with pytest.raises(ValueError, match="r1"):
        list(e.generate_batch([make_request("r1", max_tokens=500)]))

def test_any_sampler_fits_the_seam() -> None:
    for sampler in (GreedySampler(), TopKSampler(k=3, seed=1)):
        e = Engine(CharTokenizer(), sampler, pool_size=8).generate(make_request())
        outs = list(e)
        assert len(outs) == 8 and all(isinstance(o, Output) for o in outs)
