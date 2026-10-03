"""Day 5 — engine tests: laziness, stops, disconnect, batching, admission."""
import pytest
import torch

from weaver.engine import Engine
from weaver.model.model import FakeModel, Model
from weaver.sampler import CharTokenizer, GreedySampler, TopKSampler
from weaver.types import Output, Request, SamplingParams


def make_engine(pool_size: int = 64) -> Engine:
    return Engine(
        CharTokenizer(), GreedySampler(), pool_size=pool_size, model=FakeModel(42)
    )

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
    for sampler in (GreedySampler(), TopKSampler(seed=1)):
        engine = Engine(CharTokenizer(), sampler, pool_size=8, model=FakeModel(42))
        e = engine.generate(make_request())
        outs = list(e)
        assert len(outs) == 8 and all(isinstance(o, Output) for o in outs)


class EosAfterModel(Model):
    """Behaves like FakeModel for `after` forwards, then emits `eos_id`.

    FakeModel only ever peaks on letters, so nothing in the suite reaches the
    engine's EOS branch without a model that is told to emit it.
    """

    def __init__(self, eos_id: int, after: int) -> None:
        self._inner = FakeModel(42)
        self._eos_id = eos_id
        self._after = after
        self._calls = 0

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        self._calls += 1
        if self._calls <= self._after:
            return self._inner.forward(idx)
        logits = torch.zeros(1, 1, 128)
        logits[0, 0, self._eos_id] = 1.0
        return logits


def eos_engine(after: int) -> tuple[Engine, int]:
    tok = CharTokenizer()
    model = EosAfterModel(tok.eos_id, after)
    return Engine(tok, GreedySampler(), pool_size=8, model=model), tok.eos_id


def test_eos_stops_generation_and_is_not_rendered() -> None:
    engine, eos_id = eos_engine(after=2)
    outs = list(engine.generate(make_request(max_tokens=8)))

    assert len(outs) == 3  # two real tokens, then the EOS step
    assert all(not o.finished for o in outs[:-1])
    assert outs[-1].finished and outs[-1].finish_reason == "stop"

    # the token is reported, but its text never reaches the stream
    assert outs[-1].new_token_id == eos_id
    assert outs[-1].text == ""
    text = "".join(o.text for o in outs)
    assert len(text) == 2 and text.isalpha()


def test_eos_on_last_allowed_token_is_stop_not_length() -> None:
    engine, _ = eos_engine(after=2)
    outs = list(engine.generate(make_request(max_tokens=3)))

    assert len(outs) == 3
    assert outs[-1].finish_reason == "stop"
