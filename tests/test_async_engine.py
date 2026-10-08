import asyncio

import torch

from weaver.async_engine import AsyncEngine
from weaver.model.model import FakeModel, Model
from weaver.sampler import CharTokenizer, GreedySampler, TopKSampler
from weaver.types import Output, Request, SamplingParams


def make_engine(pool_size: int = 64) -> AsyncEngine:
    return AsyncEngine(CharTokenizer(), GreedySampler(), pool_size, model=FakeModel(42))


def make_request(
    rid: str = "r1", max_tokens: int = 8, prompt: str = "hello"
) -> Request:
    return Request(rid, prompt, SamplingParams(max_tokens=max_tokens), 0.0)


async def test_async_harness_works() -> None:
    await asyncio.sleep(0)
    assert True


async def test_single_request_streams_to_length() -> None:
    e = make_engine()
    await e.start()
    try:
        outs = [o async for o in e.generate(make_request(max_tokens=8))]
    finally:
        await e.aclose()
    assert len(outs) == 8
    assert outs[-1].finished and outs[-1].finish_reason == "length"
    assert all(not o.finished for o in outs[:-1])
    assert "".join(o.text for o in outs).isalpha()


async def test_two_clients_interleave_and_isolate() -> None:
    e = make_engine()
    await e.start()
    try:

        async def collect(rid: str) -> list[Output]:
            return [o async for o in e.generate(make_request(rid, max_tokens=4))]

        a, b = await asyncio.gather(collect("A"), collect("B"))
    finally:
        await e.aclose()
    assert [o.request_id for o in a] == ["A"] * 4  # each got ITS OWN stream
    assert [o.request_id for o in b] == ["B"] * 4


async def test_over_capacity_requests_wait_then_complete() -> None:
    # pool of 4; each request needs ceil((5+20)/16)=2 blocks -> only 2 fit at once
    e = make_engine(pool_size=4)
    await e.start()
    try:

        async def collect(rid: str) -> list[Output]:
            return [o async for o in e.generate(make_request(rid, max_tokens=20))]

        results = await asyncio.gather(*(collect(f"r{i}") for i in range(3)))
    finally:
        await e.aclose()
    assert all(len(r) == 20 for r in results)  # everyone finished
    assert e.free_blocks() == 4  # pool whole at the end


async def test_cancel_midstream_releases_blocks() -> None:
    e = make_engine(pool_size=4)
    await e.start()
    try:
        gen = e.generate(make_request(max_tokens=50))
        await gen.__anext__()  # token 1 — request running, holding blocks
        await gen.__anext__()  # token 2
        await gen.aclose()  # client disconnects mid-stream
        await asyncio.sleep(0.02)  # let the loop eject + free
        assert e.free_blocks() == 4  # blocks came back
    finally:
        await e.aclose()


async def test_runs_with_both_samplers() -> None:
    for sampler in (GreedySampler(), TopKSampler(seed=1)):
        e = AsyncEngine(CharTokenizer(), sampler, 16, model=FakeModel(42))
        await e.start()
        try:
            outs = [o async for o in e.generate(make_request(max_tokens=6))]
        finally:
            await e.aclose()
        assert len(outs) == 6 and all(isinstance(o, Output) for o in outs)


class CrashingModel(Model):
    """Delegates to FakeModel, but raises on any prompt containing '!'.

    Lets one request crash while another on the same engine stays healthy —
    which is the property under test: a bad request must not take the loop down.
    """

    def __init__(self) -> None:
        super().__init__()
        self._inner = FakeModel(42)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if bool((idx == ord("!")).any()):
            raise RuntimeError("model exploded")
        return self._inner.forward(idx)

async def test_model_crash_aborts_request_but_engine_survives() -> None:
    e = AsyncEngine(CharTokenizer(), GreedySampler(), 8, model=CrashingModel())
    await e.start()
    try:

        async def collect(rid: str, prompt: str) -> list[Output]:
            return [o async for o in e.generate(make_request(rid, 4, prompt))]

        # wait_for turns "client hangs forever" into a test failure, not a stuck suite
        bad, good = await asyncio.wait_for(
            asyncio.gather(collect("bad", "boom!"), collect("good", "hello")),
            timeout=2.0,
        )

        # the crashed request got exactly one terminal Output, marked abort
        assert len(bad) == 1
        assert bad[0].finished and bad[0].finish_reason == "abort"
        assert bad[0].request_id == "bad"

        # the healthy request on the same engine was unaffected
        assert len(good) == 4 and good[-1].finish_reason == "length"

        # the loop is still alive: a request submitted AFTER the crash completes
        after = await asyncio.wait_for(collect("after", "hello"), timeout=2.0)
        assert len(after) == 4

        # all blocks came back: the crashed request's `with allocate` unwound on the
        # exception; the finished ones release on the loop's next tick, so let it run
        await asyncio.sleep(0.02)
        assert e.free_blocks() == 8
    finally:
        await e.aclose()
