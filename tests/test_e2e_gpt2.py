"""End-to-end: text prompt -> GPT2Tokenizer -> Engine -> real GPT-2 -> greedy ids.

Expected ids are HF's own greedy output for this prompt
(see notes/phase1/parity_gpt2.py).
"""

import pytest

from bench.baseline_naive import make_prompt
from weaver.async_engine import AsyncEngine
from weaver.engine import Engine
from weaver.model.gpt2 import GPT2
from weaver.sampler import GreedySampler
from weaver.tokenizer import GPT2Tokenizer
from weaver.types import Output, Request, SamplingParams


@pytest.mark.slow
def test_engine_with_real_gpt2_matches_hf_greedy() -> None:
    engine = Engine(
        GPT2Tokenizer(),
        GreedySampler(),
        pool_size=4,
        model=GPT2.from_pretrained("gpt2"),
    )
    request = Request(
        "e2e", "The capital of France is", SamplingParams(max_tokens=8), 0.0
    )

    ids = [out.new_token_id for out in engine.generate(request)]

    # " the capital of the French Republic, and"
    assert ids == [262, 3139, 286, 262, 4141, 2066, 11, 290]


@pytest.mark.slow
def test_engine_with_real_gpt2_successfully_stops_at_stop_string() -> None:
    engine = Engine(
        GPT2Tokenizer(),
        GreedySampler(),
        pool_size=4,
        model=GPT2.from_pretrained("gpt2"),
    )
    request = Request(
        "e2e", "The capital of France is", SamplingParams(max_tokens=8, stop=("Fren",)),
         0.0,
    )

    outputs = list(engine.generate(request))
    res : str = ""
    for out in outputs:
        res += out.text
    assert res.endswith("Fren")
    assert(outputs[-1].finish_reason == "stop")
    # " the capital of the French Republic, and"

@pytest.mark.slow
def test_engine_with_real_gpt2_successfully_stops_with_stop_token() -> None:
    engine = Engine(
        GPT2Tokenizer(),
        GreedySampler(),
        pool_size=256,
        model=GPT2.from_pretrained("gpt2"),
    )
    request = Request(
        "e2e", "Thank you for reading.", SamplingParams(max_tokens=8),
         0.0,
    )

    outputs = list(engine.generate(request))
    assert len(outputs) == 4
    assert outputs[-1].finish_reason == "stop"
    assert outputs[-1].new_token_id == engine.tokenizer.eos_id
    assert outputs[-1].text == ""
    assert "".join(o.text for o in outputs) == "\n\nAdvertisements"

@pytest.mark.slow
async def test_async_engine_with_real_gpt2_matches_hf_greedy() -> None:
    engine = AsyncEngine(
        GPT2Tokenizer(),
        GreedySampler(),
        pool_size=4,
        model=GPT2.from_pretrained("gpt2"),
    )
    request = Request(
        "e2e", "Thank you for reading.", SamplingParams(max_tokens=8),
         0.0,
    )
    await engine.start()
    try:
        outputs = [out async for out in engine.generate(request)]
    finally:
        await engine.aclose()
    assert engine.free_blocks() == 4
    assert len(outputs) == 4
    
@pytest.mark.slow
def test_cache_enabled_works() -> None:
    engine1 = Engine(
        GPT2Tokenizer(),
        GreedySampler(),
        pool_size=4,
        model=GPT2.from_pretrained("gpt2"),
        cache_enabled=True,
    )
    engine2 = Engine(
        GPT2Tokenizer(),
        GreedySampler(),
        pool_size=4,
        model=GPT2.from_pretrained("gpt2"),
        cache_enabled=False,
    )
    request = Request("r1", "hello world", SamplingParams(max_tokens=8), 0.0)
    outs1 = list[Output](engine1.generate(request))
    outs2 = list[Output](engine2.generate(request))
    assert outs1 == outs2

@pytest.mark.slow
def test_cache_enabled_works_large_context() -> None:
    engine1 = Engine(
        GPT2Tokenizer(),
        GreedySampler(),
        pool_size=1024,
        model=GPT2.from_pretrained("gpt2"),
        cache_enabled=True,
    )
    engine2 = Engine(
        GPT2Tokenizer(),
        GreedySampler(),
        pool_size=1024,
        model=GPT2.from_pretrained("gpt2"),
        cache_enabled=False,
    )
    prompt = make_prompt(GPT2Tokenizer(), 896)
    request1 = Request("r", prompt, SamplingParams(max_tokens=32), 0.0)
    request2 = Request("r", prompt, SamplingParams(max_tokens=32), 0.0)
    outs1 = list[Output](engine1.generate(request1))
    outs2 = list[Output](engine2.generate(request2))
    assert outs1 == outs2