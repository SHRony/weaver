"""End-to-end: text prompt -> GPT2Tokenizer -> Engine -> real GPT-2 -> greedy ids.

Expected ids are HF's own greedy output for this prompt
(see notes/phase1/parity_gpt2.py).
"""

import pytest

from weaver.engine import Engine
from weaver.model.gpt2 import GPT2
from weaver.sampler import GreedySampler
from weaver.tokenizer import GPT2Tokenizer
from weaver.types import Request, SamplingParams


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