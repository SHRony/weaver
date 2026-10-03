import pytest
import torch

from weaver.sampler import (
    CharTokenizer,
    GreedySampler,
    Sampler,
    Tokenizer,
    TopKSampler,
)
from weaver.types import SamplingParams

GREEDY = SamplingParams(temperature=0.0)


@pytest.mark.parametrize(
    "logits, expected",
    [
        (torch.tensor([0.1, 0.9, 0.3]), 1),
        (torch.tensor([0.4, 0.2, 0.7]), 2),
    ],
)
def test_greedy_sampler(logits: torch.Tensor, expected: int) -> None:
    assert GreedySampler().sample(logits, SamplingParams()) == expected


def test_greedy_fits_sampler_protocol() -> None:
    s: Sampler = GreedySampler()
    assert s.sample(torch.tensor([0.1, 0.9, 0.3]), SamplingParams()) == 1


def test_topk_k1_is_greedy_for_any_seed_and_temperature() -> None:
    logits = torch.tensor([0.1, 0.9, 0.3, 0.7])
    for seed in (0, 1, 42):
        for temp in (0.1, 1.0, 5.0):
            params = SamplingParams(temperature=temp, top_k=1)
            assert TopKSampler(seed).sample(logits, params) == 1


def test_topk_temperature_zero_is_greedy() -> None:
    logits = torch.tensor([0.1, 0.9, 0.3, 0.7])
    assert TopKSampler(seed=7).sample(logits, GREEDY) == 1 # top_k=0: whole vocab


def test_topk_same_seed_same_draws() -> None:
    logits = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2, 0.4, 0.6, 0.5])
    params = SamplingParams(temperature=1.0, top_k=4)
    a, b = TopKSampler(seed=42), TopKSampler(seed=42)
    assert [a.sample(logits, params) for _ in range(20)] == [
        b.sample(logits, params) for _ in range(20)
    ]


def test_topk_never_picks_outside_top_k() -> None:
    logits = torch.tensor([10.0, 9.9, -100.0])
    params = SamplingParams(temperature=1.0, top_k=2)
    s = TopKSampler(seed=0)
    picks = {s.sample(logits, params) for _ in range(200)}
    assert picks == {0, 1}          # both survivors appear; the cut one never does


def test_topk_is_proportional_not_uniform() -> None:
    logits = torch.tensor([5.0, 0.0])             # softmax -> 0.993 / 0.007
    params = SamplingParams(temperature=1.0, top_k=2)
    s = TopKSampler(seed=0)
    wins = sum(s.sample(logits, params) == 0 for _ in range(200))
    assert wins > 180                              # uniform would give ~100


def test_topk_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        SamplingParams(top_k=-1)


def test_tokenizer_round_trip() -> None:
    assert CharTokenizer().roundtrip("hello") == "hello"


def test_tokenizer_is_abstract() -> None:
    with pytest.raises(TypeError):
        Tokenizer()  # pyright: ignore[reportAbstractUsage]
