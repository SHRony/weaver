import pytest
import torch

from weaver.sampler import (
    CharTokenizer,
    GreedySampler,
    Sampler,
    Tokenizer,
    TopKSampler,
)


@pytest.mark.parametrize(
    "sampler, logits, expected",
    [
        (GreedySampler(), torch.tensor([0.1, 0.9, 0.3]), 1),
        (GreedySampler(), torch.tensor([0.4, 0.2, 0.7]), 2),
    ],
)
def test_greedy_sampler(sampler: Sampler, logits: torch.Tensor, expected: int) -> None:
    assert sampler.sample(logits) == expected


def test_top_k_sampler() -> None:
    logits = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2, 0.4, 0.6, 0.5])
    sampler = TopKSampler(2, 42)
    assert sampler.sample(logits) in [1, 3]
    for i in range(len(logits)):
        samplera = TopKSampler(i + 1, 42)
        samplerb = TopKSampler(i + 1, 42)
        assert samplera.sample(logits) == samplerb.sample(logits)


def test_topk_with_k1_is_greedy() -> None:
    assert TopKSampler(k=1, seed=0).sample(torch.tensor([0.1, 0.9, 0.3])) == 1


def test_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer()
    assert tokenizer.roundtrip("hello") == "hello"


def test_tokenizer_is_abstract() -> None:
    with pytest.raises(TypeError):
        Tokenizer()  # pyright: ignore[reportAbstractUsage]


def test_greedy_fits_sampler_protocol() -> None:
    s: Sampler = GreedySampler()
    assert s.sample(torch.tensor([0.1, 0.9, 0.3])) == 1


def test_topk_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        TopKSampler(k=0)
