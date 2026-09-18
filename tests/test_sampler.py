import pytest

from miniserve.sampler import (
    CharTokenizer,
    GreedySampler,
    Sampler,
    Tokenizer,
    TopKSampler,
)


@pytest.mark.parametrize(
    "sampler, logits, expected",
    [
        (GreedySampler(), [0.1, 0.9, 0.3], 1),
        (GreedySampler(), [0.4, 0.2, 0.7], 2),
    ],
)
def test_greedy_sampler(sampler: Sampler, logits: list[float], expected: int) -> None:
    assert sampler.sample(logits) == expected


def test_top_k_sampler() -> None:
    logits = [0.1, 0.9, 0.3, 0.7, 0.2, 0.4, 0.6, 0.5]
    sampler = TopKSampler(2, 42)
    assert sampler.sample(logits) in [1, 3]
    for i, _ in enumerate[float](logits):
        samplera = TopKSampler(i + 1, 42)
        samplerb = TopKSampler(i + 1, 42)
        assert samplera.sample(logits) == samplerb.sample(logits)


def test_topk_with_k1_is_greedy() -> None:
    assert TopKSampler(k=1, seed=0).sample([0.1, 0.9, 0.3]) == 1


def test_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer()
    assert tokenizer.roundtrip("hello") == "hello"


def test_tokenizer_is_abstract() -> None:
    with pytest.raises(TypeError):
        Tokenizer()  # pyright: ignore[reportAbstractUsage]


def use(s: Sampler) -> None:
    s.sample([0.1, 0.9, 0.3])


use(GreedySampler())  # pyright: ignore[reportUnknownReturnType]


def test_topk_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        TopKSampler(k=0)
