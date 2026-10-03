"""DAY 3 — Protocol vs ABC. The day that maps directly off TypeScript.

THE distinction to internalise:

  Protocol = STRUCTURAL typing. "If it has these methods, it fits."
             No inheritance, no registration. This is exactly a TypeScript
             `interface`. The implementer doesn't even import the Protocol.

  ABC      = NOMINAL typing. "You must inherit from me."
             This is Java/C++ abstract base class. Gives you shared
             implementation and enforcement at instantiation time.

  Rule of thumb: Protocol when you don't own the implementers (or want them
  decoupled). ABC when you own them and want shared code + a hard contract.

Find it in the wild:
  vLLM: vllm/model_executor/layers/sampler.py
  vLLM: search the repo for `Protocol)` and for `(ABC)` — compare which
        kinds of things use which. That comparison IS the lesson.

Docs: typing.Protocol, abc.ABC
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Sampler(Protocol):
    def sample(self, logits: torch.Tensor, /) -> int: ...


class GreedySamplerBase:
    def sample(self, logits: torch.Tensor, /) -> int:
        ret = 0
        for i, logit in enumerate(logits):
            if logit > logits[ret]:
                ret = i
        return ret


class TopKSamplerBase:
    def __init__(self, k: int, seed: int = 0) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        self._random = random.Random(seed)

    def sample(self, logits: torch.Tensor, /) -> int:
        logits_copy = [(logit, i) for i, logit in enumerate (logits)]
        logits_copy.sort(reverse=True)
        top_k_indices = [i for _, i in logits_copy[: self._k]]
        return self._random.choice(top_k_indices)


class GreedySampler:
    def sample(self, logits: torch.Tensor, /) -> int:
        return int(logits.argmax())


class TopKSampler:
    def __init__(self, k: int, seed: int = 0) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        self._random = random.Random(seed)

    def sample(self, logits: torch.Tensor, /) -> int:
        return int(self._random.choice(torch.topk(logits, self._k).indices))

class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...
    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str: ...
    @abstractmethod
    def bytes_of(self, token_id: int) -> bytes: ...
    def roundtrip(self, text: str) -> str:
        return self.decode(self.encode(text))


class CharTokenizer(Tokenizer):
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)
    def bytes_of(self, token_id: int) -> bytes:
        return bytes([token_id])