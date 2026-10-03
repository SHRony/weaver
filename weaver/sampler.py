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

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

from weaver.types import SamplingParams


@runtime_checkable
class Sampler(Protocol):
    def sample(self, logits: torch.Tensor, params: SamplingParams,/) -> int: ...


class GreedySampler:
    def sample(self, logits: torch.Tensor, params : SamplingParams, /) -> int:
        return int(logits.argmax())


class TopKSampler:
    def __init__(self, seed: int = 0) -> None:
        self._random = torch.Generator().manual_seed(seed)

    def sample(self, logits: torch.Tensor, params: SamplingParams, /) -> int:
        if params.temperature == 0:                        
            return int(logits.argmax())
        scaled = logits / params.temperature             
        k = params.top_k if params.top_k > 0 else scaled.numel()
        values, indices = torch.topk(scaled, k) 
        probs = torch.softmax(values, dim=-1)             
        pos = torch.multinomial(probs, 1, generator=self._random)   
        return int(indices[pos])                           

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