"""DAY 1 — @dataclass and typing.

Goal: define the data that moves through an inference engine, and get
pyright --strict to zero errors. Nothing here has behaviour.

Learn by hitting these, in order:
  - @dataclass generates __init__/__repr__/__eq__ for you
  - frozen=True makes instances immutable & hashable  (why does that matter
    for something you use as a dict key?)
  - field(default_factory=list) — and WHY a bare `= []` default is a bug
  - Literal, Optional, Sequence vs list, Final
  - `from __future__ import annotations` and what it changes

Find it in the wild:
  vLLM: vllm/sampling_params.py     (a real SamplingParams)
  vLLM: vllm/outputs.py             (a real RequestOutput)
Read those AFTER you've written yours. Compare. What did they need that you
didn't think of?

Docs (one page each, read after coding): docs.python.org/3/library/dataclasses.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FinishReason = Literal["length", "stop", "abort", "pool_size"]


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature : float =  1.0
    top_k: int = 0
    max_tokens: int = 16
    stop: tuple[str, ...] = ()
    seed: int|None = None
    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(
              f"Temperature must be non-negative, but got {self.temperature}"
            )
        if self.top_k < 0:
            raise ValueError(
              f"Top_k must be non-negative, but got {self.top_k}"
            )
@dataclass(slots=True)
class Request:
    request_id: str
    prompt: str
    params: SamplingParams
    arrival_time: float
    token_ids: list[int] = field(default_factory=list[int])

@dataclass(slots=True)
class Output:
    request_id: str
    new_token_id: int | None
    text: str
    finished: bool
    finish_reason: FinishReason | None = None