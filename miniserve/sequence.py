"""DAY 2 — dunder methods and @property.

Goal: make a class that behaves like a builtin. This is where Python's
"protocols are just method names" design clicks.

Learn by hitting these:
  - __len__, __iter__, __repr__, __eq__
  - __getitem__ that handles BOTH an int and a slice   <- the interesting one
  - __bool__ (and what Python falls back to if you omit it)
  - @property for a value that is COMPUTED, never stored
  - why @property beats get_num_tokens() — and when it doesn't
    (rule: a property must be cheap and side-effect free)

Find it in the wild:
  vLLM: vllm/sequence.py  — the real Sequence class. It is this, plus KV state.

Docs: docs.python.org/3/reference/datamodel.html  (skim the dunder table only)
"""

from __future__ import annotations

from collections.abc import Iterator


class TokenSequence:
    def __init__(self, token_ids: list[int] | None = None) -> None:
      if token_ids is None:
        self._token_ids = []
      else:
        self._token_ids = list[int](token_ids)

    def __iter__(self) -> Iterator[int]:  
        return iter(self._token_ids)

    def __repr__(self) -> str:
        return f"TokenSequence({self._token_ids!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TokenSequence) and self._token_ids == other._token_ids

    def __bool__(self) -> bool:
        return bool(self._token_ids)
    def __getitem__ (self, index: int | slice) -> int | TokenSequence:
        if isinstance(index, int):
          return self._token_ids[index]
        return TokenSequence(self._token_ids[index])
    def __len__(self) -> int:
      return len(self._token_ids)

    def append(self, token_id: int) -> None:
      self._token_ids.append(token_id)
    @property
    def num_tokens(self) -> int:
      return len(self._token_ids)
    @property
    def last_token(self) -> int | None:
      return self._token_ids[-1] if self._token_ids else None