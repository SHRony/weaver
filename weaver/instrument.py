"""DAY 4 — decorators and context managers. The densest day.

A decorator is just:  f = decorator(f).  Nothing more.
A context manager is just: try/finally with a name.

Learn by hitting these, in order (each is harder than the last):
  1. a plain decorator            @timed
  2. why you need functools.wraps (omit it, then print f.__name__)
  3. a decorator WITH ARGUMENTS   @retry(times=3)   <- three nested functions.
     Everyone finds this confusing once. Write it, then draw the call.
  4. a class-based context manager  __enter__ / __exit__
  5. the same thing via @contextlib.contextmanager + yield
     <- note this is a GENERATOR being used as a context manager. The yield
        IS the body of the `with` block. That connection is worth sitting with.

Find it in the wild:
  vLLM: grep for `@contextmanager` — used for CUDA graph capture,
        profiling regions, and lock scopes.
  vLLM: grep for `@functools.wraps` / custom decorators in vllm/utils.py

Docs: functools.wraps, contextlib.contextmanager
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import TracebackType
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"Time taken: {(time.perf_counter() - start) * 1000} ms")
        return result

    return wrapper


def retry(
    times: int = 3, delay: float = 0.0
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if i == times - 1:
                        raise
                    time.sleep(delay)
                    continue
            raise

        return wrapper

    return decorator


class BlockPool:
    def take(self, n: int) -> list[int]:
        if n > len(self._pool):
            raise ValueError("Not enough blocks available")
        return [self._pool.pop() for _ in range(n)]

    def release(self, blocks: list[int]) -> None:
        self._pool.extend(blocks)

    def __init__(self, total_blocks: int):
        self._total_blocks = total_blocks
        self._pool = list[int](range(total_blocks))

    def __enter__(self) -> list[int]:
        ret = self.take(self._pending)
        self._current = ret
        return ret

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release(self._current)

    def allocate(self, n: int) -> BlockPool:
        self._pending = n
        return self

    def available(self) -> int:
        return len(self._pool)

    def get_pool(self) -> list[int]:
        return self._pool

    def has_available(self, sz: int) -> bool:
        return len(self._pool) >= sz

    def get_max_capacity(self) -> int:
        return self._total_blocks

    def get_size(self) -> int:
        return len(self._pool)


@contextmanager
def allocate(pool: BlockPool, n: int) -> Generator[list[int]]:
    blocks = pool.take(n)
    try:
        yield blocks
    finally:
        pool.release(blocks)
