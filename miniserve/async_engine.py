from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field

from miniserve.engine import Engine
from miniserve.sampler import Sampler, Tokenizer
from miniserve.sequence import TokenSequence
from miniserve.types import Output, Request

# import asyncio
# from collections import deque
# from collections.abc import AsyncIterator
# from dataclasses import dataclass, field
# from math import ceil


# from miniserve.engine import FakeModel
# from miniserve.instrument import BlockPool
# from miniserve.sampler import Sampler, Tokenizer
# from miniserve.sequence import TokenSequence
# from miniserve.types import Output, Request
@dataclass
class Inflight:
    request: Request  # the client's untouched data
    output_queue: asyncio.Queue[Output]  # engine plumbing lives HERE
    sequence: TokenSequence
    prompt_len: int
    blocks: list[int] = field(default_factory=list[int])
    cancelled: bool = False


class AsyncEngine:
    def __init__(self, tokenizer: Tokenizer, sampler: Sampler, pool_size: int) -> None:
        self._engine = Engine(tokenizer, sampler, pool_size)
        self._inbound: asyncio.Queue[Inflight] = asyncio.Queue()
        self._response_queues: dict[str, asyncio.Queue[Output]] = {}
        self._response_generators: dict[str, Generator[Output, None, None]] = {}
        self._pending_requests: list[Inflight] = []
        self._running_requests: list[Inflight] = []
        self._loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:  # called from inside the event loop
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self.engine_loop())

    async def aclose(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

    def free_blocks(self) -> int:
        return self._engine.get_pool_size()

    async def generate(self, request: Request) -> AsyncGenerator[Output, None]:
        inflight = Inflight(
            request,
            asyncio.Queue[Output](),
            TokenSequence(self._engine.tokenizer.encode(request.prompt)),
            len(self._engine.tokenizer.encode(request.prompt)),
        )
        await self._inbound.put(inflight)
        try:
            while not inflight.cancelled:
                out = await inflight.output_queue.get()
                yield out
                if out.finished:
                    return
        finally:
            inflight.cancelled = True

    async def engine_loop(self) -> None:
        while True:
            # idle: block until a request actually arrives (no busy-spin)
            if not self._running_requests and not self._pending_requests:
                self._pending_requests.append(await self._inbound.get())
            # drain any others already waiting, without blocking
            while not self._inbound.empty():
                self._pending_requests.append(self._inbound.get_nowait())

            while len(self._pending_requests) > 0:
                inflight = self._pending_requests[0]
                if inflight.cancelled:
                    self._pending_requests.remove(inflight)
                    continue
                if self._engine.can_never_handle_request(inflight.request):
                    inflight.output_queue.put_nowait(
                        Output(
                            request_id=inflight.request.request_id,
                            new_token_id=0,
                            text="",
                            finished=True,
                        )
                    )
                    self._pending_requests.remove(inflight)
                    continue
                if self._engine.has_enough_blocks(inflight.request):
                    self._response_queues[inflight.request.request_id] = asyncio.Queue[
                        Output
                    ]()
                    self._running_requests.append(inflight)
                    self._pending_requests.remove(inflight)
                    self._response_generators[inflight.request.request_id] = (
                        self._engine.generate(inflight.request)
                    )
                    output = next(
                        self._response_generators[inflight.request.request_id]
                    )
                    inflight.output_queue.put_nowait(output)
                else:
                    break

            to_remove: list[Inflight] = []
            for inflight in self._running_requests:
                try:
                    if inflight.cancelled:
                        self._response_generators[inflight.request.request_id].close()
                        to_remove.append(inflight)
                    else:
                        output = next(
                            self._response_generators[inflight.request.request_id]
                        )
                        inflight.output_queue.put_nowait(output)
                except StopIteration:
                    to_remove.append(inflight)
            for inflight in to_remove:
                self._running_requests.remove(inflight)
                self._response_queues.pop(inflight.request.request_id)
                self._response_generators.pop(inflight.request.request_id)
            await asyncio.sleep(0)
