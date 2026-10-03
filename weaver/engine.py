"""DAY 5 — generators. Wire everything together.

This file should use every construct from days 1-4. If any of them feels
bolted on rather than needed, you over-applied it — that's a real finding,
note it down.

Learn by hitting these:
  - a method that yields, typed as Iterator[Output]
  - yield from, to delegate to a sub-generator
  - why the CALLER controls the pace (call generate(), don't iterate, observe
    that nothing runs)
  - .close() and GeneratorExit — what happens when a client disconnects
    mid-stream? This is a real inference-server concern, not a toy one.

Find it in the wild:
  vLLM: LLM.generate / AsyncLLMEngine.generate — the real thing is an
        ASYNC generator. You'll build that version in week 4. Read it now
        anyway; you'll understand the sync half.

Docs: docs.python.org/3/reference/expressions.html#yieldexpr
"""

from __future__ import annotations

from collections import deque
from collections.abc import Generator, Iterator
from math import ceil

import torch

from weaver.instrument import BlockPool, allocate, timed
from weaver.model.model import Model
from weaver.sampler import Sampler, Tokenizer
from weaver.sequence import TokenSequence
from weaver.tokenizer import IncrementalDecoder
from weaver.types import Output, Request, SamplingParams


class Engine:
    """A fake engine. Generates tokens by sampling from made-up logits.

    TODO: __init__(self, tokenizer: Tokenizer, sampler: Sampler)

    TODO: def generate(self, request: Request) -> Iterator[Output]:
              yields one Output per token until max_tokens or a stop string.
              Decorate the whole call with @timed.
              Use a TokenSequence to hold the ids.
              Use the BlockPool context manager to hold blocks for the
              lifetime of the request.

    TODO: def generate_batch(self, requests: list[Request]) -> Iterator[Output]:
              round-robin across requests, yielding whichever token is ready.
              Use yield from where it helps.

              ^ this is a synchronous sketch of continuous batching. Week 4
                turns it into the real async version. Keep it.

    Exit test for the week: `uv run pyright` is clean at strict, `uv run
    pytest` passes, and you can explain to yourself why each construct is
    here rather than a plain function or dict.
    """
    def __init__(
      self, 
      tokenizer: Tokenizer,
      sampler: Sampler,
      pool_size:int,
      model: Model,
      device: str = "cpu",
      ) -> None:
        self.tokenizer = tokenizer
        self.sampler = sampler
        self._model = model
        self._block_pool = BlockPool(pool_size)
        self._block_size = 16
        self._pool_size = pool_size
        self._device = device
    def get_pool_size(self):
      return self._block_pool.get_size()
    @timed
    def generate(self, request: Request) -> Generator[Output, None, None]:
      with allocate(self._block_pool, self._blocks_needed(request)) as _:
        sequence: TokenSequence = TokenSequence(self.tokenizer.encode(request.prompt))
        sampling_param:SamplingParams = request.params
        finished = False
        prompt_len = len(sequence)
        decoder = IncrementalDecoder(self.tokenizer)
        generated_text = ""
        while not finished:
          ids = torch.tensor([list(sequence)], device=self._device)
          with torch.inference_mode():
            logits = self._model.forward(ids)
          token_id = self.sampler.sample(logits[0, -1], sampling_param)
          if token_id == self.tokenizer.eos_id:
            finished = True
            finish_reason = "stop"
            piece = decoder.flush()
          else:
            sequence.append(token_id)
            piece = decoder.decode(token_id)
            generated_text += piece
            if sampling_param.stop and generated_text.endswith(sampling_param.stop):
              finished = True
              finish_reason = "stop"
            elif (len(sequence) - prompt_len) >= sampling_param.max_tokens:
              finished = True
              finish_reason = "length"
            else:
              finish_reason = None
            if finished:
              final_piece = decoder.flush()
              piece += final_piece
              generated_text += final_piece
          
          yield Output(request_id=request.request_id,
            new_token_id=token_id, text=piece,
            finished=finished, finish_reason=finish_reason)
    
    def _blocks_needed(self, request: Request) -> int:
      prompt_tokens = len(self.tokenizer.encode(request.prompt))
      return ceil((prompt_tokens + request.params.max_tokens) / self._block_size)
    def generate_batch(self, requests: list[Request]) -> Iterator[Output]:
      pending : deque[Request] = deque[Request]()
      generators : list[Iterator[Output]] = []
      for request in requests:
        needed = self._blocks_needed(request)
        if self._pool_size < needed:
          raise ValueError(
                f"request {request.request_id} needs {needed} blocks; "
                f"pool only has {self._pool_size} total"
            )
        else:
          pending.append(request)
      
      while len(pending) > 0 or len(generators) > 0:
        def is_free()->bool : 
          return len(pending) > 0 and self._block_pool.has_available(
            self._blocks_needed(pending[0])
          )
        while(is_free()):
          request = pending.popleft()
          generators.append(self.generate(request))
          yield next(generators[-1])
        to_remove : list[Iterator[Output]] = []
        for generator in generators:
          try:
            yield next(generator)
          except StopIteration:
            to_remove.append(generator)
          
        for generator in to_remove:
          generators.remove(generator)
    def has_enough_blocks(self, request: Request) -> bool:
      return self._block_pool.has_available( self._blocks_needed(request))
    def can_never_handle_request(self, request: Request) -> bool:
      return self._pool_size < self._blocks_needed(request)