# Weaver

A small LLM inference engine, built from scratch to understand how the real ones
(vLLM, SGLang, TGI) actually work — then grown toward being one.

The name is the architecture: an inference server **weaves** many request-streams
through a single engine loop, advancing all of them together, one token per tick.

> **Status: educational, and honest about it.** The *plumbing* is real —
> continuous batching, async streaming, a KV-cache block pool, cancellation-safe
> resource handling. The *model* is currently faked (made-up logits, a
> character-level tokenizer) so the serving machinery can be built and understood
> without a GPU. Bolting in a real model is the next milestone (see Roadmap).

## What's real

- **Continuous batching** — one living batch; requests join as they arrive and
  leave the instant they finish, each at a different point in its own generation.
  Not "grab N, run to completion" — a short request never waits on a long one.
- **Async streaming** — `async for token in engine.generate(request)`. Many
  clients served concurrently on one event loop, each streaming its own tokens.
- **Admission control** — a fixed KV-cache **block pool**; requests wait when it's
  full and are admitted as blocks free (worst-case reservation, no preemption — a
  deliberate simplicity/utilization tradeoff).
- **Cancellation-safe** — a client disconnecting mid-stream frees its blocks
  immediately; no leaks.
- **Pluggable samplers & tokenizer** — greedy / top-k behind a structural
  `Protocol`; swap without the engine caring.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest          # 32 tests
uv run pyright         # strict, zero errors
uv run ruff check
```

## Layout

```
weaver/
  types.py          # SamplingParams, Request, Output — the data that flows through
  sequence.py       # TokenSequence — a growable token list that acts like a builtin
  sampler.py        # Sampler protocol + GreedySampler / TopKSampler; Tokenizer ABC
  instrument.py     # @timed, @retry, BlockPool (the KV-cache stand-in)
  engine.py         # synchronous engine: generate() + round-robin generate_batch()
  async_engine.py   # async continuous-batching engine (the real thing)
tests/              # a test per behavior, including the disconnect-frees-blocks one
```

## How it works, briefly

Clients push requests onto an inbound queue. One persistent engine-loop coroutine
admits what fits (block pool), steps every active request one token per tick via a
fake model → sampler, and routes each token to that request's own output queue,
which its `generate()` async-generator yields. A disconnect throws into the
generator; a `finally` flags it; the loop ejects it and frees its blocks.

Blocks are integer handles into a fixed pool — a stand-in for paged KV-cache
memory. That "cardboard" design turned out to mirror how real engines actually
manage GPU memory (they manage handles, not bytes), which is much of the point.

## Roadmap

- [ ] Replace the fake model with a real forward pass (load GPT-2 weights, nanoGPT-style)
- [ ] A real tokenizer (BPE) in place of the char-level one
- [ ] Actual KV-cache tensors behind the block handles
- [ ] An HTTP/streaming layer (FastAPI) so real clients can connect
- [ ] Prefix caching; smarter scheduling

## Why

I wanted to understand inference engines, so I built the plumbing from scratch
before reading vLLM and SGLang, then compared. It started as a learning exercise;
it's becoming a real (small) engine.

## License

MIT — see [LICENSE](LICENSE).
