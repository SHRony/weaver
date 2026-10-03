# Weaver

A small LLM inference engine, built from scratch to understand how the real ones
(vLLM, SGLang, TGI) actually work — then grown toward being one.

The name is the architecture: an inference server **weaves** many request-streams
through a single engine loop, advancing all of them together, one token per tick.

> **Status: educational, and honest about it.** The serving machinery is real:
> continuous-batching scheduling, async streaming, admission control over a block
> pool, cancellation-safe cleanup. The model is real too: GPT-2 small, written
> from scratch in PyTorch and loaded from the published weights. What it is not
> yet is fast. There is no KV cache, so every new token recomputes the whole
> sequence, and each request gets its own forward pass (see Roadmap).

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
- **A real model** — GPT-2 small written from scratch in plain PyTorch
  (`weaver/model/gpt2.py`), loading the published weights. Each module was checked
  against HuggingFace's while it was built, and a slow end-to-end test asserts
  that the engine's greedy output matches HuggingFace's token ids.
- **Streaming that respects UTF-8** — GPT-2's tokens are byte fragments, so one
  emoji can span several tokens. An incremental decoder holds partial characters
  back, and a client never sees `�` in the middle of one.
- **Sampling per request** — greedy, or temperature → top-k → proportional draw,
  with the parameters carried by each request.
- **Stops and failures** — end-of-text token, stop strings, or a token limit. A
  model error aborts that one request and leaves the engine serving the others.
- **Pluggable samplers, tokenizer and model** — a structural `Protocol` for
  samplers, abstract base classes for the tokenizer and the model. The fast test
  suite runs on a fake model and a character tokenizer through the same seams.

## What isn't there yet

- **A KV cache** — each new token runs a full forward over the request's whole
  sequence, so the cost of a token grows with the context before it.
- **A batched forward** — the scheduler interleaves requests, but each one gets
  its own forward pass. Nothing is batched at the tensor level yet.
- **Real KV memory** — the block pool is still bookkeeping: integer handles with
  no tensors behind them.
- **A network layer** — the engine is a Python API. There is no HTTP server.
- **GPU** — the device is configurable, but it has only been run on CPU.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest            # fast suite: fake model, no weights needed
uv run pytest -m slow    # real GPT-2 through the engine, checked against HuggingFace
uv run pyright           # strict, zero errors
uv run ruff check
```

The fast suite fetches GPT-2's tokenizer files (about 1.5 MB) the first time it
runs. Anything that loads the real model downloads the GPT-2 weights (about
520 MB) from HuggingFace the first time.

## Try it

```bash
uv run python -m examples.stream "The capital of France is"
```

```
The capital of France is also the capital of the French Republic. The capital city is located in the east of France, in the Rhine, the eastern Rhine, the Alps, the Vauxhall, St. John the Evangelist's Square, Marseille, and St. Paul's Cathedral. The capital is also

[60 tokens, 23.9 tok/s, stopped: length]
```

That is GPT-2 small being GPT-2 small. The text arrives one token at a time through
the async engine. The speed is from one run on a laptop CPU with no KV cache, and
it varies a lot with what else the machine is doing. `--temperature`, `--top-k`,
`--max-tokens` and `--seed` change the sampling.

## Layout

```
weaver/
  types.py          # SamplingParams, Request, Output — the data that flows through
  sequence.py       # TokenSequence — a growable token list that acts like a builtin
  sampler.py        # Sampler protocol + GreedySampler / TopKSampler; Tokenizer ABC
  tokenizer.py      # GPT2Tokenizer (BPE via tiktoken) + the incremental byte decoder
  instrument.py     # @timed, @retry, BlockPool (the KV-cache stand-in)
  engine.py         # synchronous engine: generate() + round-robin generate_batch()
  async_engine.py   # async continuous-batching engine (the real thing)
  model/
    model.py        # Model interface + FakeModel (what the fast tests run on)
    gpt2.py         # GPT-2 from scratch: LayerNorm, MLP, Attention, Block, loader
examples/
  stream.py         # stream a completion from the terminal
tests/              # a test per behavior, including the disconnect-frees-blocks one
```

## How it works, briefly

Clients push requests onto an inbound queue. One persistent engine-loop coroutine
admits what fits (block pool), steps every active request one token per tick via
the model → sampler, and routes each token to that request's own output queue,
which its `generate()` async-generator yields. A disconnect throws into the
generator; a `finally` flags it; the loop ejects it and frees its blocks.

A step today is the naive one: encode the request's whole sequence, run a full
forward, take the logits at the last position, sample one token, and decode it to
text through the byte buffer.

Blocks are integer handles into a fixed pool — a stand-in for paged KV-cache
memory. That "cardboard" design turned out to mirror how real engines actually
manage GPU memory (they manage handles, not bytes), which is much of the point.

## Roadmap

- [x] Replace the fake model with a real forward pass (GPT-2, written from scratch)
- [x] A real tokenizer (BPE) in place of the char-level one
- [ ] A KV cache, so a new token stops recomputing the whole sequence
- [ ] Actual KV-cache tensors behind the block handles
- [ ] A batched forward across requests
- [ ] An HTTP/streaming layer (FastAPI) so real clients can connect
- [ ] Prefix caching; smarter scheduling

## Why

I wanted to understand inference engines, so I built the plumbing from scratch
before reading vLLM and SGLang, then compared. It started as a learning exercise;
it's becoming a real (small) engine.

## License

MIT — see [LICENSE](LICENSE).
