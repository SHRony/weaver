# miniserve — week 2 structural Python

A fake LLM engine. No ML, no tensors, no model. Pure plumbing — which is
exactly the part of vLLM/SGLang that is Python.

Week 2 builds the synchronous object model.
Week 4 adds asyncio batching on top of these same objects.

## Setup

    uv init && uv add --dev pytest pyright ruff
    uv run pyright --level error      # keep this at zero
    uv run pytest

Put pyright on STRICT from day one (pyproject below). It will yell at you
constantly. That is the point — it is the fastest teacher of `typing`, and
coming from TypeScript the feedback loop is one you already know.

## The rule for each day

1. Write the code first. Get pyright to zero errors.
2. THEN read the one doc page for that construct.
3. THEN grep for it in real code (see "find it in the wild" in each file).

Reading first doesn't stick. Reading after you've fought the thing does.

## Day map  (~1.5h each)

  day 1  types.py        @dataclass, typing        strict mode on
  day 2  sequence.py     dunder methods, @property
  day 3  sampler.py      Protocol vs ABC           the TypeScript mapping
  day 4  instrument.py   decorators, context managers
  day 5  engine.py       generators                wire it all together
