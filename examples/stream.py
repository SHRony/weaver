"""Stream a GPT-2 completion through the async engine, token by token.

Run from the repo root:
    uv run python -m examples.stream "The capital of France is"
    uv run python -m examples.stream --temperature 0 --max-tokens 30 "Once upon"
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import time

from weaver.async_engine import AsyncEngine
from weaver.model.gpt2 import GPT2
from weaver.sampler import TopKSampler
from weaver.tokenizer import GPT2Tokenizer
from weaver.types import Request, SamplingParams


def load_model() -> GPT2:
    # Loading prints a progress bar and a hub notice to stderr; keep them out of
    # the demo. A failed load still raises.
    with contextlib.redirect_stderr(io.StringIO()):
        return GPT2.from_pretrained("gpt2")


async def stream(prompt: str, params: SamplingParams, seed: int) -> None:
    engine = AsyncEngine(
        GPT2Tokenizer(), TopKSampler(seed), pool_size=64, model=load_model()
    )
    await engine.start()
    request = Request("demo", prompt, params, time.time())
    print(prompt, end="", flush=True)
    count = 0
    reason: str | None = None
    start = time.perf_counter()
    try:
        async for out in engine.generate(request):
            print(out.text, end="", flush=True)  # one piece per token, as it arrives
            count += 1
            reason = out.finish_reason
    finally:
        await engine.aclose()
    elapsed = time.perf_counter() - start
    print(f"\n\n[{count} tokens, {count / elapsed:.1f} tok/s, stopped: {reason}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    params = SamplingParams(
        temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_tokens
    )
    asyncio.run(stream(args.prompt, params, args.seed))


if __name__ == "__main__":
    main()
