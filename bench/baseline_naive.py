"""Baseline benchmark: naive decode, before any KV cache.

Run from the repo root:
    uv run python -m bench.baseline_naive

Naive decode = a full forward over every token so far, for each new token. So
the cost of one token depends on how many came before it. This script measures
that at several prompt lengths and writes bench/baseline_naive.csv.

Timing is wall-clock around each `next()` on the engine's generator, so one
sample covers ids-to-tensor + forward + sample + decode for one token.
"""

from __future__ import annotations

import contextlib
import csv
import io
import platform
import statistics
import time
from pathlib import Path

import torch

from weaver.engine import Engine
from weaver.model.gpt2 import GPT2
from weaver.sampler import GreedySampler
from weaver.tokenizer import GPT2Tokenizer
from weaver.types import Request, SamplingParams

PROMPT_LENGTHS = (32, 128, 512, 896)
NEW_TOKENS = 32
RUNS = 3
DEVICE = "cpu"
OUT = Path(__file__).with_name("baseline_naive.csv")


def make_prompt(tokenizer: GPT2Tokenizer, n_tokens: int) -> str:
    """A prompt that encodes to exactly n_tokens (" the" is a single token)."""
    prompt = " the" * n_tokens
    got = len(tokenizer.encode(prompt))
    if got != n_tokens:
        raise ValueError(f"prompt encodes to {got} tokens, wanted {n_tokens}")
    return prompt


def time_request(engine: Engine, prompt: str, new_tokens: int) -> list[float]:
    """Seconds spent producing each token of one request."""
    params = SamplingParams(temperature=0.0, max_tokens=new_tokens)
    request = Request("bench", prompt, params, 0.0)
    # Engine.generate is wrapped in @timed, which prints when the generator is
    # created; keep that out of the benchmark's output.
    with contextlib.redirect_stdout(io.StringIO()):
        gen = engine.generate(request)
    seconds: list[float] = []
    while True:
        start = time.perf_counter()
        try:
            next(gen)
        except StopIteration:
            return seconds
        seconds.append(time.perf_counter() - start)


def main() -> None:
    tokenizer = GPT2Tokenizer()
    model = GPT2.from_pretrained("gpt2")
    engine = Engine(
        tokenizer, GreedySampler(), pool_size=64, model=model, device=DEVICE
    )
    time_request(engine, make_prompt(tokenizer, 8), 4)  # warm-up, not recorded

    threads = torch.get_num_threads()
    print(f"machine : {platform.machine()}, python {platform.python_version()}")
    print(f"torch   : {torch.__version__}, device {DEVICE}, {threads} threads")
    print(f"request : {NEW_TOKENS} new tokens, greedy; median of {RUNS} runs\n")
    print("prompt  made  tok/s (min-max)        ms/token  first ms  last ms")

    rows: list[dict[str, str | int | float]] = []
    for n in PROMPT_LENGTHS:
        prompt = make_prompt(tokenizer, n)
        runs = [time_request(engine, prompt, NEW_TOKENS) for _ in range(RUNS)]
        rates = [len(r) / sum(r) for r in runs]
        row: dict[str, str | int | float] = {
            "prompt_tokens": n,
            "tokens_produced": min(len(r) for r in runs),
            "runs": RUNS,
            "tok_per_s_median": round(statistics.median(rates), 2),
            "tok_per_s_min": round(min(rates), 2),
            "tok_per_s_max": round(max(rates), 2),
            "ms_per_token": round(
                statistics.median(1000 * sum(r) / len(r) for r in runs), 1
            ),
            "ms_first_token": round(statistics.median(1000 * r[0] for r in runs), 1),
            "ms_last_token": round(statistics.median(1000 * r[-1] for r in runs), 1),
            "device": DEVICE,
            "threads": threads,
            "torch": str(torch.__version__),
        }
        rows.append(row)
        print(
            f"{n:6d}  {row['tokens_produced']:4}  {row['tok_per_s_median']:6} "
            f"({row['tok_per_s_min']}-{row['tok_per_s_max']})".ljust(36)
            + f"{row['ms_per_token']:8}  {row['ms_first_token']:8}  "
            f"{row['ms_last_token']:7}"
        )

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
