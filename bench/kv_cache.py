"""Benchmark with KV cache:

Run from the repo root:
    uv run python -m bench.kv_cache

KV cache = a cache of the previous tokens' keys and values. This script measures
the performance of the engine with and without the KV cache at several prompt lengths
and writes bench/kv_cache.csv.

Timing is wall-clock around each `next()` on the engine's generator, so one
sample covers ids-to-tensor + forward + sample + decode for one token.

The KV cache is enabled by default.
"""

from __future__ import annotations

import csv
import platform
import statistics
from pathlib import Path

import torch

from bench.baseline_naive import make_prompt, time_request
from weaver.engine import Engine
from weaver.model.gpt2 import GPT2
from weaver.sampler import GreedySampler
from weaver.tokenizer import GPT2Tokenizer

PROMPT_LENGTHS = (32, 128, 512, 896)
NEW_TOKENS = 32
RUNS = 3
DEVICE = "cpu"
OUT = Path(__file__).with_name("kv_cache.csv")



def run_prompt(
    engine: Engine, 
    threads: int,
    tokenizer: GPT2Tokenizer,
    prompt_length: int,
    new_tokens: int,
    cache_enabled: bool
    ) -> dict[str, str | int | float]:

    prompt = make_prompt(tokenizer, prompt_length)
    runs = [time_request(engine, prompt, new_tokens) for _ in range(RUNS)]
    rates = [len(r) / sum(r) for r in runs]
    row: dict[str, str | int | float] = {
        "cache_enabled": cache_enabled,
        "prompt_tokens": prompt_length,
        "tokens_produced": min(len(r) for r in runs),
        "runs": RUNS,
        "tok_per_s_median": round(statistics.median(rates), 2),
        "tok_per_s_min": round(min(rates), 2),
        "tok_per_s_max": round(max(rates), 2),
        "ms_per_token": round(
            statistics.median(1000 * sum(r) / len(r) for r in runs), 1
        ),
        "ms_first_token": round(
            statistics.median(1000 * r[0] for r in runs), 1
            ),
        "ms_last_token": round(
            statistics.median(1000 * r[-1] for r in runs), 1
            ),
        "device": DEVICE,
        "threads": threads,
        "torch": str(torch.__version__),
    }
    print(
        f"{prompt_length:6d}  {row['tokens_produced']:4}  {row['tok_per_s_median']:6} "
        f"({row['tok_per_s_min']}-{row['tok_per_s_max']})".ljust(36)
        + f"{row['ms_per_token']:8}  {row['ms_first_token']:8}  "
        f"{row['ms_last_token']:7}"
    )
    return row



def main() -> None:
    tokenizer = GPT2Tokenizer()
    model = GPT2.from_pretrained("gpt2")
    # check and print cache size for 1024 length
    dump_cache = model.build_cache(torch.device(DEVICE),1024)
    if dump_cache is not None:
        print(
            f"Cache size: {dump_cache.numel() * dump_cache.element_size()/1024/1024} MB"
        )
    else:
        print("Cache is None")
    rows: list[dict[str, str | int | float]] = []
    for cache_enabled in [True, False]:
        engine = Engine(
            tokenizer,
            GreedySampler(),
            pool_size=64,
            model=model,
            cache_enabled=cache_enabled,
            device=DEVICE
        )
        time_request(engine, make_prompt(tokenizer, 8), 4)  # warm-up, not recorded

        threads = torch.get_num_threads()
        print(f"machine : {platform.machine()}, python {platform.python_version()}")
        print(f"torch   : {torch.__version__}, device {DEVICE}, {threads} threads")
        print(f"request : {NEW_TOKENS} new tokens, greedy; median of {RUNS} runs\n")
        print("prompt  made  tok/s (min-max)        ms/token  first ms  last ms")

        for n in PROMPT_LENGTHS:
            row = run_prompt(engine, threads, tokenizer, n, NEW_TOKENS, cache_enabled)
            rows.append(row)
        if cache_enabled:
            rows.append(run_prompt(engine, threads, tokenizer, 1, 900, cache_enabled))

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
