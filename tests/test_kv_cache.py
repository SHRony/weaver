

import torch

from weaver.model.gpt2 import GPT2, GPT2Config
from weaver.tokenizer import GPT2Tokenizer

n_embd = 32
n_head = 2
n_layer = 2
def test_kv_cache_random_initialization() -> None:
    max_tokens = 4
    kv_cache = torch.zeros(
        1, n_layer, 2, n_head, max_tokens, n_embd // n_head, dtype=torch.float32
    )
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    tokenizer = GPT2Tokenizer()
    model = GPT2(GPT2Config(n_embd=n_embd, n_head=n_head, n_layer=n_layer))
    token_ids = tokenizer.encode("Hello, world!")
    model.eval()
    tokens = torch.tensor([token_ids])
    with torch.inference_mode():    
        logits1 = model.forward(tokens, None)
        model.forward(tokens[:, :-1], kv_cache[:, :, :, :, :len(token_ids) - 1])
        logits2 = model.forward(tokens[:, -1:], kv_cache[:, :, :, :, :len(token_ids)])
        torch.testing.assert_close(
            logits1[:, -1:], logits2[:, -1:], rtol=1e-4, atol=1e-4
        )
def test_kv_cache_decode_write() -> None:
    max_tokens = 4
    kv_cache = torch.zeros(
        1, n_layer, 2, n_head, max_tokens, n_embd // n_head, dtype=torch.float32
    )
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    tokenizer = GPT2Tokenizer()
    model = GPT2(GPT2Config(n_embd=n_embd, n_head=n_head, n_layer=n_layer))
    token_ids = tokenizer.encode("Hello, world!")
    model.eval()
    tokens = torch.tensor([token_ids])
    with torch.inference_mode():    
        logits1 = model.forward(tokens, None)
        model.forward(tokens[:, :-2], kv_cache[:, :, :, :, :len(token_ids) - 2])
        model.forward(
            tokens[:, -2:-1], kv_cache[:, :, :, :, :len(token_ids) - 1:]
        )
        logits2 = model.forward(
            tokens[:, -1:], kv_cache[:, :, :, :, :len(token_ids)]
        )
        torch.testing.assert_close(
            logits1[:, -1:], logits2[:, -1:], rtol=1e-4, atol=1e-4
        )
def test_kv_cache_batch_forward() -> None:
    max_tokens = 4
    kv_cache = torch.zeros(
        1, n_layer, 2, n_head, max_tokens, n_embd // n_head, dtype=torch.float32
    )
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    tokenizer = GPT2Tokenizer()
    model = GPT2(GPT2Config(n_embd=n_embd, n_head=n_head, n_layer=n_layer))
    token_ids = tokenizer.encode("Hello, world!")
    model.eval()
    tokens = torch.tensor([token_ids])
    with torch.inference_mode():    
        logits1 = model.forward(tokens, None)
        model.forward(tokens[:, :-2], kv_cache[:, :, :, :, :len(token_ids) - 2])
        logits2 = model.forward(
            tokens[:, -2:], kv_cache[:, :, :, :, :len(token_ids)]
        )
        torch.testing.assert_close(
            logits1[:, -1:], logits2[:, -1:], rtol=1e-4, atol=1e-4
        )