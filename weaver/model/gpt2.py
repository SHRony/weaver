"""GPT-2 from scratch — built one module at a time, each parity-tested against HF.

Config (GPT-2 small): 12 layers, 12 heads, 768 dim, ctx 1024, vocab 50257.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for training; set to 0.0 for inference/parity testing
    bias: bool = True
class LayerNorm(nn.Module):
    """Normalize each token's vector to ~zero-mean / unit-variance over the last
    dim, then apply a learned per-channel scale and shift. (Week-3 "voltage
    regulator".) Parity target: transformer.h.*.ln_1 / ln_2 / transformer.ln_f.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))    # learned scale  (gamma)
        self.bias = nn.Parameter(torch.zeros(dim))      # learned shift  (beta)
        self.eps = eps

    # Forward pass for a single layer, normalizes features before doing the wx + b
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias

class MLP(nn.Module):
  def __init__(self, config : GPT2Config) -> None:
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    # GPT-2 uses tanh approximation instead of using gelu directly
    self.gelu = nn.GELU(approximate="tanh")
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    self.dropout = nn.Dropout(config.dropout)
  # widens -> introduces non linearity -> projects back to original embedding space
  def forward(self, x : torch.Tensor) -> torch.Tensor:
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, config : GPT2Config) -> None:
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.config = config
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # causal mask to ensure that attention is only applied to the left in the
        # input sequence
        # need to register buffer instead of self.bias = value, so that it moves to
        # whichever device the attention is running in
        self.bias: torch.Tensor
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        bias_val = mask.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        x = self.c_attn(x)
        q, k, v = torch.split(x, self.config.n_embd, dim=2)
        n_head = self.config.n_head

        # dividing into multiple heads to capture different aspects of similarities
        # independently
        # doing the transpose to make sure the mix happens cross tokens and not cross
        # heads
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2) 
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        # dividing by approximate variance for normalization
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
  def __init__(self, config : GPT2Config) -> None:
    super().__init__()
    self.ln_1 = LayerNorm(config.n_embd)
    self.attn = Attention(config)
    self.ln_2 = LayerNorm(config.n_embd)
    self.mlp = MLP(config)
  def forward(self, x : torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln_1(x))
    return x + self.mlp(self.ln_2(x))
class Transformer(nn.Module):
    def __init__(self, config:GPT2Config)->None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        
class GPT2(nn.Module):
    def __init__(self, config : GPT2Config) -> None:
        super().__init__()
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config
        self.lm_head.weight = self.transformer.wte.weight
    def forward(self, idx : torch.Tensor) -> torch.Tensor:
        device = idx.device
        _, t = idx.size()
        msg = (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.config.block_size}"
        )
        assert t <= self.config.block_size, msg
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)
    @classmethod
    def from_pretrained(cls, model_type:str)->GPT2:
        """Loads pretrained GPT-2 weights from Hugging Face. Implements weight tying."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        # n_layer, n_head, n_embd are determined by model type
        configs: dict[str, GPT2Config] = {
            "gpt2": GPT2Config(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": GPT2Config(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": GPT2Config(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": GPT2Config(n_layer=48, n_head=25, n_embd=1600),
        }
        config = configs[model_type]
                model = cls(config)

        # import HERE, not module top — keep HF a dev dep
        from transformers import GPT2LMHeadModel

        hf = GPT2LMHeadModel.from_pretrained(model_type)  # pyright: ignore[reportUnknownMemberType]
        # walk hf.state_dict(): skip ("attn","bias") by segment match, transpose the
        # 4 Conv1D suffixes, copy_ into model.state_dict()[key] under torch.no_grad()
        hf_dict = hf.state_dict()
        model_dict = model.state_dict()
        transposed_keys = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        for key in hf_dict:
            if any (key.endswith(k) for k in transposed_keys):
                model_dict[key].copy_(hf_dict[key].transpose(0, 1))
            else:
                model_dict[key].copy_(hf_dict[key])
        return model.eval()