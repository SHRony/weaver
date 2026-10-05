from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import torch


class Model(ABC):
  @abstractmethod
  def forward(self, idx: torch.Tensor) -> torch.Tensor:
    ...
  @abstractmethod
  def eval(self) -> Self:
    ...
class FakeModel(Model):
    def __init__(self, seed: int) -> None:
      self._seed = seed
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
      # sum over all ids, as a Python int
      sm = int(idx.sum().item()) ^ self._seed
      # index == token id now; keep it a letter
      peak = ord("a") + sm % 26
      logits = torch.tensor([1 / (abs(i - peak) + 1) + i * 1e-6 for i in range(128)])
      return logits.reshape(1, 1, -1) 
    def eval(self) -> Self:
      return self