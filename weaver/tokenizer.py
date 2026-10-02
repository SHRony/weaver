

from collections.abc import Sequence

import tiktoken

from weaver.sampler import Tokenizer


class GPT2Tokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()
        self._encoder = tiktoken.get_encoding("gpt2")
    def encode(self, text: str) -> list[int]:
        return self._encoder.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._encoder.decode(token_ids)
    @property
    def eos_id(self) -> int:
        return self._encoder.eot_token