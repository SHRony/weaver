

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
    def bytes_of(self, token_id: int) -> bytes:
        return self._encoder.decode_single_token_bytes(token_id)
    @property
    def eos_id(self) -> int:
        return self._encoder.eot_token

class IncrementalDecoder:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._buffer: bytes = b""
    def decode(self, token_id: int) -> str:
        self._buffer += (self._tokenizer.bytes_of(token_id))
        try:
            text = self._buffer.decode("utf-8")
            self._buffer = b""
            return text
        except UnicodeDecodeError as e:
            text = self._buffer[:e.start].decode("utf-8") 
            self._buffer = self._buffer[e.start:]
            return text
    def flush(self) -> str:
        text = self._buffer.decode("utf-8", errors="replace") # dangling fragment → �
        self._buffer = b""
        return text