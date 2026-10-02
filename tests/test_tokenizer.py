import pytest

from weaver.tokenizer import GPT2Tokenizer


@pytest.mark.parametrize("text", [
    "hello",
    " leading space",
    "emoji 🎉 inside",
    "বাংলা",
    "\n\n  indented\tTabs",
])
def test_roundtrip(text: str) -> None:
    assert GPT2Tokenizer().roundtrip(text) == text


def test_eos_id() -> None:
    assert GPT2Tokenizer().eos_id == 50256