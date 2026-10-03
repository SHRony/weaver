import pytest

from weaver.tokenizer import GPT2Tokenizer, IncrementalDecoder


def test_eos_id() -> None:
    assert GPT2Tokenizer().eos_id == 50256

HOSTILE = [
    "hello",
    " leading space",
    "emoji 🎉 inside",
    "বাংলা",
    "\n\n  indented\tTabs",
]


@pytest.mark.parametrize("text", HOSTILE)
def test_roundtrip(text: str) -> None:
    assert GPT2Tokenizer().roundtrip(text) == text

@pytest.mark.parametrize("text", HOSTILE)   
def test_decoder_pieces_concatenate_to_whole(text: str) -> None:
    tok = GPT2Tokenizer()
    dec = IncrementalDecoder(tok)
    pieces = [dec.decode(i) for i in tok.encode(text)]
    assert "".join(pieces) + dec.flush() == text

def test_decoder_holds_partial_emoji_until_complete() -> None:
    tok = GPT2Tokenizer()
    ids = tok.encode("🎉")
    assert len(ids) == 3                     
    dec = IncrementalDecoder(tok)
    assert dec.decode(ids[0]) == ""           
    assert dec.decode(ids[1]) == ""           
    assert dec.decode(ids[2]) == "🎉"         
    assert dec.flush() == ""