from c3rnt2.continuous.types import Sample
from c3rnt2.continuous.formatting import format_chat_sample


def test_format_chat_sample_roundtrip() -> None:
    sample = Sample(prompt="Hola", response="Que tal")
    text = format_chat_sample(sample)
    assert "Hola" in text
    assert "Que tal" in text
    assert "### User" in text
    assert "### Assistant" in text
