from pathlib import Path

import pytest

from core.ingest import ingest

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_ingest_rejects_unsupported_format(tmp_path):
    bogus = tmp_path / "manuscript.pdf"
    bogus.write_text("whatever")
    with pytest.raises(ValueError):
        ingest(bogus)


def test_pharaoh_duck_strips_title_and_byline():
    result = ingest(FIXTURES / "pharaoh_duck" / "source.docx")

    joined = "\n".join(result.narration_lines)
    assert "Pharaoh Duck" not in result.narration_lines
    assert "By Sara Braas" not in joined
    assert result.narration_lines[0] == "Meep Meep Meep"


def test_pharaoh_duck_keeps_bracketed_background_dialogue_in_narration():
    result = ingest(FIXTURES / "pharaoh_duck" / "source.docx")

    joined = "\n".join(result.narration_lines)
    assert "feral duck" in joined
    assert "Waterfowl Center" in joined
    # No author illo notes in this manuscript -- nothing to extract.
    assert result.illustration_notes == []


def test_turkey_takeover_strips_front_and_back_matter():
    result = ingest(FIXTURES / "turkey_takeover" / "source.docx")

    joined = "\n".join(result.narration_lines)
    assert "Turkey Takeover" not in result.narration_lines
    assert "Based on a true story" not in joined
    assert "sarajbraas@gmail.com" not in joined
    assert "Wordcount" not in joined
    assert "Pitch:" not in joined
    assert "for reference only" not in joined
    assert result.narration_lines[0].startswith("Terrible Tom is a riot at school.")


def test_turkey_takeover_extracts_author_illustration_notes():
    result = ingest(FIXTURES / "turkey_takeover" / "source.docx")

    joined = "\n".join(result.narration_lines)
    assert "(illo:" not in joined.lower()

    note_texts = [n.note_text for n in result.illustration_notes]
    assert len(note_texts) == 9
    assert "the other kids think he’s cool" in note_texts
    assert "turkeys are winning" in note_texts
    assert "now the kids are gobbling too" in note_texts

    for note in result.illustration_notes:
        assert 0 <= note.position_after_line < len(result.narration_lines)


def test_peter_rabbit_strips_gutenberg_boilerplate_and_illustration_markers():
    result = ingest(FIXTURES / "peter_rabbit" / "source.txt")

    joined = "\n".join(result.narration_lines)
    assert "Gutenberg" not in joined
    assert "[Illustration" not in joined
    assert "SAALFIELD" not in joined
    assert result.narration_lines[0].startswith("Once upon a time there were four little rabbits")


def test_peter_rabbit_preserves_one_word_per_line_pacing_device():
    result = ingest(FIXTURES / "peter_rabbit" / "source.txt")

    assert any(line == "He\nAte\nSome\nRadishes" for line in result.narration_lines)


def test_peter_rabbit_preserves_lippity_dash_pacing_device():
    result = ingest(FIXTURES / "peter_rabbit" / "source.txt")

    assert any("lippity--\nlippity--" in line for line in result.narration_lines)
