import json
import re
from pathlib import Path

from core.chunker import chunk_narration
from core.ingest import ingest

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_chunk_ids_are_sequential_and_zero_padded():
    chunks = chunk_narration(["First line.", "Second line.", "Third line."], "xx")
    assert [c.id for c in chunks] == ["xx-001", "xx-002", "xx-003"]


def test_bracketed_background_aside_splits_into_its_own_chunk():
    lines = ['And just like that, she’s on the phone. [“...no, it’s a feral duck,” Mom says.]']
    chunks = chunk_narration(lines, "pd")
    assert [c.text for c in chunks] == [
        "And just like that, she’s on the phone.",
        "“...no, it’s a feral duck,” Mom says.",
    ]


def test_bracket_only_line_becomes_single_chunk_with_brackets_stripped():
    lines = ['[“You have a Waterfowl Center?” Mom asks into her phone]']
    chunks = chunk_narration(lines, "pd")
    assert len(chunks) == 1
    assert "[" not in chunks[0].text and "]" not in chunks[0].text


def test_adjacent_identical_interjections_merge_into_one_chunk():
    chunks = chunk_narration(["Meep Meep Meep", "Meep Meep Meep", "Ava tugs at her mom's shirt."], "pd")
    assert len(chunks) == 2
    assert chunks[0].text == "Meep Meep Meep. Meep Meep Meep."
    assert chunks[0].type == "sound_effect"


def test_isolated_interjection_does_not_merge_with_unrelated_neighbor():
    chunks = chunk_narration(["Mama stands and squints.", "Meep Meep Meep", "Ava shouts for help."], "pd")
    assert len(chunks) == 3
    assert chunks[1].text == "Meep Meep Meep"
    assert chunks[1].type == "sound_effect"


def test_onomatopoeia_chorus_speaker_left_unknown_rather_than_guessed():
    # A repeated chorus is too ambiguous (often non-human) to safely
    # attribute to whichever character happened to be named most recently.
    chunks = chunk_narration(["Mama stands and squints.", "Meep Meep Meep"], "pd")
    assert chunks[1].type == "sound_effect"
    assert chunks[1].speaker is None


def test_bodily_sound_interjection_gets_speaker_from_recent_context():
    # A candidate name needs 2+ mentions across the book before it's trusted
    # as a real character name (guards against stray capitalized words).
    lines = [
        "Terrible Tom is a riot at school.",
        "So Tom must disrupt it, from when the bell dings.",
        "A fart…",
    ]
    chunks = chunk_narration(lines, "tt")
    assert chunks[-1].type == "sound_effect"
    assert chunks[-1].speaker == "Tom"


def test_inline_onomatopoeia_detected_without_splitting_the_sentence():
    chunks = chunk_narration(['Presently Peter sneezed "Kertyschoo!"'], "pr")
    assert len(chunks) == 1
    assert "Kertyschoo" in chunks[0].inline_sound_effects


def test_hyphen_fragmented_sound_word_detected():
    chunks = chunk_narration(["he heard the noise of a hoe--scr-r-ritch, scratch, scratch, scritch."], "pr")
    assert any("scr-r-ritch" in hit for hit in chunks[0].inline_sound_effects)


def test_informal_slang_words_are_not_mistaken_for_onomatopoeia():
    chunks = chunk_narration(['"It might be germy," Mama says.', '"Come here, Pharaoh Duckie."'], "pd")
    assert chunks[0].inline_sound_effects == []
    assert chunks[1].inline_sound_effects == []


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _completely_missed_boundaries(mine_texts, expected_texts):
    """Expected chunks whose text can't be reconstructed from a contiguous
    run of my chunker's output, in order. A chunker that groups sentences
    differently than the fixture (a subjective, non-lossy choice) does NOT
    count as a miss here -- only content that doesn't line up at all does.
    """
    mine_norm = [_normalize(c) for c in mine_texts]
    misses = []
    mi = 0
    for exp_text in expected_texts:
        exp_norm = _normalize(exp_text)
        if mi < len(mine_norm) and mine_norm[mi] == exp_norm:
            mi += 1
            continue
        acc, j, matched = "", mi, False
        while j < len(mine_norm) and len(acc) <= len(exp_norm) + 10:
            acc = (acc + " " + mine_norm[j]).strip() if acc else mine_norm[j]
            j += 1
            if acc == exp_norm:
                matched = True
                break
        if matched:
            mi = j
            continue
        found_at = next((k for k in range(mi, min(mi + 6, len(mine_norm))) if mine_norm[k] == exp_norm), None)
        if found_at is not None:
            mi = found_at + 1
            continue
        misses.append(exp_text)
    return misses


def _load_expected_texts(name):
    with open(FIXTURES / name / "expected.json") as f:
        return [c["text"] for c in json.load(f)["chunks"]]


def test_turkey_takeover_no_completely_missed_boundaries():
    result = ingest(FIXTURES / "turkey_takeover" / "source.docx")
    chunks = chunk_narration(result.narration_lines, "tt")
    misses = _completely_missed_boundaries([c.text for c in chunks], _load_expected_texts("turkey_takeover"))
    assert misses == []


def test_peter_rabbit_no_completely_missed_boundaries():
    result = ingest(FIXTURES / "peter_rabbit" / "source.txt")
    chunks = chunk_narration(result.narration_lines, "pr")
    misses = _completely_missed_boundaries([c.text for c in chunks], _load_expected_texts("peter_rabbit"))
    assert misses == []


def test_pharaoh_duck_only_known_subjective_misses():
    # One known, unresolvable divergence remains: a hand-corrected manuscript
    # typo (missing closing quote) that no structural signal could detect.
    # The tender-cuddle/hand-washing dry-irony split (previously a second
    # known miss) is now handled by _split_tonal_shift_beats.
    result = ingest(FIXTURES / "pharaoh_duck" / "source.docx")
    chunks = chunk_narration(result.narration_lines, "pd")
    misses = _completely_missed_boundaries([c.text for c in chunks], _load_expected_texts("pharaoh_duck"))
    assert misses == ['"...great, yes, we will be here..." Mom says, still on the phone.']


def test_tonal_shift_splits_warm_beat_from_mundane_deadpan_followup():
    chunks = chunk_narration(
        ["Mama gives Ava and Pharaoh Duck a quick cuddle too. Then they wash their hands very well."],
        "pd",
    )
    assert [c.text for c in chunks] == [
        "Mama gives Ava and Pharaoh Duck a quick cuddle too.",
        "Then they wash their hands very well.",
    ]


def test_tonal_shift_does_not_split_when_second_sentence_is_also_warm():
    chunks = chunk_narration(
        ["Mama gives Ava a quick cuddle too. Then she hugs Pharaoh Duck as well."],
        "pd",
    )
    assert len(chunks) == 1
