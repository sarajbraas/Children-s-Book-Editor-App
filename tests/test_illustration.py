import os
from pathlib import Path

import pytest

from core.chunker import Chunk
from core.ingest import IllustrationNote, ingest
from core.chunker import chunk_narration
from core.illustration import (
    VISUAL_ENERGY_LEVELS,
    ChunkIllustration,
    _concrete_density,
    _find_anaphoric_run_chunk_ids,
    _heuristic_hint,
    map_author_notes_to_chunks,
    validate,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

live_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- live illustration tests need real API credentials",
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---- map_author_notes_to_chunks (no API needed) ----


def test_maps_note_to_the_single_chunk_derived_from_that_line():
    chunks = [Chunk(id="x-001", text="a line", source_lines=[0]), Chunk(id="x-002", text="b line", source_lines=[1])]
    notes = [IllustrationNote(position_after_line=0, note_text="a note")]
    mapping = map_author_notes_to_chunks(chunks, notes)
    assert mapping == {"x-001": "a note"}


def test_maps_note_to_the_last_chunk_when_a_line_was_split():
    # A single original line split (bracket/tonal-shift) into two chunks --
    # the note belongs to the tail end, so the LAST resulting chunk.
    chunks = [
        Chunk(id="x-001", text="first half", source_lines=[0]),
        Chunk(id="x-002", text="second half", source_lines=[0]),
    ]
    notes = [IllustrationNote(position_after_line=0, note_text="a note")]
    mapping = map_author_notes_to_chunks(chunks, notes)
    assert mapping == {"x-002": "a note"}


def test_maps_note_correctly_when_two_lines_were_merged_into_one_chunk():
    chunks = [Chunk(id="x-001", text="merged chunk", source_lines=[0, 1])]
    notes = [IllustrationNote(position_after_line=1, note_text="a note")]
    mapping = map_author_notes_to_chunks(chunks, notes)
    assert mapping == {"x-001": "a note"}


def test_turkey_takeover_all_nine_author_notes_map_correctly():
    # Pure ingest+chunk pipeline, no API needed -- exercises the real
    # provenance-tracking path against real fixture data.
    result = ingest(FIXTURES / "turkey_takeover" / "source.docx")
    chunks = chunk_narration(result.narration_lines, "tt")
    mapping = map_author_notes_to_chunks(chunks, result.illustration_notes)

    assert len(mapping) == len(result.illustration_notes) == 9
    assert mapping["tt-047"] == "turkeys are winning"
    assert mapping["tt-058"] == "now the kids are gobbling too"
    assert "cool" in mapping["tt-002"]


# ---- heuristic gating (no API needed) ----


def test_anaphoric_run_detects_repeated_leading_subject():
    chunks = [
        Chunk(id="x-001", text="The court entertains their Duck King."),
        Chunk(id="x-002", text="The court promenades with their Duck King."),
        Chunk(id="x-003", text="The court keeps their Duck King safe."),
        Chunk(id="x-004", text="Something unrelated happens next."),
    ]
    run_ids = _find_anaphoric_run_chunk_ids(chunks)
    assert run_ids == {"x-001", "x-002", "x-003"}


def test_anaphoric_run_requires_at_least_two_consecutive_matches():
    chunks = [
        Chunk(id="x-001", text="The court entertains their Duck King."),
        Chunk(id="x-002", text="Something else entirely."),
    ]
    assert _find_anaphoric_run_chunk_ids(chunks) == set()


def test_sound_effect_chunks_are_never_candidates_even_if_dense():
    chunk = Chunk(id="x-001", text="Meep Meep Meep. Meep Meep Meep.", type="sound_effect")
    assert _heuristic_hint(chunk, set()) is None


def test_long_descriptive_sentence_is_not_a_candidate():
    chunk = Chunk(
        id="x-001",
        text=(
            "He was so tired that he flopped down upon the nice soft sand on the floor of the "
            "rabbit hole, and shut his eyes, his mother busy cooking nearby."
        ),
    )
    assert _heuristic_hint(chunk, set()) is None


def test_short_line_in_anaphoric_run_is_a_candidate():
    chunk = Chunk(id="x-001", text="The court entertains their Duck King.")
    hint = _heuristic_hint(chunk, {"x-001"})
    assert hint is not None
    assert "repeated subject" in hint or "template" in hint


def test_concrete_density_of_plain_line_is_low():
    assert _concrete_density("They want him to sit. They want him to learn.") < 0.15


# ---- validate() (no API needed) ----


def test_validate_flags_missing_author_note_mapping():
    results = [ChunkIllustration(chunk_id="x-001", implied_scene="a note", source="author")]
    flags = validate(results, author_notes_count=2)
    assert any("expected 2" in f for f in flags)


def test_validate_flags_author_source_with_visual_energy_set():
    results = [
        ChunkIllustration(chunk_id="x-001", implied_scene="a note", source="author", visual_energy="high")
    ]
    flags = validate(results, author_notes_count=1)
    assert any("author-sourced but has visual_energy set" in f for f in flags)


def test_validate_flags_inferred_with_invalid_visual_energy():
    results = [
        ChunkIllustration(chunk_id="x-001", implied_scene="a scene", source="inferred", visual_energy="extreme")
    ]
    flags = validate(results, author_notes_count=0)
    assert any("invalid visual_energy" in f for f in flags)


def test_validate_clean_when_correct():
    results = [
        ChunkIllustration(chunk_id="x-001", implied_scene="a note", source="author"),
        ChunkIllustration(chunk_id="x-002", implied_scene="a scene", source="inferred", visual_energy="medium"),
    ]
    flags = validate(results, author_notes_count=1)
    assert flags == []


# ---- live tests ----


@live_api
def test_pharaoh_duck_royal_court_sequence_matches_fixture_visual_energy():
    import anthropic

    from core.book_profile import compute_book_profile
    from core.illustration import infer_illustrations
    from core.verse import compute_verse

    result = ingest(FIXTURES / "pharaoh_duck" / "source.docx")
    chunks = chunk_narration(result.narration_lines, "pd")
    verse = compute_verse([(c.id, [c.text]) for c in chunks])
    profile = compute_book_profile(result.narration_lines, verse.book_profile.verse_form)

    client = anthropic.Anthropic()
    results = infer_illustrations(client, chunks, result.illustration_notes, profile)
    by_id = {r.chunk_id: r for r in results}

    # Fixture's actual visual_energy for this exact anaphoric run: high, high,
    # medium, medium -- a graduated decrease as the "royal court" bit winds down.
    expected_energy = {"pd-019": "high", "pd-020": "high", "pd-021": "medium", "pd-022": "medium"}
    for chunk_id, expected in expected_energy.items():
        assert chunk_id in by_id, f"expected {chunk_id} to be inferred"
        assert by_id[chunk_id].source == "inferred"
        assert by_id[chunk_id].visual_energy == expected
        assert len(by_id[chunk_id].implied_scene) > 20  # a real sentence, not a stub

    assert validate(results, len(result.illustration_notes)) == []


@live_api
def test_peter_rabbit_no_signal_chunks_correctly_omit_the_field():
    # A Gutenberg public-domain text: no author illustration notes at all
    # (bare "[Illustration]" placeholders were already stripped in Phase 1
    # with nothing descriptive to preserve), and classic_literary prose is
    # mostly long, already-descriptive sentences -- exactly the case that
    # should produce NO field rather than a forced low-confidence guess.
    import anthropic

    from core.book_profile import compute_book_profile
    from core.illustration import infer_illustrations
    from core.verse import compute_verse

    result = ingest(FIXTURES / "peter_rabbit" / "source.txt")
    chunks = chunk_narration(result.narration_lines, "pr")
    verse = compute_verse([(c.id, [c.text]) for c in chunks])
    profile = compute_book_profile(result.narration_lines, verse.book_profile.verse_form)
    assert result.illustration_notes == []

    client = anthropic.Anthropic()
    results = infer_illustrations(client, chunks, result.illustration_notes, profile)
    covered_ids = {r.chunk_id for r in results}

    # The 5 genuine candidates -- short action-fragment lines -- should be covered.
    expected_candidates = {"pr-011", "pr-014", "pr-030", "pr-033", "pr-035"}
    assert expected_candidates <= covered_ids
    assert all(r.source == "inferred" for r in results if r.chunk_id in expected_candidates)

    # Long, already-descriptive prose should have NO entry at all -- not a
    # low-confidence guess.
    no_signal_ids = {"pr-001", "pr-002", "pr-003", "pr-006", "pr-008"}
    assert no_signal_ids.isdisjoint(covered_ids)

    assert len(results) < len(chunks) / 2  # most of this book's prose has no signal
    assert validate(results, len(result.illustration_notes)) == []


@live_api
def test_turkey_takeover_author_notes_never_regenerated_through_full_pipeline():
    import anthropic

    from core.book_profile import compute_book_profile
    from core.illustration import infer_illustrations
    from core.verse import compute_verse

    result = ingest(FIXTURES / "turkey_takeover" / "source.docx")
    chunks = chunk_narration(result.narration_lines, "tt")
    verse = compute_verse([(c.id, [c.text]) for c in chunks])
    profile = compute_book_profile(result.narration_lines, verse.book_profile.verse_form)

    client = anthropic.Anthropic()
    results = infer_illustrations(client, chunks, result.illustration_notes, profile)

    author_results = {r.chunk_id: r for r in results if r.source == "author"}
    assert len(author_results) == 9
    assert author_results["tt-047"].implied_scene == "turkeys are winning"
    assert author_results["tt-047"].visual_energy is None
    assert all(r.visual_energy is None for r in author_results.values())

    assert validate(results, len(result.illustration_notes)) == []
