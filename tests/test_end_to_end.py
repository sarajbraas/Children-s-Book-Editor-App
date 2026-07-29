"""Full-pipeline (Phases 1-11) end-to-end tests against the three fixture
manuscripts. These are the only tests in the suite that run the ENTIRE live
pipeline -- ingest -> chunk -> verse -> book_profile -> scenes -> tag ->
humor -> illustration -> pronunciation -> reports -- and assemble the final
tagged-JSON manuscript, the same way a real production run would.

Because tagging is an LLM judgment call, byte-identical output vs.
expected.json is neither expected nor the bar. Per the project's own
philosophy (flag for human review rather than silently pass/fail on a
subjective call), this suite checks INVARIANTS the output must always
satisfy no matter which specific judgment call the model makes, and
separately reports large emotion-category divergences from the hand-tagged
fixture as a "needs human review" list rather than a hard test failure.

Each fixture's full pipeline run is expensive (dozens of live tag calls plus
scene/humor/illustration calls), so it runs ONCE per book behind a
module-scoped fixture and every assertion test below reuses that one run.
"""

import json
import os
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import anthropic

from core.book_profile import compute_book_profile
from core.chunker import chunk_narration
from core.humor import assign_humor_types, validate_coverage
from core.illustration import infer_illustrations
from core.ingest import ingest
from core.pronunciation import resolve_manuscript_pronunciations
from core.reports import (
    _build_scene_refs_by_chunk,
    build_tagged_manuscript,
    generate_analysis_document,
    write_analysis_document,
    write_tagged_manuscript,
)
from core.scenes import detect_scenes
from core.tagger import INTENSITY_LEVELS, VOLUME_LEVELS, load_taxonomy, tag_chunks
from core.verse import compute_verse

live_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- full end-to-end pipeline tests need real API credentials",
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
E2E_OUTPUT_DIR = Path(__file__).parent / "e2e_output"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

BOOKS = {
    "pharaoh_duck": dict(filename="source.docx", bookcode="pd", title="Pharaoh Duck", author="Sara Braas"),
    "turkey_takeover": dict(filename="source.docx", bookcode="tt", title="Turkey Takeover", author="Sara Braas"),
    "peter_rabbit": dict(filename="source.txt", bookcode="pr", title="The Tale of Peter Rabbit", author="Beatrix Potter"),
}


def _emotion_names() -> set[str]:
    taxonomy = load_taxonomy()
    return {e["name"] for e in taxonomy["emotions"]}


def _run_pipeline(book_dir: str) -> dict:
    """Runs Phases 1-11 in sequence for one fixture book and returns
    everything a test might want to inspect: the assembled manuscript dict,
    the intermediate chunks/book_profile, and every phase's own validation
    flags collected along the way."""
    spec = BOOKS[book_dir]
    client = anthropic.Anthropic()

    ingest_result = ingest(FIXTURES / book_dir / spec["filename"])
    chunks = chunk_narration(ingest_result.narration_lines, spec["bookcode"])

    verse = compute_verse([(c.id, [c.text]) for c in chunks])
    book_profile = compute_book_profile(ingest_result.narration_lines, verse.book_profile.verse_form)

    scene_detection = detect_scenes(client, chunks, book_profile)
    scene_refs_by_chunk = _build_scene_refs_by_chunk(scene_detection.scenes)

    tag_results = tag_chunks(client, chunks, book_profile, scene_refs_by_id=scene_refs_by_chunk)
    tags_by_id = {r.chunk_id: r.tag for r in tag_results}
    tag_validation_flags = {r.chunk_id: r.validation_flags for r in tag_results if r.validation_flags}

    humor_results = assign_humor_types(client, chunks, tags_by_id, scene_detection.scenes, book_profile)
    humor_by_id = {r.chunk_id: r.humor_type for r in humor_results}
    humor_validation_flags = validate_coverage(humor_results, chunks, tags_by_id, scene_detection.scenes)

    illustration_results = infer_illustrations(client, chunks, ingest_result.illustration_notes, book_profile)
    illustration_by_id = {r.chunk_id: r for r in illustration_results}

    pronunciation_report = resolve_manuscript_pronunciations(
        ingest_result.narration_lines, verse.oov_words, spec["title"]
    )

    manuscript = build_tagged_manuscript(
        title=spec["title"],
        author=spec["author"],
        note="",
        chunks=chunks,
        tags_by_id=tags_by_id,
        scenes=scene_detection.scenes,
        book_profile=book_profile,
        verse_profile=verse.book_profile,
        humor_by_id=humor_by_id,
        illustration_by_id=illustration_by_id,
        verse_by_id=verse.verse_by_chunk,
    )

    analysis_result = generate_analysis_document(client, manuscript)

    book_output_dir = OUTPUT_DIR / book_dir
    write_tagged_manuscript(manuscript, book_output_dir / "tagged.json")
    write_analysis_document(analysis_result.markdown, book_output_dir / "analysis.md")

    with open(FIXTURES / book_dir / "expected.json") as f:
        expected = json.load(f)

    return dict(
        manuscript=manuscript,
        chunks=chunks,
        book_profile=book_profile,
        pronunciation_report=pronunciation_report,
        scene_validation_flags=scene_detection.validation_flags,
        tag_validation_flags=tag_validation_flags,
        humor_validation_flags=humor_validation_flags,
        analysis_validation_flags=analysis_result.validation_flags,
        expected=expected,
    )


@pytest.fixture(scope="module")
def pharaoh_duck_run():
    return _run_pipeline("pharaoh_duck")


@pytest.fixture(scope="module")
def turkey_takeover_run():
    return _run_pipeline("turkey_takeover")


@pytest.fixture(scope="module")
def peter_rabbit_run():
    return _run_pipeline("peter_rabbit")


# ---------------------------------------------------------------------------
# Shared invariant checks -- applied to every book's assembled manuscript.
# ---------------------------------------------------------------------------


def _assert_valid_emotion_enums(manuscript: dict, emotion_names: set[str]) -> None:
    for chunk in manuscript["chunks"]:
        assert chunk["emotion"] in emotion_names, f"{chunk['id']}: invalid emotion {chunk['emotion']!r}"
        secondary = chunk.get("secondary_emotion")
        if secondary is not None:
            assert secondary in emotion_names, f"{chunk['id']}: invalid secondary_emotion {secondary!r}"
        assert chunk["intensity"] in INTENSITY_LEVELS, f"{chunk['id']}: invalid intensity {chunk['intensity']!r}"
        assert chunk["volume"] in VOLUME_LEVELS, f"{chunk['id']}: invalid volume {chunk['volume']!r}"


def _assert_ceilings_and_disabled_emotions_respected(manuscript: dict) -> None:
    er = manuscript["book_profile"]["emotional_range"]
    intensity_ceiling_idx = INTENSITY_LEVELS.index(er["intensity_ceiling"])
    volume_ceiling_idx = VOLUME_LEVELS.index(er["volume_ceiling"])
    disabled = set(er["disabled_emotions"])

    for chunk in manuscript["chunks"]:
        assert INTENSITY_LEVELS.index(chunk["intensity"]) <= intensity_ceiling_idx, (
            f"{chunk['id']}: intensity {chunk['intensity']!r} exceeds book ceiling {er['intensity_ceiling']!r}"
        )
        assert VOLUME_LEVELS.index(chunk["volume"]) <= volume_ceiling_idx, (
            f"{chunk['id']}: volume {chunk['volume']!r} exceeds book ceiling {er['volume_ceiling']!r}"
        )
        assert chunk["emotion"] not in disabled, f"{chunk['id']}: uses disabled emotion {chunk['emotion']!r}"
        secondary = chunk.get("secondary_emotion")
        assert secondary not in disabled, f"{chunk['id']}: uses disabled secondary_emotion {secondary!r}"


def _assert_sound_effects_correctly_tagged(manuscript: dict, chunks: list) -> None:
    chunk_by_id = {c["id"]: c for c in manuscript["chunks"]}
    for chunk in chunks:
        if chunk.type != "sound_effect":
            continue
        tagged = chunk_by_id[chunk.id]
        assert tagged.get("type") == "sound_effect", (
            f"{chunk.id}: chunker marked this a sound_effect but the tagged output does not "
            f"-- risk of scoring a creature's own sound as a narrator emotion"
        )
        assert tagged.get("speaker"), f"{chunk.id}: sound_effect chunk has no speaker set"


def _assert_cross_references_point_to_real_chunks(manuscript: dict) -> None:
    valid_ids = {c["id"] for c in manuscript["chunks"]}

    for scene in manuscript["scenes"]:
        for member in scene["members"]:
            assert member["chunk_id"] in valid_ids, (
                f"scene '{scene['scene_id']}' references unknown chunk_id {member['chunk_id']!r}"
            )

    for chunk in manuscript["chunks"]:
        for ref in chunk.get("scene_refs", []):
            assert ref["scene_id"] in {s["scene_id"] for s in manuscript["scenes"]}, (
                f"{chunk['id']}: scene_refs points to unknown scene_id {ref['scene_id']!r}"
            )
        verse = chunk.get("verse")
        if verse and verse.get("rhymes_with_chunk"):
            assert verse["rhymes_with_chunk"] in valid_ids, (
                f"{chunk['id']}: rhymes_with_chunk points to unknown chunk_id {verse['rhymes_with_chunk']!r}"
            )


def _assert_manuscript_invariants(run: dict, emotion_names: set[str]) -> None:
    manuscript = run["manuscript"]
    _assert_valid_emotion_enums(manuscript, emotion_names)
    _assert_ceilings_and_disabled_emotions_respected(manuscript)
    _assert_sound_effects_correctly_tagged(manuscript, run["chunks"])
    _assert_cross_references_point_to_real_chunks(manuscript)


def _collect_emotion_divergences(manuscript: dict, expected: dict) -> list[dict]:
    """Chunks where the automated emotion differs entirely from the hand-
    tagged fixture (not just a different intensity/volume) -- reported for
    human review, never a hard failure. The fixture is one considered
    judgment call, not the only correct answer."""
    expected_by_id = {c["id"]: c for c in expected["chunks"]}
    divergences = []
    for chunk in manuscript["chunks"]:
        expected_chunk = expected_by_id.get(chunk["id"])
        if expected_chunk is None:
            continue
        if chunk["emotion"] != expected_chunk["emotion"]:
            divergences.append({
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "expected_emotion": expected_chunk["emotion"],
                "actual_emotion": chunk["emotion"],
            })
    return divergences


def _write_divergence_report(book_dir: str, divergences: list[dict]) -> Path:
    E2E_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = E2E_OUTPUT_DIR / f"{book_dir}_emotion_divergences.json"
    with open(path, "w") as f:
        json.dump(divergences, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Pharaoh Duck
# ---------------------------------------------------------------------------


@live_api
def test_pharaoh_duck_pipeline_output_satisfies_invariants(pharaoh_duck_run):
    _assert_manuscript_invariants(pharaoh_duck_run, _emotion_names())


@live_api
def test_pharaoh_duck_pronunciation_resolves_known_tiers(pharaoh_duck_run):
    resolved_by_word = {r.word.lower(): r for r in pharaoh_duck_run["pronunciation_report"].resolved}
    assert resolved_by_word["ava"].status == "dictionary"
    # "Pharaoh" has a book-scoped lexicon entry, but the pipeline checks the
    # CMU dictionary first and "Pharaoh" is a real dictionary word, so it
    # resolves at the dictionary tier before the lexicon is ever consulted --
    # a known, previously-confirmed discrepancy from the original task text
    # (which expected lexicon), not a bug in this test.
    assert resolved_by_word["pharaoh"].status == "dictionary"


@live_api
def test_pharaoh_duck_emotion_divergences_report(pharaoh_duck_run):
    divergences = _collect_emotion_divergences(pharaoh_duck_run["manuscript"], pharaoh_duck_run["expected"])
    path = _write_divergence_report("pharaoh_duck", divergences)
    print(f"\npharaoh_duck: {len(divergences)} chunk(s) diverge from the fixture emotion -- see {path}")
    for d in divergences:
        print(f"  [{d['chunk_id']}] expected={d['expected_emotion']!r} actual={d['actual_emotion']!r} :: {d['text']!r}")


# ---------------------------------------------------------------------------
# Turkey Takeover
# ---------------------------------------------------------------------------


@live_api
def test_turkey_takeover_pipeline_output_satisfies_invariants(turkey_takeover_run):
    _assert_manuscript_invariants(turkey_takeover_run, _emotion_names())


@live_api
def test_turkey_takeover_pronunciation_resolves_known_tiers(turkey_takeover_run):
    resolved_by_word = {r.word.lower(): r for r in turkey_takeover_run["pronunciation_report"].resolved}
    assert resolved_by_word["caruncles"].status == "lexicon"
    assert resolved_by_word["snoods"].status == "lexicon"


@live_api
def test_turkey_takeover_emotion_divergences_report(turkey_takeover_run):
    divergences = _collect_emotion_divergences(turkey_takeover_run["manuscript"], turkey_takeover_run["expected"])
    path = _write_divergence_report("turkey_takeover", divergences)
    print(f"\nturkey_takeover: {len(divergences)} chunk(s) diverge from the fixture emotion -- see {path}")
    for d in divergences:
        print(f"  [{d['chunk_id']}] expected={d['expected_emotion']!r} actual={d['actual_emotion']!r} :: {d['text']!r}")


# ---------------------------------------------------------------------------
# Peter Rabbit
# ---------------------------------------------------------------------------


@live_api
def test_peter_rabbit_pipeline_output_satisfies_invariants(peter_rabbit_run):
    _assert_manuscript_invariants(peter_rabbit_run, _emotion_names())


@live_api
def test_peter_rabbit_pronunciation_resolves_known_tiers(peter_rabbit_run):
    resolved_by_word = {r.word.lower(): r for r in peter_rabbit_run["pronunciation_report"].resolved}
    flagged_lower = {w.lower() for w in peter_rabbit_run["pronunciation_report"].pronunciation_flags}

    for name in ("flopsy", "mopsy", "cottontail"):
        assert name in flagged_lower, f"expected {name!r} to be flagged for review (no dictionary/lexicon entry)"
        assert name not in resolved_by_word


@live_api
def test_peter_rabbit_intensity_ceiling_is_high_with_volume_capped_below_shout(peter_rabbit_run):
    # The case most likely to get flattened by a classifier that conflates a
    # restrained, literary narrator with low narrative stakes: real danger
    # (McGregor's chase, near-capture) means intensity_ceiling must be
    # "high" even though the composed narratorial delivery caps volume at
    # "loud" rather than "shout".
    er = peter_rabbit_run["manuscript"]["book_profile"]["emotional_range"]
    assert er["intensity_ceiling"] == "high"
    assert er["volume_ceiling"] != "shout"
    assert VOLUME_LEVELS.index(er["volume_ceiling"]) < VOLUME_LEVELS.index("shout")


@live_api
def test_peter_rabbit_emotion_divergences_report(peter_rabbit_run):
    divergences = _collect_emotion_divergences(peter_rabbit_run["manuscript"], peter_rabbit_run["expected"])
    path = _write_divergence_report("peter_rabbit", divergences)
    print(f"\npeter_rabbit: {len(divergences)} chunk(s) diverge from the fixture emotion -- see {path}")
    for d in divergences:
        print(f"  [{d['chunk_id']}] expected={d['expected_emotion']!r} actual={d['actual_emotion']!r} :: {d['text']!r}")
