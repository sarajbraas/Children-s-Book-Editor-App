import json
from pathlib import Path

from core.book_profile import compute_book_profile
from core.chunker import chunk_narration
from core.ingest import ingest
from core.verse import compute_verse

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _profile_for(name, fname, code):
    result = ingest(FIXTURES / name / fname)
    chunks = chunk_narration(result.narration_lines, code)
    verse = compute_verse([(c.id, [c.text]) for c in chunks])
    return compute_book_profile(result.narration_lines, verse.book_profile.verse_form)


def _expected_book_profile(name):
    with open(FIXTURES / name / "expected.json") as f:
        return json.load(f)["book_profile"]


def test_pharaoh_duck_genre_style_and_emotional_ceiling():
    profile = _profile_for("pharaoh_duck", "source.docx", "pd")
    expected = _expected_book_profile("pharaoh_duck")
    assert profile.genre == expected["genre"]
    assert profile.style_era == expected["style_era"]
    assert profile.emotional_range.intensity_ceiling == expected["emotional_range"]["intensity_ceiling"]
    assert profile.emotional_range.volume_ceiling == expected["emotional_range"]["volume_ceiling"]


def test_turkey_takeover_genre_style_and_emotional_ceiling():
    profile = _profile_for("turkey_takeover", "source.docx", "tt")
    expected = _expected_book_profile("turkey_takeover")
    assert profile.genre == expected["genre"]
    assert profile.style_era == expected["style_era"]
    assert profile.emotional_range.intensity_ceiling == expected["emotional_range"]["intensity_ceiling"]
    assert profile.emotional_range.volume_ceiling == expected["emotional_range"]["volume_ceiling"]


def test_peter_rabbit_genre_style_and_emotional_ceiling():
    # The key edge case: classic_literary's restrained narrator voice must
    # not get conflated with low stakes -- real fear/danger content in the
    # text should still push intensity_ceiling to "high" independent of the
    # narrator-register signal that drives style_era/volume_ceiling.
    profile = _profile_for("peter_rabbit", "source.txt", "pr")
    expected = _expected_book_profile("peter_rabbit")
    assert profile.genre == expected["genre"]
    assert profile.style_era == expected["style_era"]
    assert profile.emotional_range.intensity_ceiling == "high"
    assert profile.emotional_range.intensity_ceiling == expected["emotional_range"]["intensity_ceiling"]
    assert profile.emotional_range.volume_ceiling == expected["emotional_range"]["volume_ceiling"]


def test_peter_rabbit_narrator_stance_mentions_restraint():
    # Full hand-written prose descriptions aren't reproducible by a rule-based
    # classifier -- check the calibration-anchor stance we fall back to is at
    # least thematically aligned (omniscient/measured), not an exact string match.
    profile = _profile_for("peter_rabbit", "source.txt", "pr")
    stance = profile.narrator_stance.lower()
    assert "omniscient" in stance or "measured" in stance


def test_recurring_antagonist_flagged_for_review_not_auto_disabled():
    # Both Turkey Takeover (the turkeys) and Peter Rabbit (Mr. McGregor) have
    # a recurring antagonist/disruptor -- whether that should disable
    # Villainous/Menacing is a real judgment call (comic chaos vs. genuine
    # threat) the classifier deliberately doesn't make on its own. It should
    # flag both for human review while leaving disabled_emotions untouched
    # (safe default: enabled), rather than guessing either way.
    for name, fname, code in [("turkey_takeover", "source.docx", "tt"), ("peter_rabbit", "source.txt", "pr")]:
        profile = _profile_for(name, fname, code)
        assert profile.flags_for_review, f"{name}: expected an antagonist review flag"
        assert "Villainous/Menacing" not in profile.emotional_range.disabled_emotions


def test_no_antagonist_no_flag():
    profile = _profile_for("pharaoh_duck", "source.docx", "pd")
    assert profile.flags_for_review == []


def test_favored_humor_types_overlap_with_fixture():
    # Real humor-type classification needs per-chunk content tagging (Phase
    # 5, not yet built) -- book_profile can only offer the closest anchor's
    # list. Check meaningful overlap rather than an exact match.
    for name, fname, code in [
        ("pharaoh_duck", "source.docx", "pd"),
        ("turkey_takeover", "source.docx", "tt"),
        ("peter_rabbit", "source.txt", "pr"),
    ]:
        profile = _profile_for(name, fname, code)
        expected = _expected_book_profile(name)
        mine = set(profile.emotional_range.favored_humor_types)
        theirs = set(expected["emotional_range"]["favored_humor_types"])
        assert mine & theirs, f"{name}: no overlap between {mine} and {theirs}"


def test_expository_nonfiction_detected_from_generic_definitional_text():
    lines = [
        "Ducks are waterfowl that live near ponds and lakes.",
        "A duck is a bird with webbed feet and a broad bill.",
        "Ducks eat plants, insects, and small fish.",
        "Most ducks migrate south for the winter.",
        "A group of ducks is called a raft.",
    ]
    profile = compute_book_profile(lines, verse_form=False)
    assert profile.genre == "expository_nonfiction"
    assert "Villainous/Menacing" in profile.emotional_range.disabled_emotions


def test_oral_fable_detected_from_archetype_and_closing_moral():
    lines = [
        "The Fox saw a Crow sitting in a tree with a piece of cheese.",
        '"How beautiful you are," said the Fox, "surely your voice is just as fine."',
        "The Crow opened her beak to sing, and the cheese fell to the ground.",
        "And so the Crow learned never to trust a flatterer.",
    ]
    profile = compute_book_profile(lines, verse_form=False)
    assert profile.style_era == "oral_fable"


def test_direct_reader_address_signals_classic_literary():
    lines = [
        "I think he might have gotten away if he had only been more careful.",
        "The little rabbit ran as fast as his legs could carry him through the garden.",
    ]
    profile = compute_book_profile(lines, verse_form=False)
    assert profile.style_era == "classic_literary"
