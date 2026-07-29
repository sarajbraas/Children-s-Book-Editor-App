import json
from pathlib import Path

from core.chunker import chunk_narration
from core.ingest import ingest
from core.verse import _fallback_syllable_count, _rhymes, compute_verse

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Ground-truth grouping of Turkey Takeover's 73 ingested narration lines into
# the 49 chunks expected.json actually uses (verified by hand against the
# fixture's chunk text during the chunker phase -- chunker.py doesn't
# reproduce this couplet-driven merging itself, verse-scheme grouping is out
# of its scope, but verse.py's own line/rhyme logic can still be validated
# directly against the fixture's real chunk boundaries using this mapping).
TURKEY_TAKEOVER_GROUPS = [
    [0, 1], [2], [3], [4], [5], [6], [7], [8, 9], [10], [11], [12], [13],
    [14, 15], [16, 17], [18], [19], [20, 21], [22, 23], [24, 25], [26],
    [27, 28], [29, 30], [31, 32], [33, 34], [35, 36], [37], [38], [39, 40],
    [41], [42], [43], [44, 45], [46], [47, 48], [49, 50], [51, 52], [53, 54],
    [55], [56], [57], [58, 59], [60, 61], [62, 63], [64, 65], [66], [67, 68],
    [69, 70], [71], [72],
]


def _turkey_takeover_chunk_lines():
    result = ingest(FIXTURES / "turkey_takeover" / "source.docx")
    lines = result.narration_lines
    assert len(lines) == 73
    return [(f"tt-{i + 1:03d}", [lines[idx] for idx in group]) for i, group in enumerate(TURKEY_TAKEOVER_GROUPS)]


def _expected_chunks(name):
    with open(FIXTURES / name / "expected.json") as f:
        return json.load(f)["chunks"]


def test_turkey_takeover_verse_matches_expected_json():
    verse_result = compute_verse(_turkey_takeover_chunk_lines())

    mismatches = {}
    for c in _expected_chunks("turkey_takeover"):
        cid = c["id"]
        expected = c["verse"]
        mine = verse_result.verse_by_chunk[cid]
        diffs = {}
        if mine.syllables_per_line != expected["syllables_per_line"]:
            diffs["syllables_per_line"] = (mine.syllables_per_line, expected["syllables_per_line"])
        if mine.meter_break != expected["meter_break"]:
            diffs["meter_break"] = (mine.meter_break, expected["meter_break"])
        if mine.rhyme_role != expected["rhyme_role"]:
            diffs["rhyme_role"] = (mine.rhyme_role, expected["rhyme_role"])
        if mine.rhymes_with_chunk != expected.get("rhymes_with_chunk"):
            diffs["rhymes_with_chunk"] = (mine.rhymes_with_chunk, expected.get("rhymes_with_chunk"))
        if diffs:
            mismatches[cid] = diffs

    # Only known, explained divergence: the fixture undercounts tt-008's
    # second line by one syllable (CMU dict independently confirms 11, not
    # 10, for "In Music he tickles. He whistles in Art.") -- a fixture typo,
    # not a detection bug.
    assert mismatches == {"tt-008": {"syllables_per_line": ([11, 11], [11, 10])}}


def test_turkey_takeover_book_profile_is_verse():
    verse_result = compute_verse(_turkey_takeover_chunk_lines())
    profile = verse_result.book_profile
    assert profile.verse_form is True
    assert profile.rhyme_scheme == "AABB couplets"
    assert profile.meter_break_count == 7  # tt-003, 009, 020, 030, 033, 040, 045


def test_turkey_takeover_oov_words_include_known_gaps():
    verse_result = compute_verse(_turkey_takeover_chunk_lines())
    for word in ["casseroles", "cubbyholes", "preens"]:
        assert word in verse_result.oov_words


def _prose_chunk_lines(name, fname, code):
    result = ingest(FIXTURES / name / fname)
    chunks = chunk_narration(result.narration_lines, code)
    return [(c.id, [c.text]) for c in chunks]


def test_pharaoh_duck_resolves_to_prose():
    verse_result = compute_verse(_prose_chunk_lines("pharaoh_duck", "source.docx", "pd"))
    assert verse_result.book_profile.verse_form is False
    assert verse_result.book_profile.rhyme_scheme is None


def test_peter_rabbit_resolves_to_prose():
    verse_result = compute_verse(_prose_chunk_lines("peter_rabbit", "source.txt", "pr"))
    assert verse_result.book_profile.verse_form is False
    assert verse_result.book_profile.rhyme_scheme is None


def test_peter_rabbit_oov_words_include_known_gaps():
    verse_result = compute_verse(_prose_chunk_lines("peter_rabbit", "source.txt", "pr"))
    for word in ["Flopsy", "Mopsy", "Cottontail"]:
        assert word in verse_result.oov_words


def test_fallback_syllable_counter_does_not_crash_on_oov_words():
    # The vowel-group heuristic is approximate (e.g. it overcounts "-es"
    # plurals like "casseroles" by one) -- these words get flagged as OOV
    # for Phase 6 review regardless, so exact linguistic accuracy isn't the
    # bar here, just a sane, non-crashing, non-zero estimate.
    for word, expected in [("preens", 1), ("Flopsy", 2), ("Mopsy", 2)]:
        assert _fallback_syllable_count(word) == expected
    for word in ["casseroles", "cubbyholes"]:
        assert 1 <= _fallback_syllable_count(word) <= 5


def test_fallback_syllable_counter_never_returns_zero_for_real_words():
    assert _fallback_syllable_count("gobbledy") >= 1


def test_near_rhyme_eleanor_door_tolerated():
    oov = set()
    assert _rhymes("Eleanor", "door", oov) is True


def test_near_rhyme_chin_in_tolerated():
    oov = set()
    assert _rhymes("chin", "in", oov) is True


def test_plural_slant_rhyme_tolerated():
    oov = set()
    assert _rhymes("chests", "test", oov) is True


def test_non_rhyming_words_are_not_falsely_matched():
    oov = set()
    assert _rhymes("school", "banana", oov) is False
