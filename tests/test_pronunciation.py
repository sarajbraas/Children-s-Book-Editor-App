from pathlib import Path

from core.chunker import chunk_narration
from core.ingest import ingest
from core.pronunciation import (
    PronunciationResolution,
    resolve_manuscript_pronunciations,
    resolve_word,
    validate,
)
from core.verse import compute_verse

FIXTURES = Path(__file__).parent.parent / "fixtures"

TEST_LEXICON = {
    "entries": [
        {"word": "Pharaoh", "respelling": "FAIR-oh", "ipa": "ˈfɛr.oʊ", "scope": "book:Pharaoh Duck"},
        {"word": "caruncles", "respelling": "kuh-RUNK-uhlz", "ipa": "kəˈrʌŋ.kəlz", "scope": "book:Turkey Takeover"},
        {"word": "Ava", "respelling": "AY-vuh", "ipa": "ˈeɪ.və", "scope": "global"},
    ]
}


# ---- step ordering: dictionary always wins first ----


def test_dictionary_resolves_before_lexicon_is_even_checked():
    # "Ava" is in both CMU dict and the test lexicon -- dictionary must win.
    result = resolve_word("Ava", "Some Book", TEST_LEXICON)
    assert result.status == "dictionary"
    assert result.respelling is None


def test_lexicon_resolves_oov_word_with_matching_book_scope():
    result = resolve_word("caruncles", "Turkey Takeover", TEST_LEXICON)
    assert result.status == "lexicon"
    assert result.respelling == "kuh-RUNK-uhlz"


def test_book_scoped_entry_does_not_apply_to_a_different_book():
    result = resolve_word("caruncles", "Pharaoh Duck", TEST_LEXICON)
    assert result is None  # OOV, no matching scope -- caller flags it


def test_global_scoped_entry_applies_to_any_book():
    lexicon = {"entries": [{"word": "Zibbo", "respelling": "ZIB-oh", "ipa": "x", "scope": "global"}]}
    for book in ["Book A", "Book B", "Anything"]:
        result = resolve_word("Zibbo", book, lexicon)
        assert result.status == "lexicon"


def test_unresolved_word_returns_none_for_caller_to_flag():
    result = resolve_word("Flopsy", "The Tale of Peter Rabbit", TEST_LEXICON)
    assert result is None


# ---- apostrophe normalization and possessive fallback ----


def test_curly_apostrophe_possessive_resolves_via_dictionary():
    # "tom's" is a real CMU dict entry, but only with a straight apostrophe --
    # manuscripts use curly quotes throughout.
    result = resolve_word("Tom’s", "Some Book", {"entries": []})
    assert result is not None
    assert result.status == "dictionary"


def test_possessive_of_dictionary_name_inherits_base_resolution():
    # "ava's" isn't its own CMU dict entry, but "ava" is.
    result = resolve_word("Ava’s", "Some Book", {"entries": []})
    assert result is not None
    assert result.status == "dictionary"
    assert "possessive" in (result.note or "")


def test_possessive_of_lexicon_name_inherits_lexicon_resolution():
    # "Pharaoh" itself resolves via dictionary (step 1), so it's a bad
    # example here -- use a genuinely dictionary-unresolvable invented name
    # that only the lexicon knows, even in its base form.
    lexicon = {"entries": [{"word": "Zibbowitz", "respelling": "ZIB-oh-witz", "ipa": "x", "scope": "global"}]}
    result = resolve_word("Zibbowitz’s", "Some Book", lexicon)
    assert result is not None
    assert result.status == "lexicon"
    assert result.respelling.startswith("ZIB-oh-witz")


# ---- validate() ----


def test_validate_flags_word_in_both_resolved_and_flagged():
    from core.pronunciation import PronunciationReport

    report = PronunciationReport(
        resolved=[PronunciationResolution(word="Flopsy", status="dictionary")],
        pronunciation_flags=["Flopsy"],
    )
    flags = validate(report)
    assert any("both resolved and pronunciation_flags" in f for f in flags)


def test_validate_flags_lexicon_entry_missing_respelling():
    from core.pronunciation import PronunciationReport

    report = PronunciationReport(
        resolved=[PronunciationResolution(word="X", status="lexicon", respelling=None, ipa=None)],
    )
    flags = validate(report)
    assert any("missing respelling/ipa" in f for f in flags)


def test_validate_clean_report_has_no_flags():
    from core.pronunciation import PronunciationReport

    report = PronunciationReport(
        resolved=[
            PronunciationResolution(word="Ava", status="dictionary"),
            PronunciationResolution(word="caruncles", status="lexicon", respelling="x", ipa="y"),
        ],
        pronunciation_flags=["Flopsy"],
    )
    assert validate(report) == []


# ---- integration against the three fixtures ----


def _report_for(name, fname, code, title):
    result = ingest(FIXTURES / name / fname)
    chunks = chunk_narration(result.narration_lines, code)
    verse = compute_verse([(c.id, [c.text]) for c in chunks])
    return resolve_manuscript_pronunciations(result.narration_lines, verse.oov_words, title)


def test_pharaoh_duck_ava_resolves_via_dictionary_alone():
    report = _report_for("pharaoh_duck", "source.docx", "pd", "Pharaoh Duck")
    ava = next(r for r in report.resolved if r.word == "Ava")
    assert ava.status == "dictionary"
    assert validate(report) == []


def test_pharaoh_duck_pharaoh_resolves_via_dictionary_not_lexicon():
    # CMU dict already has "pharaoh" natively (confirmed directly:
    # pronouncing.phones_for_word("pharaoh") -> non-empty), even though the
    # lexicon also carries a "Pharaoh" entry (added to lock in a consistent
    # reading across the book). Per the pipeline's own step order --
    # dictionary first, "no human involvement needed" when it resolves --
    # that lexicon entry is correctly never reached. The pipeline should not
    # be bent to route around a real, working dictionary hit just to land on
    # a specific step.
    report = _report_for("pharaoh_duck", "source.docx", "pd", "Pharaoh Duck")
    pharaoh = next(r for r in report.resolved if r.word == "Pharaoh")
    assert pharaoh.status == "dictionary"
    assert pharaoh.respelling is None
    assert validate(report) == []


def test_turkey_takeover_caruncles_and_snoods_resolve_via_lexicon():
    # Unlike "Pharaoh" (see test above), these two are genuinely absent from
    # CMU dict (confirmed directly: pronouncing.phones_for_word returns []
    # for both) -- they actually exercise the step-2 lexicon path, rather
    # than a dictionary hit short-circuiting it.
    report = _report_for("turkey_takeover", "source.docx", "tt", "Turkey Takeover")
    by_word = {r.word: r for r in report.resolved}
    assert by_word["caruncles"].status == "lexicon"
    assert by_word["caruncles"].respelling == "kuh-RUNK-uhlz"
    assert by_word["snoods"].status == "lexicon"
    assert by_word["snoods"].respelling == "SNOODZ"
    assert validate(report) == []


def test_peter_rabbit_invented_names_are_flagged_not_guessed():
    # Flopsy, Mopsy, Cottontail are confirmed CMU-dict gaps with no lexicon
    # entry -- must be flagged for human review, never silently guessed.
    report = _report_for("peter_rabbit", "source.txt", "pr", "The Tale of Peter Rabbit")
    flagged_lower = {w.lower() for w in report.pronunciation_flags}
    for name in ["flopsy", "mopsy"]:
        assert name in flagged_lower
    assert "cottontail" in flagged_lower or "cotton-tail" in flagged_lower
    assert validate(report) == []


def test_peter_rabbit_pharaoh_is_not_relevant_here_but_dictionary_covers_peter():
    report = _report_for("peter_rabbit", "source.txt", "pr", "The Tale of Peter Rabbit")
    peter = next(r for r in report.resolved if r.word == "Peter")
    assert peter.status == "dictionary"
