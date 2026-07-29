"""Pronunciation resolution: the three-step pipeline from data/lexicon/
pronunciation_lexicon.json -> "resolution_pipeline".

For every proper noun and every out-of-vocabulary word:
1. Dictionary lookup (pronouncing / CMU dict) -- no human involvement needed
   when it resolves. Covers most common personal names for free.
2. Custom lexicon lookup -- data/lexicon/pronunciation_lexicon.json, scoped
   either "global" (every manuscript) or "book:<title>" (one book only).
3. Flag for review -- never let a TTS engine silently guess. A word in
   neither source goes on pronunciation_flags for a human to resolve.

This is deterministic/heuristic, like core.ingest/chunker/verse/book_profile
-- no Claude call needed, the pipeline itself is the whole judgment call.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pronouncing

LEXICON_PATH = Path(__file__).parent.parent / "data" / "lexicon" / "pronunciation_lexicon.json"

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# CMU dict / the `pronouncing` package only recognizes the straight ASCII
# apostrophe -- these manuscripts (Word-authored) use curly quotes
# throughout, so every possessive/contraction ("Tom's", "don't") would
# otherwise spuriously fail dictionary lookup and get flagged even when a
# straight-apostrophe form is a real, resolvable dictionary entry.
_CURLY_APOSTROPHES = "‘’"


def _normalize_apostrophes(word: str) -> str:
    return word.translate(str.maketrans(_CURLY_APOSTROPHES, "''"))

# Sentence-initial capitalization is excluded at the point of scanning (it
# doesn't signal a proper noun); these recur capitalized mid-sentence too.
NAME_STOPWORDS = {
    "The", "A", "An", "He", "She", "They", "It", "His", "Her", "Their",
    "And", "But", "So", "Then", "Now", "I", "You", "We", "Mr", "Mrs",
}


@dataclass
class PronunciationResolution:
    word: str
    status: str  # "dictionary" | "lexicon"
    respelling: str | None = None
    ipa: str | None = None
    note: str | None = None


@dataclass
class PronunciationReport:
    resolved: list[PronunciationResolution] = field(default_factory=list)
    pronunciation_flags: list[str] = field(default_factory=list)


def load_lexicon(lexicon_path: Path = LEXICON_PATH) -> dict:
    with open(lexicon_path) as f:
        return json.load(f)


def _is_proper_noun_candidate(word: str) -> bool:
    if not word or not word[0].isupper():
        return False
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return False
    # All-caps emphasis words ("ROAR", "OK") aren't proper nouns.
    return not all(c.isupper() for c in letters)


def find_capitalized_words(narration_lines: list[str]) -> set[str]:
    found: set[str] = set()
    for line in narration_lines:
        for sentence in SENTENCE_SPLIT_RE.split(line.replace("\n", " ")):
            words = sentence.split()
            # Skip the sentence-initial word -- its capitalization doesn't
            # signal a proper noun the way a mid-sentence capital does.
            for word in words[1:]:
                cleaned = word.strip("\"“”.,!?;:()[]")
                if cleaned and cleaned not in NAME_STOPWORDS and _is_proper_noun_candidate(cleaned):
                    found.add(cleaned)
    return found


def collect_candidates(narration_lines: list[str], oov_words: list[str]) -> list[str]:
    by_lower: dict[str, str] = {}
    for word in oov_words:
        by_lower.setdefault(word.lower(), word)
    for word in find_capitalized_words(narration_lines):
        by_lower.setdefault(word.lower(), word)
    return list(by_lower.values())


def _find_lexicon_entry(word: str, book_title: str, lexicon: dict) -> dict | None:
    lowered = word.lower()
    matches = [e for e in lexicon["entries"] if e["word"].lower() == lowered]
    if not matches:
        return None
    book_scope = f"book:{book_title}"
    for entry in matches:
        if entry["scope"] == book_scope:
            return entry
    for entry in matches:
        if entry["scope"] == "global":
            return entry
    return None


def resolve_word(word: str, book_title: str, lexicon: dict) -> PronunciationResolution | None:
    cleaned = word.strip("\"“”.,!?;:()[]'’")
    if not cleaned:
        return None
    normalized = _normalize_apostrophes(cleaned)

    if pronouncing.phones_for_word(normalized.lower()):
        return PronunciationResolution(word=cleaned, status="dictionary")

    entry = _find_lexicon_entry(cleaned, book_title, lexicon)
    if entry:
        return PronunciationResolution(
            word=cleaned, status="lexicon",
            respelling=entry["respelling"], ipa=entry["ipa"], note=entry.get("note"),
        )

    # A possessive form ("Ava's") is often not its own dictionary entry even
    # though CMU dict resolves the base name just fine -- English forms
    # possessives regularly (name + an added -s sound), so inherit the base
    # word's resolution rather than flagging a name that's already known.
    if normalized.lower().endswith("'s") and len(normalized) > 2:
        base = normalized[:-2]
        if pronouncing.phones_for_word(base.lower()):
            return PronunciationResolution(word=cleaned, status="dictionary", note=f"possessive of '{base}'")
        base_entry = _find_lexicon_entry(base, book_title, lexicon)
        if base_entry:
            return PronunciationResolution(
                word=cleaned, status="lexicon",
                respelling=base_entry["respelling"] + "z", ipa=base_entry["ipa"] + "z",
                note=f"possessive of '{base}': {base_entry.get('note', '')}".strip(),
            )

    return None  # neither resolved -- caller adds it to pronunciation_flags


def resolve_manuscript_pronunciations(
    narration_lines: list[str],
    oov_words: list[str],
    book_title: str,
    lexicon: dict | None = None,
) -> PronunciationReport:
    lexicon = lexicon or load_lexicon()
    candidates = collect_candidates(narration_lines, oov_words)

    report = PronunciationReport()
    for word in sorted(candidates, key=str.lower):
        resolution = resolve_word(word, book_title, lexicon)
        if resolution is None:
            report.pronunciation_flags.append(word)
        else:
            report.resolved.append(resolution)
    return report


def validate(report: PronunciationReport) -> list[str]:
    flags = []
    flagged_lower = {w.lower() for w in report.pronunciation_flags}
    for r in report.resolved:
        if r.word.lower() in flagged_lower:
            flags.append(f"'{r.word}' appears in both resolved and pronunciation_flags")
        if r.status not in ("dictionary", "lexicon"):
            flags.append(f"'{r.word}' has invalid status '{r.status}'")
        if r.status == "lexicon" and not (r.respelling and r.ipa):
            flags.append(f"'{r.word}' is lexicon-resolved but missing respelling/ipa")
        if r.status == "dictionary" and (r.respelling or r.ipa):
            flags.append(f"'{r.word}' is dictionary-resolved but has a respelling/ipa set")
    return flags
