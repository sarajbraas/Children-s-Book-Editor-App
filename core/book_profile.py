"""Book-level profiling: genre, style era, narrator stance, emotional range.

Sets the emotional ceiling and narrator personality for a manuscript before
any chunk-level tagging happens (Phase 5). Calibration anchors come straight
from data/taxonomy/emotion_taxonomy.json's book_profile_schema.example_profiles
-- treat that file as canonical; this module reads it rather than duplicating
it, so an edit there stays the single source of truth.

Two axes are deliberately kept independent, per the taxonomy's own worked
example (Peter Rabbit): style_era/genre describe how the narrator DELIVERS
the book (composed vs. theatrical -- drives volume_ceiling), while
intensity_ceiling is driven by actual stakes/danger content in the text.
A restrained narrator voice does not imply low stakes.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "taxonomy" / "emotion_taxonomy.json"

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z']+")
DIALOGUE_RE = re.compile(r'["“][^"”]+["”]')
DIRECT_ADDRESS_RE = re.compile(r"\bI (?:think|believe|suppose|am sorry|dare say|fancy|must say)\b", re.IGNORECASE)
GENERIC_DEFINITION_RE = re.compile(r"^(?:A|An|The)\s+[a-z]+s?\s+(?:is|are)\b", re.IGNORECASE)
SPEECH_ATTRIBUTION_RE = re.compile(r"\b([A-Z][a-z]+)\s+(?:says|said|asks|asked|shouts|shouted|whispers?|proclaims?)\b")
ARCHETYPE_SUBJECT_RE = re.compile(r"\bthe\s+(fox|crow|hare|tortoise|lion|mouse|wolf|ant|grasshopper)\b", re.IGNORECASE)
MORAL_CLOSING_RE = re.compile(r"\b(?:the moral|and so|from that day)\b.{0,40}\blearn", re.IGNORECASE)

# Classic-literary sentences run noticeably longer than the short, punchy,
# exclamation-heavy lines contemporary picture books favor.
CLASSIC_AVG_SENTENCE_LENGTH = 16.0

# Danger/fear vocabulary density (hits per 1000 words) thresholds for
# intensity_ceiling -- independent of narrator register (see module note).
INTENSITY_HIGH_DENSITY = 4.0
INTENSITY_MEDIUM_DENSITY = 1.0

# High stakes aren't always fear-coded (a chaotic comedic action climax reads
# as high-intensity too, just via excitement rather than danger vocabulary)
# -- exclamation density is an independent signal that catches that case.
EXCLAMATION_HIGH_DENSITY = 15.0
EXCLAMATION_MEDIUM_DENSITY = 5.0

DANGER_WORDS = {
    "frightened", "fright", "frighten", "terror", "terrified", "terrifying",
    "danger", "dangerous", "chase", "chased", "chasing", "trembling", "tremble",
    "sob", "sobs", "sobbed", "scream", "screamed", "escape", "caught", "trap",
    "trapped", "afraid", "scared", "panic", "threat", "threatened", "dread",
    "dreadfully", "horror", "attack", "hurt", "kill", "killed", "death", "die",
    "died", "cry", "cried", "crying", "tears", "menacing", "sinister", "evil",
    "wicked", "snarled", "growled",
}


@dataclass
class EmotionalRange:
    intensity_ceiling: str
    volume_ceiling: str
    disabled_emotions: list[str] = field(default_factory=list)
    favored_humor_types: list[str] = field(default_factory=list)


@dataclass
class BookProfile:
    genre: str
    style_era: str
    narrator_stance: str
    emotional_range: EmotionalRange
    flags_for_review: list[str] = field(default_factory=list)


ANTAGONIST_REVIEW_NOTE = (
    "Antagonist type ambiguous: this manuscript has a recurring antagonist or "
    "disruptor character. Confirm whether Villainous/Menacing should be "
    "disabled for this book's antagonist (a comic/chaotic disruptor, not a "
    "real threat) or left enabled (a genuine, if age-appropriate, threat) "
    "before production -- the classifier deliberately leaves it enabled by "
    "default rather than guessing, since wrongly disabling it for a real "
    "threat undersells actual danger, while leaving it enabled for a comic "
    "antagonist only costs an occasional line read slightly too dark."
)

# Verbs/phrases that mark a character or entity as acting antagonistically
# toward the protagonist or their world. Detecting recurrence (2+ hits) is
# just a proxy for "a human should look at whether there's a villain here" --
# it deliberately does NOT try to judge comic-chaos vs. genuine-menace itself.
CONFLICT_VERB_RE = re.compile(
    r"\b(?:chas(?:e|ed|es|ing)|attack(?:s|ed|ing)?|disrupt(?:s|ed|ing)?|"
    r"trampl(?:e|ed|es|ing)|invad(?:e|ed|es|ing)|threaten(?:s|ed|ing)?|"
    r"menac\w*|terroriz\w*|pursu(?:e|ed|es|ing)|hunt(?:s|ed|ing)?|"
    r"scar(?:e|ed|es|ing)|frighten(?:s|ed|ing)?|wreck(?:s|ed|ing)?|"
    r"raid(?:s|ed|ing)?|ran after|run after|ran straight at|jumped up and ran after)\b",
    re.IGNORECASE,
)


def compute_book_profile(narration_lines: list[str], verse_form: bool, taxonomy_path: Path = TAXONOMY_PATH) -> BookProfile:
    text = "\n".join(narration_lines)
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]

    genre = _detect_genre(text, sentences)
    style_era = _detect_style_era(text, sentences, verse_form)

    anchors = _load_anchors(taxonomy_path)
    anchor = anchors.get((genre, style_era)) or anchors[("fiction", "contemporary")]

    intensity_ceiling = _detect_intensity_ceiling(text)

    flags = []
    if _has_recurring_antagonist(sentences):
        flags.append(ANTAGONIST_REVIEW_NOTE)

    return BookProfile(
        genre=genre,
        style_era=style_era,
        narrator_stance=anchor["narrator_stance"],
        flags_for_review=flags,
        emotional_range=EmotionalRange(
            intensity_ceiling=intensity_ceiling,
            volume_ceiling=anchor["emotional_range"]["volume_ceiling"],
            disabled_emotions=list(anchor["emotional_range"]["disabled_emotions"]),
            favored_humor_types=list(anchor["emotional_range"]["favored_humor_types"]),
        ),
    )


def _load_anchors(taxonomy_path: Path) -> dict[tuple[str, str], dict]:
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    profiles = taxonomy["book_profile_schema"]["example_profiles"]
    return {(p["genre"], p["style_era"]): p for p in profiles}


def _detect_genre(text: str, sentences: list[str]) -> str:
    if not sentences:
        return "fiction"

    dialogue_sentences = sum(1 for s in sentences if DIALOGUE_RE.search(s) or SPEECH_ATTRIBUTION_RE.search(s))
    generic_sentences = sum(1 for s in sentences if GENERIC_DEFINITION_RE.match(s.strip()))

    dialogue_ratio = dialogue_sentences / len(sentences)
    generic_ratio = generic_sentences / len(sentences)

    # No text-only signal reliably tells invented-character fiction apart
    # from real-people narrative nonfiction (both read like scenes with
    # dialogue) -- see module note. Only the strong expository signal
    # (generic class-level statements, little/no dialogue) is trustworthy
    # without external BISAC/Thema/ONIX metadata, which most manuscripts
    # won't carry anyway per the taxonomy's own detection_signals.
    if generic_ratio >= 0.2 and dialogue_ratio < 0.1:
        return "expository_nonfiction"
    return "fiction"


def _detect_style_era(text: str, sentences: list[str], verse_form: bool) -> str:
    if _is_oral_fable(text):
        return "oral_fable"

    if not sentences:
        return "contemporary"

    avg_len = sum(len(WORD_RE.findall(s)) for s in sentences) / len(sentences)
    direct_address_hits = len(DIRECT_ADDRESS_RE.findall(text))

    if direct_address_hits >= 1 or avg_len >= CLASSIC_AVG_SENTENCE_LENGTH:
        return "classic_literary"

    return "contemporary"


def _is_oral_fable(text: str) -> bool:
    has_archetype_subject = bool(ARCHETYPE_SUBJECT_RE.search(text))
    has_stated_moral = bool(MORAL_CLOSING_RE.search(text))
    return has_archetype_subject and has_stated_moral


def _has_recurring_antagonist(sentences: list[str]) -> bool:
    hits = sum(1 for s in sentences if CONFLICT_VERB_RE.search(s))
    return hits >= 2


def _detect_intensity_ceiling(text: str) -> str:
    words = WORD_RE.findall(text.lower())
    if not words:
        return "low"
    danger_hits = sum(1 for w in words if w in DANGER_WORDS)
    danger_density = danger_hits / len(words) * 1000
    exclamation_density = text.count("!") / len(words) * 1000

    if danger_density >= INTENSITY_HIGH_DENSITY or exclamation_density >= EXCLAMATION_HIGH_DENSITY:
        return "high"
    if danger_density >= INTENSITY_MEDIUM_DENSITY or exclamation_density >= EXCLAMATION_MEDIUM_DENSITY:
        return "medium"
    return "low"
