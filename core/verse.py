"""Verse/meter scansion: syllable counts, rhyme detection, and meter breaks.

Operates on "lines" grouped by chunk (a chunk may hold one original poem
line, or several merged together -- see core.chunker). Chunk-level rhyme
pairing may reference a DIFFERENT chunk's id via rhymes_with_chunk when a
couplet's two rhyming lines ended up in separate chunks.
"""

import re
from dataclasses import dataclass, field
from statistics import median

import pronouncing

WORD_RE = re.compile(r"[A-Za-z']+")

# A line undershooting the book's dominant syllable count by this much is a
# strong secondary signal for an intentional pacing pause, even when rhyme
# resolution (the primary signal) can't be computed (e.g. OOV gaps).
UNDERSHOOT_RATIO = 0.6

# Below this fraction of lines participating in a detected rhyme pairing,
# treat the book as prose rather than verse.
VERSE_FORM_THRESHOLD = 0.3


@dataclass
class VerseInfo:
    syllables_per_line: list[int]
    # meter_break and rhyme_role are computed unconditionally, for every
    # chunk, because BookVerseProfile.verse_form is itself derived from the
    # aggregate pattern of these two fields across the manuscript -- prose
    # can't be told from verse without computing them first. That means
    # they're raw signal, not meaningful in isolation: for a chunk in a
    # prose book they're just coincidental noise (a sentence that happened
    # to not rhyme with whatever came before it). Any downstream consumer
    # (tagging, reports/output) MUST check book_profile.verse_form is True
    # before treating either field as meaningful.
    meter_break: bool
    rhyme_role: str
    rhymes_with_chunk: str | None = None


@dataclass
class BookVerseProfile:
    verse_form: bool
    rhyme_scheme: str | None
    dominant_meter: str | None
    meter_break_count: int


@dataclass
class VerseResult:
    verse_by_chunk: dict[str, VerseInfo] = field(default_factory=dict)
    oov_words: list[str] = field(default_factory=list)
    book_profile: BookVerseProfile | None = None


def compute_verse(chunk_lines: list[tuple[str, list[str]]]) -> VerseResult:
    """chunk_lines: ordered list of (chunk_id, [original_line_text, ...])."""
    oov_words: set[str] = set()

    flat_lines = []  # (chunk_id, line_idx_in_chunk, text, syllables, last_word)
    for chunk_id, lines in chunk_lines:
        for line_idx, text in enumerate(lines):
            syllables = _syllables_for_line(text, oov_words)
            flat_lines.append((chunk_id, line_idx, text, syllables, _last_word(text)))

    if not flat_lines:
        return VerseResult(book_profile=BookVerseProfile(False, None, None, 0))

    dominant = median(s for *_, s, _ in flat_lines) or 1

    roles, meter_breaks, rhymes_with = _resolve_rhyme_roles(flat_lines, dominant, oov_words)

    verse_by_chunk: dict[str, VerseInfo] = {}
    per_chunk_lines: dict[str, list[tuple]] = {}
    for entry in flat_lines:
        per_chunk_lines.setdefault(entry[0], []).append(entry)

    for chunk_id, lines in chunk_lines:
        entries = per_chunk_lines.get(chunk_id, [])
        syllables_per_line = [e[3] for e in entries]
        line_roles = [roles[(e[0], e[1])] for e in entries]
        line_breaks = [meter_breaks[(e[0], e[1])] for e in entries]
        line_rhymes_with = [rhymes_with.get((e[0], e[1])) for e in entries]

        chunk_role = _combine_roles(line_roles)
        # A chunk whose two lines close one couplet and open another can't
        # point rhymes_with_chunk at a single partner -- leave it unset.
        if chunk_role == "closes_couplet_then_opens_new":
            chunk_rhymes_with = None
        else:
            chunk_rhymes_with = next((r for r in line_rhymes_with if r), None)
        chunk_meter_break = any(line_breaks)

        verse_by_chunk[chunk_id] = VerseInfo(
            syllables_per_line=syllables_per_line,
            meter_break=chunk_meter_break,
            rhyme_role=chunk_role,
            rhymes_with_chunk=chunk_rhymes_with,
        )

    resolved_count = sum(
        1 for r in roles.values() if r in ("self_contained_couplet", "opens_couplet", "closes_couplet")
    )
    verse_form = (resolved_count / len(flat_lines)) >= VERSE_FORM_THRESHOLD

    book_profile = BookVerseProfile(
        verse_form=verse_form,
        rhyme_scheme="AABB couplets" if verse_form else None,
        dominant_meter=f"~{int(dominant)} syllables per line (median)" if verse_form else None,
        meter_break_count=sum(meter_breaks.values()),
    )

    return VerseResult(verse_by_chunk=verse_by_chunk, oov_words=sorted(oov_words), book_profile=book_profile)


def _last_word(text: str) -> str:
    words = WORD_RE.findall(text)
    return words[-1] if words else ""


def _syllables_for_line(text: str, oov_words: set[str]) -> int:
    return sum(_syllables_for_word(w, oov_words) for w in WORD_RE.findall(text))


def _syllables_for_word(word: str, oov_words: set[str]) -> int:
    cleaned = word.strip("'")
    if not cleaned:
        return 0
    phones = pronouncing.phones_for_word(cleaned.lower())
    if phones:
        return pronouncing.syllable_count(phones[0])
    oov_words.add(cleaned)
    return _fallback_syllable_count(cleaned)


def _fallback_syllable_count(word: str) -> int:
    lowered = re.sub(r"[^a-z]", "", word.lower())
    if not lowered:
        return 0
    groups = re.findall(r"[aeiouy]+", lowered)
    count = len(groups)
    if lowered.endswith("e") and count > 1:
        count -= 1
    if lowered.endswith("le") and len(lowered) > 2 and lowered[-3] not in "aeiouy":
        count += 1
    return max(count, 1)


def _is_onomatopoeia_chorus(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    return len(words) >= 2 and len({w.lower() for w in words}) == 1


def _rhymes(word_a: str, word_b: str, oov_words: set[str]) -> bool:
    variants_a = pronouncing.phones_for_word(word_a.lower())
    variants_b = pronouncing.phones_for_word(word_b.lower())
    if not variants_a:
        oov_words.add(word_a)
    if not variants_b:
        oov_words.add(word_b)
    if not variants_a or not variants_b:
        return (
            len(word_a) >= 3
            and len(word_b) >= 3
            and word_a.lower()[-3:] == word_b.lower()[-3:]
        )

    for pa in variants_a:
        for pb in variants_b:
            rp_a, rp_b = pronouncing.rhyming_part(pa), pronouncing.rhyming_part(pb)
            if rp_a and rp_b:
                if rp_a == rp_b or _strip_stress(rp_a) == _strip_stress(rp_b):
                    return True
            if _strip_stress(_last_phones(pa, 2)) == _strip_stress(_last_phones(pb, 2)):
                return True
            # Tolerate a trailing plural/possessive S or Z (e.g. "chests" /
            # "test") -- a very common slant-rhyme pattern in children's verse.
            if _strip_stress(_last_phones(_drop_trailing_sz(pa), 2)) == _strip_stress(
                _last_phones(_drop_trailing_sz(pb), 2)
            ):
                return True
    return False


def _last_phones(phones: str, n: int) -> str:
    tokens = phones.split()
    return " ".join(tokens[-n:])


def _drop_trailing_sz(phones: str) -> str:
    tokens = phones.split()
    if tokens and tokens[-1] in ("S", "Z"):
        return " ".join(tokens[:-1])
    return phones


def _strip_stress(phones: str) -> str:
    return re.sub(r"\d", "", phones)


def _resolve_rhyme_roles(flat_lines, dominant, oov_words):
    """Greedily pairs lines into AABB-style couplets in document order.

    A line that sharply undershoots the dominant syllable count is treated
    as ineligible to *seed* a new couplet (it's almost always a deliberate
    pacing interruption, not the start of a real rhyme pair) -- but once a
    couplet IS pending, any non-rhyming line in between (short or not, e.g.
    a normal-length line that simply needs another beat to reach its rhyme
    partner) is an interruption too, and does not disturb the pending line.
    meter_break is true for every line that ends up in an interruption role,
    since that's what actually breaks the book's regular AABB rhythm --
    syllable count alone is only a heuristic for *why* a line ended up there.
    """
    roles: dict[tuple, str] = {}
    meter_breaks: dict[tuple, bool] = {}
    rhymes_with: dict[tuple, str] = {}
    by_key = {(c, i): (text, syllables, last_word) for c, i, text, syllables, last_word in flat_lines}

    pending_key = None

    for chunk_id, line_idx, text, syllables, last_word in flat_lines:
        key = (chunk_id, line_idx)

        if _is_onomatopoeia_chorus(text) or not last_word:
            roles[key] = "interruption_between_rhyme" if pending_key else "standalone_interruption"
            continue

        if pending_key is None:
            undershoots = syllables > 0 and syllables <= dominant * UNDERSHOOT_RATIO
            if undershoots:
                roles[key] = "standalone_interruption"
            else:
                pending_key = key
                roles[key] = "opens_couplet"  # provisional; may be revised below
            continue

        pending_text, _, pending_last_word = by_key[pending_key]
        pending_chunk_id = pending_key[0]
        if _rhymes(last_word, pending_last_word, oov_words):
            if chunk_id == pending_chunk_id:
                roles[pending_key] = "self_contained_couplet"
                roles[key] = "self_contained_couplet"
            else:
                roles[pending_key] = "opens_couplet"
                roles[key] = "closes_couplet"
                rhymes_with[pending_key] = chunk_id
                rhymes_with[key] = pending_chunk_id
            pending_key = None
        else:
            roles[key] = "interruption_between_rhyme"

    if pending_key is not None and roles.get(pending_key) == "opens_couplet":
        # Never found a partner -- don't claim a couplet role that isn't real.
        roles[pending_key] = "standalone_interruption"

    for key, role in roles.items():
        meter_breaks[key] = role in ("interruption_between_rhyme", "standalone_interruption")

    return roles, meter_breaks, rhymes_with


def _combine_roles(line_roles: list[str]) -> str:
    if len(line_roles) == 1:
        return line_roles[0]
    unique = list(dict.fromkeys(line_roles))
    if len(unique) == 1:
        return unique[0]
    if unique == ["closes_couplet", "opens_couplet"]:
        return "closes_couplet_then_opens_new"
    return "_then_".join(unique)
