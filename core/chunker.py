"""Chunking: group ingested narration lines into sequential, narrator-sized chunks.

Takes the narration_lines produced by core.ingest and splits/merges them into
"chunks" -- roughly one spoken breath/beat each -- assigning each a sequential
mock ID. Does not assign emotion, intensity, scenes, or verse/meter data --
that happens in later phases.
"""

import re
from dataclasses import dataclass, field

import pronouncing

BRACKET_ASIDE_RE = re.compile(r"\[([^\]]*)\]")
QUOTED_SPAN_RE = re.compile(r'["“]([^"”]+)["”]')
HYPHEN_FRAGMENT_RE = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+){2,}\b")
WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# A tender/affectionate sentence immediately followed by a mundane, practical
# one is a common bathos/dry-irony pattern in children's picture books (the
# warm beat gets deflated by an ordinary follow-up action) -- e.g. "Mama
# gives Ava and Pharaoh Duck a quick cuddle too. Then they wash their hands
# very well." The two sentences read as one continuous moment on the page,
# but they're tonally opposed (warm vs. detached/wry) and need separate
# delivery -- split rather than merge.
WARMTH_WORDS = {
    "cuddle", "cuddles", "cuddled", "cuddling", "hug", "hugs", "hugged", "hugging",
    "kiss", "kisses", "kissed", "kissing", "snuggle", "snuggles", "snuggled",
    "tender", "gentle", "gently", "love", "loves", "loved", "loving",
    "sweet", "dear", "cherish", "cherishes", "cherished", "embrace", "embraces", "embraced",
}
MUNDANE_ACTION_WORDS = {
    "wash", "washes", "washed", "washing", "clean", "cleans", "cleaned", "cleaning",
    "wipe", "wipes", "wiped", "wiping", "tidy", "tidies", "tidied", "tidying",
    "scrub", "scrubs", "scrubbed", "scrubbing",
}

# Best-effort, expandable: common bodily/vocal sound words that mark a short
# line as a non-verbal sound_effect even when they aren't an onomatopoeia
# spelling (e.g. "A fart..." rather than "Ffffbbt!").
BODILY_SOUND_WORDS = {
    "fart", "farts", "burp", "burps", "belch", "belches", "cough", "coughs",
    "sneeze", "sneezes", "hiccup", "hiccups", "snore", "snores", "gulp", "gulps",
}

# Common words that get capitalized for reasons other than being a proper
# noun (sentence-initial position handles most of this already, but these
# recur mid-sentence too) -- excluded from candidate-speaker-name detection.
NAME_STOPWORDS = {
    "The", "A", "An", "He", "She", "They", "It", "His", "Her", "Their",
    "And", "But", "So", "Then", "Now", "I", "You", "We", "Mr", "Mrs",
}


@dataclass
class Chunk:
    id: str
    text: str
    type: str | None = None
    speaker: str | None = None
    inline_sound_effects: list[str] = field(default_factory=list)
    # Which narration_lines indices (core.ingest's ordering, pre-chunking)
    # contributed to this chunk. Usually one index; a line split by bracket
    # or tonal-shift splitting produces several chunks that each carry that
    # same index, and a merge (repeated interjections) produces one chunk
    # carrying multiple indices. This is how core.illustration maps an
    # author-provided illustration note (indexed by original line position)
    # forward to the right chunk after splitting/merging changes the count.
    source_lines: list[int] = field(default_factory=list)


def chunk_narration(narration_lines: list[str], bookcode: str) -> list[Chunk]:
    segments = []  # (text, source_line_index)
    for line_idx, line in enumerate(narration_lines):
        for part in _split_bracket_asides(line):
            for beat in _split_tonal_shift_beats(part):
                segments.append((beat, line_idx))

    merged = _merge_repeated_interjections(segments)
    candidate_names = _find_candidate_names([text for text, _ in merged])

    chunks = []
    last_seen_name = None
    for i, (text, source_lines) in enumerate(merged):
        chunk_type = None
        speaker = None
        is_chorus = _is_onomatopoeia_chorus(text)
        is_bodily = _is_bodily_sound_interjection(text)
        if is_chorus or is_bodily:
            chunk_type = "sound_effect"
            # A repeated onomatopoeia chorus ("Meep Meep Meep", "Gobble
            # gobble...") is usually an animal/collective sound, not the
            # most-recently-named human character -- too unreliable to
            # guess. A bodily-sound interjection ("A fart...") is almost
            # always the POV character, where "most recently named" holds.
            if is_bodily:
                speaker = last_seen_name

        # Sound-effect text itself is never a reliable "who's speaking next"
        # signal (e.g. a capitalized, repeated "Gobble" isn't a name).
        if chunk_type is None:
            names_here = [n for n in candidate_names if _contains_word(text, n)]
            if names_here:
                last_seen_name = names_here[-1]

        chunks.append(
            Chunk(
                id=f"{bookcode}-{i + 1:03d}",
                text=text,
                type=chunk_type,
                speaker=speaker,
                inline_sound_effects=_find_inline_onomatopoeia(text),
                source_lines=source_lines,
            )
        )
    return chunks


def _split_bracket_asides(line: str) -> list[str]:
    parts = []
    last_end = 0
    for m in BRACKET_ASIDE_RE.finditer(line):
        before = line[last_end:m.start()].strip()
        if before:
            parts.append(before)
        inside = m.group(1).strip()
        if inside:
            parts.append(inside)
        last_end = m.end()
    tail = line[last_end:].strip()
    if tail:
        parts.append(tail)
    if not parts and line.strip():
        parts.append(line.strip())
    return parts


def _has_word_from(text: str, words: set[str]) -> bool:
    tokens = {w.strip("\"“”.,!?;:'").lower() for w in text.split()}
    return bool(tokens & words)


def _split_tonal_shift_beats(text: str) -> list[str]:
    sentences = SENTENCE_SPLIT_RE.split(text)
    if len(sentences) < 2:
        return [text]

    beats = []
    current = sentences[0]
    for prev, cur in zip(sentences, sentences[1:]):
        is_warm_to_mundane = (
            _has_word_from(prev, WARMTH_WORDS)
            and _has_word_from(cur, MUNDANE_ACTION_WORDS)
            and not _has_word_from(cur, WARMTH_WORDS)
        )
        if is_warm_to_mundane:
            beats.append(current)
            current = cur
        else:
            current = f"{current} {cur}"
    beats.append(current)
    return beats


def _normalize_for_comparison(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _merge_repeated_interjections(segments: list[tuple[str, int]]) -> list[tuple[str, list[int]]]:
    merged = []
    i = 0
    while i < len(segments):
        current, current_idx = segments[i]
        if (
            i + 1 < len(segments)
            and _normalize_for_comparison(current) == _normalize_for_comparison(segments[i + 1][0])
            and len(current.split()) <= 6
        ):
            nxt, next_idx = segments[i + 1]
            merged_text = f"{current.rstrip('.!? ')}. {nxt.rstrip('.!? ')}."
            merged.append((merged_text, [current_idx, next_idx]))
            i += 2
        else:
            merged.append((current, [current_idx]))
            i += 1
    return merged


def _is_onomatopoeia_chorus(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    return len(words) >= 2 and len({w.lower() for w in words}) == 1


def _is_bodily_sound_interjection(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    return bool(words) and len(words) <= 4 and any(w.lower() in BODILY_SOUND_WORDS for w in words)


def _is_oov(word: str) -> bool:
    return not pronouncing.phones_for_word(word.lower())


INFORMAL_SUFFIXES = ("y", "ie")


def _is_informal_derivation(word: str) -> bool:
    """True for OOV words that are just a real word plus a diminutive/informal
    suffix (e.g. "germy" -> "germ", "Duckie" -> "duck") -- these are informal
    vocabulary, not onomatopoeia, even though the dictionary lacks them."""
    lowered = word.lower()
    for suffix in INFORMAL_SUFFIXES:
        if lowered.endswith(suffix) and not _is_oov(lowered[: -len(suffix)]):
            return True
    return False


def _find_inline_onomatopoeia(text: str) -> list[str]:
    hits = []
    for quoted in QUOTED_SPAN_RE.findall(text):
        for word in WORD_RE.findall(quoted):
            if _is_oov(word) and not _is_informal_derivation(word):
                hits.append(word)
    for word in HYPHEN_FRAGMENT_RE.findall(text):
        hits.append(word)
    return hits


def _find_candidate_names(chunks: list[str]) -> set[str]:
    counts: dict[str, int] = {}
    for text in chunks:
        if _is_onomatopoeia_chorus(text):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            words = sentence.split()
            # Skip the sentence-initial word -- its capitalization doesn't
            # signal a proper noun the way a mid-sentence capital does.
            for word in words[1:]:
                cleaned = word.strip("\"“”.,!?;:")
                if cleaned and cleaned[0].isupper() and cleaned not in NAME_STOPWORDS:
                    counts[cleaned] = counts.get(cleaned, 0) + 1
    return {name for name, count in counts.items() if count >= 2}


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None
