"""Illustration inference -- what does the (unseen) companion illustration likely show?

Modern picture books are "show don't tell": a bare sentence like "The court
entertains their Duck King." often corresponds to a full illustrated spread
doing most of the emotional/narrative work. A text-only pipeline systematically
under-reads energy on these lines unless it explicitly accounts for the
missing picture. Ground truth beats guessing every time:

- If Phase 1 captured an author-provided illustration note tied to this
  chunk's position, that's the real, actual planned art -- use it verbatim,
  never regenerated or second-guessed by a heuristic.
- Otherwise, imagining what the illustration probably shows is a genuinely
  creative/visual reasoning task (the same kind of judgment core.tagger and
  core.humor need Claude for, not a template). The taxonomy's
  detection_signals (sparse/parallel anaphoric structure, concrete noun/verb
  density) are used here as heuristic GATING and a hint passed into the
  prompt -- deciding whether a chunk is even a candidate, and how to frame
  the request -- not as the thing that produces the description itself.
- If neither a note nor a strong heuristic signal exists (e.g. an old
  public-domain reprint with only bare "[Illustration]" placeholders and no
  descriptive text), the field is omitted entirely. No basis, no guess.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import anthropic

from core.book_profile import BookProfile
from core.ingest import IllustrationNote

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "taxonomy" / "emotion_taxonomy.json"

MODEL = "claude-opus-4-8"
CONTEXT_WINDOW = 2
VISUAL_ENERGY_LEVELS = ["low", "medium", "high"]

# A chunk needs to actually BE sparse for "the picture is doing the work" to
# apply -- a long, already-descriptive sentence isn't leaving anything
# meaningful for an illustration to add. This gates both heuristic signals
# below; without it, "concrete noun density" alone would fire on any richly
# detailed (but already fully-described) sentence.
SPARSE_WORD_LIMIT = 10

CONCRETE_SETTING_WORDS = {
    "lawn", "basket", "blanket", "palace", "castle", "garden", "court", "picnic",
    "table", "hose", "bath", "kingdom", "throne", "crown", "parade", "yard",
    "porch", "kitchen", "nest", "nook", "window", "gate",
}
MOTION_VERBS = {
    "walks", "walked", "runs", "ran", "jumps", "jumped", "marches", "marched",
    "promenades", "promenaded", "waves", "waved", "dances", "danced", "places",
    "placed", "follows", "followed", "entertains", "entertained", "keeps", "kept",
    "dresses", "dressed", "carries", "carried", "gathers", "gathered",
    "proclaims", "pronounces", "proclaimed", "pronounced",
}
CONCRETE_DENSITY_THRESHOLD = 0.18


@dataclass
class ChunkIllustration:
    chunk_id: str
    implied_scene: str
    source: str  # "author" | "inferred"
    visual_energy: str | None = None  # only ever set for "inferred"


def load_taxonomy(taxonomy_path: Path = TAXONOMY_PATH) -> dict:
    with open(taxonomy_path) as f:
        return json.load(f)


def map_author_notes_to_chunks(chunks: list, author_notes: list[IllustrationNote]) -> dict[str, str]:
    """Map each Phase 1 illustration note (indexed by original narration-line
    position) to the chunk it belongs to: the LAST chunk derived from that
    line, since a note sits at the tail end of the original line's raw text."""
    last_chunk_for_line: dict[int, str] = {}
    for chunk in chunks:
        for line_idx in chunk.source_lines:
            last_chunk_for_line[line_idx] = chunk.id

    mapping = {}
    for note in author_notes:
        chunk_id = last_chunk_for_line.get(note.position_after_line)
        if chunk_id:
            mapping[chunk_id] = note.note_text
    return mapping


def _leading_stub(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text)
    return " ".join(w.lower() for w in words[:2])


def _find_anaphoric_run_chunk_ids(chunks: list) -> set[str]:
    run_ids: set[str] = set()
    i = 0
    while i < len(chunks):
        j = i
        stub = _leading_stub(chunks[i].text)
        while stub and j + 1 < len(chunks) and _leading_stub(chunks[j + 1].text) == stub:
            j += 1
        if j > i:
            run_ids.update(c.id for c in chunks[i:j + 1])
        i = j + 1
    return run_ids


def _concrete_density(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.lower() in CONCRETE_SETTING_WORDS or w.lower() in MOTION_VERBS)
    hits += sum(1 for w in words[1:] if w[:1].isupper())
    return hits / len(words)


def _heuristic_hint(chunk, anaphoric_run_ids: set[str]) -> str | None:
    if chunk.type == "sound_effect":
        return None  # a non-verbal sound isn't a scene an illustration depicts

    word_count = len(re.findall(r"[A-Za-z']+", chunk.text))
    if word_count > SPARSE_WORD_LIMIT:
        return None  # already descriptive enough -- nothing meaningful left implied

    if chunk.id in anaphoric_run_ids:
        return (
            "This line is part of a run of short sentences sharing the same repeated "
            "subject/sentence template -- a strong signal each one corresponds to its own "
            "full illustrated spread doing most of the visual work."
        )
    if _concrete_density(chunk.text) >= CONCRETE_DENSITY_THRESHOLD:
        return (
            "This short line is dense with concrete, visualizable nouns and verbs (named "
            "settings, props, or motion) -- likely anchoring a specific, busy illustration."
        )
    return None


def build_system_prompt(taxonomy: dict) -> str:
    signals = taxonomy["illustration_inference"]["detection_signals"]
    signals_block = "\n".join(f"- {s}" for s in signals)
    return f"""Modern picture books are "show don't tell" -- a bare, plain sentence often \
corresponds to a full illustrated spread doing most of the emotional/narrative work. Your job \
is to imagine, for ONE specific chunk, what the (unseen) companion illustration most likely \
shows -- not what the sentence literally states, but the fuller visual scene implied by it \
(character expressions, spatial arrangement, motion, background action).

Background on why a line might be a candidate for this at all:
{signals_block}

You've been told WHY this particular chunk is a candidate (a heuristic hint, below) -- treat \
that as a reason to look harder for what's implied, not as the answer itself. Use the \
surrounding context and the book's register to actually imagine the scene.

Produce:
- implied_scene: one sentence, concrete and specific (who/what is doing what, arranged how) -- \
not a restatement of the chunk's own words.
- visual_energy: low | medium | high -- how busy/dynamic the imagined image likely is. This is \
independent of the text's own stated emotion; a plain sentence can still imply a high-energy \
image (per the show-don't-tell principle above).

This is an approximation for a human/illustrator review pass, not a claim of fact -- an actual \
illustrator can and does make different choices than the text implies. Do the best specific, \
concrete guess you can; do not hedge or produce something generic enough to apply to any scene."""


def build_user_message(
    chunk_id: str,
    chunk_text: str,
    before: list[tuple[str, str]],
    after: list[tuple[str, str]],
    book_profile: BookProfile,
    hint: str,
) -> str:
    before_block = "\n".join(f"[{cid}] {text}" for cid, text in before) or "(start of book)"
    after_block = "\n".join(f"[{cid}] {text}" for cid, text in after) or "(end of book)"
    return f"""BOOK PROFILE:
genre: {book_profile.genre}
style_era: {book_profile.style_era}
narrator_stance: {book_profile.narrator_stance}

HEURISTIC HINT for this chunk: {hint}

CONTEXT -- chunks immediately before:
{before_block}

CONTEXT -- chunks immediately after:
{after_block}

CHUNK TO IMAGINE THE ILLUSTRATION FOR [{chunk_id}]:
{chunk_text}"""


def build_inference_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "implied_scene": {"type": "string"},
            "visual_energy": {"type": "string", "enum": VISUAL_ENERGY_LEVELS},
        },
        "required": ["implied_scene", "visual_energy"],
        "additionalProperties": False,
    }


def infer_scene(
    client: anthropic.Anthropic,
    chunk_id: str,
    chunk_text: str,
    before: list[tuple[str, str]],
    after: list[tuple[str, str]],
    book_profile: BookProfile,
    hint: str,
    taxonomy: dict | None = None,
    model: str = MODEL,
) -> tuple[str, str]:
    taxonomy = taxonomy or load_taxonomy()
    schema = build_inference_schema()
    system_prompt = build_system_prompt(taxonomy)
    user_message = build_user_message(chunk_id, chunk_text, before, after, book_profile, hint)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )

    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return data["implied_scene"], data["visual_energy"]


def infer_illustrations(
    client: anthropic.Anthropic,
    chunks: list,
    author_notes: list[IllustrationNote],
    book_profile: BookProfile,
    taxonomy: dict | None = None,
    model: str = MODEL,
) -> list[ChunkIllustration]:
    taxonomy = taxonomy or load_taxonomy()
    author_by_chunk = map_author_notes_to_chunks(chunks, author_notes)
    anaphoric_run_ids = _find_anaphoric_run_chunk_ids(chunks)

    results = []
    for i, chunk in enumerate(chunks):
        if chunk.id in author_by_chunk:
            results.append(
                ChunkIllustration(chunk_id=chunk.id, implied_scene=author_by_chunk[chunk.id], source="author")
            )
            continue

        hint = _heuristic_hint(chunk, anaphoric_run_ids)
        if hint is None:
            continue  # genuinely nothing to go on -- omit rather than guess

        before = [(c.id, c.text) for c in chunks[max(0, i - CONTEXT_WINDOW):i]]
        after = [(c.id, c.text) for c in chunks[i + 1:i + 1 + CONTEXT_WINDOW]]
        implied_scene, visual_energy = infer_scene(
            client, chunk.id, chunk.text, before, after, book_profile, hint, taxonomy, model
        )
        results.append(
            ChunkIllustration(
                chunk_id=chunk.id, implied_scene=implied_scene, source="inferred", visual_energy=visual_energy
            )
        )

    return results


def validate(results: list[ChunkIllustration], author_notes_count: int) -> list[str]:
    flags = []
    author_results = [r for r in results if r.source == "author"]
    if len(author_results) != author_notes_count:
        flags.append(
            f"expected {author_notes_count} author-sourced illustration notes to map to "
            f"chunks, but only {len(author_results)} did -- some notes may not have been "
            "matched to a chunk"
        )
    for r in results:
        if r.source not in ("author", "inferred"):
            flags.append(f"chunk '{r.chunk_id}' has invalid source '{r.source}'")
        if r.source == "author" and r.visual_energy is not None:
            flags.append(f"chunk '{r.chunk_id}' is author-sourced but has visual_energy set")
        if r.source == "inferred" and r.visual_energy not in VISUAL_ENERGY_LEVELS:
            flags.append(f"chunk '{r.chunk_id}' is inferred but has invalid visual_energy '{r.visual_energy}'")
    return flags
