"""Humor mechanism classification -- WHY a line is funny, not HOW it sounds.

emotion="Humorous/Funny" (core.tagger) says a line should land as amused/comedic
delivery. It says nothing about WHICH comedic mechanism is doing the work, and
two "Humorous/Funny" lines can need completely different deliveries: a dry,
understated callback reads nothing like a big slapstick pratfall. humor_type
(one of the 9 entries in the taxonomy's humor_types) is that separate axis.

Per the taxonomy's own scene model, a joke's humor_type is usually a property
of the SCENE (core.scenes' comedic_motif already carries one) rather than any
single chunk -- a setup chunk and its payoff chunk share one mechanism across
the whole arc. So this module does two different things depending on whether
a chunk's joke is scene-spanning or self-contained:

1. If a chunk is a member of a comedic_motif scene, its humor_type is simply
   inherited from that scene -- no new judgment call needed, it was already
   made in core.scenes.
2. If a chunk's own emotion/secondary_emotion is "Humorous/Funny" but it
   ISN'T part of any comedic_motif scene, the joke is fully self-contained in
   that one line, so a fresh classification call decides its humor_type.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import anthropic

from core.book_profile import BookProfile
from core.tagger import ChunkTag

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "taxonomy" / "emotion_taxonomy.json"

MODEL = "claude-opus-4-8"
CONTEXT_WINDOW = 2


@dataclass
class ChunkHumor:
    chunk_id: str
    humor_type: str
    source: str  # "scene" (inherited from a comedic_motif scene) | "chunk" (classified standalone)


def load_taxonomy(taxonomy_path: Path = TAXONOMY_PATH) -> dict:
    with open(taxonomy_path) as f:
        return json.load(f)


def _humor_type_names(taxonomy: dict) -> list[str]:
    return [h["name"] for h in taxonomy["humor_types"]["types"]]


def build_classification_schema(humor_type_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"humor_type": {"type": "string", "enum": humor_type_names}},
        "required": ["humor_type"],
        "additionalProperties": False,
    }


def build_system_prompt(taxonomy: dict) -> str:
    humor_block = "\n".join(
        f"- {h['name']}: {h['description']} (deliver: {h['delivery']})"
        for h in taxonomy["humor_types"]["types"]
    )
    return f"""A chunk of a children's audiobook manuscript has already been tagged \
emotion="Humorous/Funny" (or has it as a secondary emotion) -- that tag only says the line \
should sound amused/comedic. Your job is different: classify WHICH comedic mechanism makes \
this SPECIFIC line funny, from this fixed list:

{humor_block}

Two lines can both be "Humorous/Funny" in tone and still need opposite deliveries -- a dry, \
understated callback (dry_ironic) should be underplayed, while a pratfall (slapstick_physical) \
should be big and physical. Picking the right mechanism is what tells a later production step \
how to actually perform the line, not just that it's funny.

incongruity_absurd and dry_ironic are the two easiest to conflate -- both can sound "wry." The \
test that tells them apart: does the humor come from a MISMATCHED REGISTER (fancy, elevated, or \
formal language/imagery applied to something silly or mundane, or the reverse)? That's \
incongruity_absurd. Or does it come from UNDERSTATING a reaction that the situation would \
naturally call for, in otherwise plain, ordinary vocabulary -- no register mismatch, just \
restraint? That's dry_ironic. A plainly-worded sentence that undersells its own consequences \
("he got in trouble" stated flatly) is dry_ironic even with zero fancy vocabulary; incongruity \
requires an actual clash in the words themselves, not just a flat tone.

This chunk is self-contained -- if it needed a setup from many chunks earlier to land, it \
would already be part of a scene-level running gag or callback and wouldn't be reaching you \
for standalone classification. Judge it on what's funny about THIS line on its own (with the \
surrounding context only for tone, not as a required setup).

Respond with exactly one humor_type from the list above."""


def build_user_message(
    chunk_id: str,
    chunk_text: str,
    before: list[tuple[str, str]],
    after: list[tuple[str, str]],
    book_profile: BookProfile,
) -> str:
    before_block = "\n".join(f"[{cid}] {text}" for cid, text in before) or "(start of book)"
    after_block = "\n".join(f"[{cid}] {text}" for cid, text in after) or "(end of book)"
    return f"""BOOK PROFILE:
genre: {book_profile.genre}
style_era: {book_profile.style_era}
narrator_stance: {book_profile.narrator_stance}
favored_humor_types: {book_profile.emotional_range.favored_humor_types or "none specified"}

CONTEXT -- chunks immediately before:
{before_block}

CONTEXT -- chunks immediately after:
{after_block}

CHUNK TO CLASSIFY [{chunk_id}] (already tagged Humorous/Funny):
{chunk_text}"""


def classify_standalone_humor(
    client: anthropic.Anthropic,
    chunk_id: str,
    chunk_text: str,
    before: list[tuple[str, str]],
    after: list[tuple[str, str]],
    book_profile: BookProfile,
    taxonomy: dict | None = None,
    model: str = MODEL,
) -> str:
    taxonomy = taxonomy or load_taxonomy()
    schema = build_classification_schema(_humor_type_names(taxonomy))
    system_prompt = build_system_prompt(taxonomy)
    user_message = build_user_message(chunk_id, chunk_text, before, after, book_profile)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["humor_type"]


def assign_humor_types(
    client: anthropic.Anthropic,
    chunks: list,
    tag_by_id: dict[str, ChunkTag],
    scenes: list,
    book_profile: BookProfile,
    taxonomy: dict | None = None,
    model: str = MODEL,
) -> list[ChunkHumor]:
    taxonomy = taxonomy or load_taxonomy()

    scene_humor_by_chunk: dict[str, str] = {}
    for scene in scenes:
        if scene.type != "comedic_motif":
            continue
        for member in scene.members:
            scene_humor_by_chunk[member.chunk_id] = scene.humor_type

    results = []
    for i, chunk in enumerate(chunks):
        if chunk.id in scene_humor_by_chunk:
            results.append(ChunkHumor(chunk.id, scene_humor_by_chunk[chunk.id], "scene"))
            continue

        tag = tag_by_id.get(chunk.id)
        is_funny = bool(tag) and (tag.emotion == "Humorous/Funny" or tag.secondary_emotion == "Humorous/Funny")
        if not is_funny:
            continue

        before = [(c.id, c.text) for c in chunks[max(0, i - CONTEXT_WINDOW):i]]
        after = [(c.id, c.text) for c in chunks[i + 1:i + 1 + CONTEXT_WINDOW]]
        humor_type = classify_standalone_humor(
            client, chunk.id, chunk.text, before, after, book_profile, taxonomy, model
        )
        results.append(ChunkHumor(chunk.id, humor_type, "chunk"))

    return results


def validate_coverage(
    results: list[ChunkHumor], chunks: list, tag_by_id: dict[str, ChunkTag], scenes: list, taxonomy: dict | None = None
) -> list[str]:
    """Loud safety net: confirm every chunk that SHOULD have a humor_type got
    one, and that every assigned value is actually a real humor_type."""
    taxonomy = taxonomy or load_taxonomy()
    allowed = set(_humor_type_names(taxonomy))
    covered = {r.chunk_id for r in results}

    comedic_members = {
        m.chunk_id for s in scenes if s.type == "comedic_motif" for m in s.members
    }

    flags = []
    for chunk in chunks:
        tag = tag_by_id.get(chunk.id)
        should_have = chunk.id in comedic_members or (
            bool(tag) and (tag.emotion == "Humorous/Funny" or tag.secondary_emotion == "Humorous/Funny")
        )
        if should_have and chunk.id not in covered:
            flags.append(f"chunk '{chunk.id}' should have a humor_type but got none")

    for r in results:
        if r.humor_type not in allowed:
            flags.append(f"chunk '{r.chunk_id}' has invalid humor_type '{r.humor_type}'")

    return flags
