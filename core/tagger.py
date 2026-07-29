"""Emotion/intensity/volume tagging via the Claude API.

This is the interpretive phase: is this line funny, is this whisper still
high-intensity, does this sound effect belong to a character rather than the
narrator? These are real language-understanding judgment calls, not things a
regex classifier can make -- so unlike core.ingest/chunker/verse/book_profile,
this module calls Claude (via the `anthropic` SDK) per chunk, with 1-2 chunks
of surrounding context, the book_profile ceilings, and the full emotion
taxonomy, and asks for structured JSON back.

Ceilings and disabled emotions are stated in the prompt, but never trusted
blindly -- _validate() re-checks every tag against book_profile after the
fact and surfaces violations as loud, explicit flags rather than silently
clamping or silently passing them.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from core.book_profile import BookProfile

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "taxonomy" / "emotion_taxonomy.json"

MODEL = "claude-opus-4-8"

INTENSITY_LEVELS = ["low", "medium", "high"]
VOLUME_LEVELS = ["whisper", "soft", "normal", "loud", "shout"]

# Known pairs that pull in genuinely different energetic directions, per the
# taxonomy's own secondary_emotion compatibility_guidance ("pairs that pull
# in opposite energetic directions... that's a signal the line is actually a
# transition and should be split into two chunks"). This is deliberately a
# small, curated safety net, not a formal compatibility matrix over all
# 17x16 pairs -- it exists to catch the model pairing two feelings that
# don't actually blend even when it doesn't self-flag the conflict (e.g. it
# can read generic "Humorous/Funny" as warm per its own taxonomy
# description, missing that THIS instance is the detached, dry-ironic
# register, which doesn't blend with Tender/Loving). Extend as new cases
# turn up in review, the same way BODILY_SOUND_WORDS/DANGER_WORDS grow.
INCOMPATIBLE_SECONDARY_EMOTION_PAIRS = {
    frozenset({"Calm/Soothing", "Excited/Energetic"}),
    frozenset({"Tender/Loving", "Humorous/Funny"}),
}

CONTEXT_WINDOW = 2  # chunks of context on each side


@dataclass
class ChunkTag:
    emotion: str
    secondary_emotion: str | None
    intensity: str
    volume: str
    emphasis: list[str]
    type: str | None
    speaker: str | None
    note: str | None
    flag_for_review: bool
    flag_reason: str | None


@dataclass
class TagResult:
    chunk_id: str
    tag: ChunkTag
    validation_flags: list[str] = field(default_factory=list)


def load_taxonomy(taxonomy_path: Path = TAXONOMY_PATH) -> dict:
    with open(taxonomy_path) as f:
        return json.load(f)


def _emotion_names(taxonomy: dict) -> list[str]:
    return [e["name"] for e in taxonomy["emotions"]]


def build_tag_schema(emotion_names: list[str]) -> dict:
    emotion_enum = {"type": "string", "enum": emotion_names}
    return {
        "type": "object",
        "properties": {
            "emotion": emotion_enum,
            "secondary_emotion": {"anyOf": [emotion_enum, {"type": "null"}]},
            "intensity": {"type": "string", "enum": INTENSITY_LEVELS},
            "volume": {"type": "string", "enum": VOLUME_LEVELS},
            "emphasis": {"type": "array", "items": {"type": "string"}},
            "type": {"anyOf": [{"type": "string", "enum": ["sound_effect"]}, {"type": "null"}]},
            "speaker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "flag_for_review": {"type": "boolean"},
            "flag_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": [
            "emotion", "secondary_emotion", "intensity", "volume", "emphasis",
            "type", "speaker", "note", "flag_for_review", "flag_reason",
        ],
        "additionalProperties": False,
    }


def build_system_prompt(taxonomy: dict) -> str:
    emotions_block = "\n".join(f"- {e['name']}: {e['description']}" for e in taxonomy["emotions"])
    humor_block = "\n".join(
        f"- {h['name']}: {h['description']}" for h in taxonomy["humor_types"]["types"]
    )
    return f"""You are tagging one chunk of a children's audiobook manuscript for narration \
performance: emotion, intensity, volume, and related delivery cues. This tag drives an \
actual voice performance, so precision matters more than a plausible-sounding guess.

ALLOWED EMOTIONS (choose "emotion" and, optionally, "secondary_emotion" only from this list):
{emotions_block}

HUMOR TYPES (reference only -- a later phase assigns these, you do not need to output one,
but they help you understand the register of a comedic line):
{humor_block}

CRITICAL DISTINCTIONS:

1. intensity vs. volume are INDEPENDENT axes. intensity is emotional AROUSAL (how much the \
character/narrator feels), volume is how LOUD it is actually spoken. Never assume they move \
together, and do not let a low volume pull intensity down with it -- rate them from two \
different questions. A whispered line can be high intensity: a character whispering a final \
goodbye to someone they love, or sharing a tense secret, should get volume="whisper" AND \
intensity="high" at the same time -- the quiet is a performance choice, not a sign the feeling \
is small. A loud line can be low intensity (an enthusiastic but casual greeting, "loud" but not \
emotionally heavy). When a line is emotionally weighty AND quiet, resist the pull to average \
the two into "medium" -- keep volume low and intensity genuinely high.

2. secondary_emotion is rare and optional. Only set it when two feelings genuinely coexist in \
the same line AND share similar energy/direction (e.g. fear + adrenaline, worry riding under \
fading excitement, tenderness + joy). If you find yourself wanting to express two feelings \
that pull in OPPOSITE energetic directions (e.g. calm + excited), that is a signal the chunk \
itself contains two separate beats that should have been split into two chunks upstream -- do \
NOT force an incoherent pairing. Instead leave secondary_emotion null, set flag_for_review to \
true, and explain in flag_reason that the chunk likely needs to be split.

3. Sound effects belong to a character, animal, or object -- NEVER to the narrator's own \
emotional state. Set "type" to "sound_effect" whenever the ENTIRE chunk IS a non-verbal sound \
rather than narration, whether it's spelled out as onomatopoeia (a repeated chant like "Gobble \
gobble gobble!" or "Meep Meep Meep") OR simply narrated in plain words as the chunk's whole \
content (e.g. a standalone chunk that is just "A fart..." or "A loud burp." is still the sound \
itself, not commentary about it -- treat it the same as if it were spelled out). Set "speaker" \
to who or what is making the sound (e.g. "Pharaoh Duck (duckling)", "Tom", "Turkeys", "Tom and \
the kids" for a group). Still fill in "emotion" to describe the vocal color/energy of the sound \
itself (e.g. Curious/Wondering, Excited/Energetic) -- just don't let it read as a human \
narrator's own feeling. If the same kind of sound happens only briefly INLINE inside an \
otherwise normal narrated sentence (e.g. "Presently Peter sneezed 'Kertyschoo!'" -- the sneeze \
is one moment inside a longer descriptive sentence, not the whole chunk), do not set type -- \
that chunk is still ordinary narration; just mention the inline sound in "note" if it affects \
delivery. The distinguishing question: is this chunk's text ENTIRELY the sound/the action of \
making it, or is the sound just one word/phrase embedded inside a larger narrated sentence?

4. Sparse, "show don't tell" picture-book prose systematically undersells its own energy. A \
short plain sentence in a modern picture book (e.g. "The court entertains their Duck King.") \
often corresponds to a full illustrated spread doing much of the emotional/narrative work. \
When a line reads flatter than the surrounding scene's stakes would suggest, lean the \
intensity/volume toward what the SCENE implies rather than just what the bare words state, and \
say why in "note".

5. Never exceed the book's emotional ceilings. The book_profile below states an \
intensity_ceiling and volume_ceiling -- these are the most extreme values allowed ANYWHERE in \
this book. Do not tag a value more extreme than the ceiling. It also lists disabled_emotions \
-- never choose one of those for "emotion" or "secondary_emotion".

5a. If book_profile lists any flags_for_review, those are unresolved judgment calls a human \
has not made yet -- not permission to guess an answer. Most commonly this flags whether a \
recurring antagonist/disruptor is a genuine threat or a comic/chaotic one (e.g. "are these \
turkeys real villains or just chaos?"). If a chunk touches that exact ambiguity (you are \
tempted to reach for Villainous/Menacing, or an unusually dark intensity, for that flagged \
antagonist), prefer the less extreme, more comedic/energetic reading UNLESS the text itself is \
unambiguous, and set this chunk's flag_for_review to true with a short flag_reason noting it \
touches the book-level ambiguity -- so a human resolves it consistently across the whole book \
rather than chunk-by-chunk.

6. emphasis is optional -- only the specific word(s) in the line that need vocal stress to land \
correctly (e.g. a punchline word, a deliberately stressed word). Leave it as an empty list \
when nothing needs special stress.

7. note is optional -- use it only when the tag would not be obvious from the text alone (why \
this is funnier/darker/quieter than it reads, a pronunciation-adjacent performance note, etc). \
Leave it null when the tag is self-evident.

Respond with the structured tag for the CURRENT chunk only. The chunks before/after are \
context to help you read tone and continuity -- do not tag them."""


def _format_chunk_context(chunk_id: str, text: str) -> str:
    return f"[{chunk_id}] {text}"


def build_user_message(
    chunk_id: str,
    chunk_text: str,
    before: list[tuple[str, str]],
    after: list[tuple[str, str]],
    book_profile: BookProfile,
    scene_refs: list[dict] | None,
) -> str:
    before_block = "\n".join(_format_chunk_context(cid, text) for cid, text in before) or "(start of book)"
    after_block = "\n".join(_format_chunk_context(cid, text) for cid, text in after) or "(end of book)"
    er = book_profile.emotional_range
    scene_block = json.dumps(scene_refs) if scene_refs else "none detected yet"
    flags_block = "\n".join(f"- {f}" for f in book_profile.flags_for_review) or "none"

    return f"""BOOK PROFILE:
genre: {book_profile.genre}
style_era: {book_profile.style_era}
narrator_stance: {book_profile.narrator_stance}
intensity_ceiling: {er.intensity_ceiling}
volume_ceiling: {er.volume_ceiling}
disabled_emotions: {er.disabled_emotions or "none"}
flags_for_review (unresolved human judgment calls -- see system prompt point 5a):
{flags_block}

CONTEXT -- chunks immediately before:
{before_block}

CONTEXT -- chunks immediately after:
{after_block}

SCENE REFS already known for this chunk: {scene_block}

CHUNK TO TAG [{chunk_id}]:
{chunk_text}"""


def tag_chunk(
    client: anthropic.Anthropic,
    chunk_id: str,
    chunk_text: str,
    before: list[tuple[str, str]],
    after: list[tuple[str, str]],
    book_profile: BookProfile,
    scene_refs: list[dict] | None = None,
    taxonomy: dict | None = None,
    model: str = MODEL,
    is_sound_effect_chunk: bool = False,
) -> TagResult:
    taxonomy = taxonomy or load_taxonomy()
    schema = build_tag_schema(_emotion_names(taxonomy))
    system_prompt = build_system_prompt(taxonomy)
    user_message = build_user_message(chunk_id, chunk_text, before, after, book_profile, scene_refs)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )

    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    tag = ChunkTag(**data)

    return TagResult(
        chunk_id=chunk_id,
        tag=tag,
        validation_flags=_validate(chunk_id, tag, book_profile, is_sound_effect_chunk),
    )


def tag_chunks(
    client: anthropic.Anthropic,
    chunks: list,
    book_profile: BookProfile,
    scene_refs_by_id: dict[str, list[dict]] | None = None,
    taxonomy: dict | None = None,
    model: str = MODEL,
) -> list[TagResult]:
    taxonomy = taxonomy or load_taxonomy()
    scene_refs_by_id = scene_refs_by_id or {}

    results = []
    for i, chunk in enumerate(chunks):
        before = [(c.id, c.text) for c in chunks[max(0, i - CONTEXT_WINDOW):i]]
        after = [(c.id, c.text) for c in chunks[i + 1:i + 1 + CONTEXT_WINDOW]]
        results.append(
            tag_chunk(
                client,
                chunk.id,
                chunk.text,
                before,
                after,
                book_profile,
                scene_refs=scene_refs_by_id.get(chunk.id),
                taxonomy=taxonomy,
                model=model,
                is_sound_effect_chunk=getattr(chunk, "type", None) == "sound_effect",
            )
        )

    return results


def _validate(chunk_id: str, tag: ChunkTag, book_profile: BookProfile, is_sound_effect_chunk: bool) -> list[str]:
    flags = []
    er = book_profile.emotional_range

    if INTENSITY_LEVELS.index(tag.intensity) > INTENSITY_LEVELS.index(er.intensity_ceiling):
        flags.append(f"intensity '{tag.intensity}' exceeds book ceiling '{er.intensity_ceiling}'")
    if VOLUME_LEVELS.index(tag.volume) > VOLUME_LEVELS.index(er.volume_ceiling):
        flags.append(f"volume '{tag.volume}' exceeds book ceiling '{er.volume_ceiling}'")
    if tag.emotion in er.disabled_emotions:
        flags.append(f"emotion '{tag.emotion}' is disabled for this book")
    if tag.secondary_emotion and tag.secondary_emotion in er.disabled_emotions:
        flags.append(f"secondary_emotion '{tag.secondary_emotion}' is disabled for this book")
    if is_sound_effect_chunk and tag.type != "sound_effect":
        flags.append(
            "chunker identified this chunk as a sound_effect but the tag does not set "
            "type='sound_effect' -- possible narrator-emotion misattribution of a non-human sound"
        )
    if tag.flag_for_review:
        flags.append(f"model self-flagged: {tag.flag_reason or '(no reason given)'}")
    if tag.secondary_emotion and frozenset({tag.emotion, tag.secondary_emotion}) in INCOMPATIBLE_SECONDARY_EMOTION_PAIRS:
        flags.append(
            f"'{tag.emotion}' + '{tag.secondary_emotion}' pull in different energetic directions "
            "(warm vs. detached/wry) -- possible chunk boundary problem, not a real blend; "
            "consider whether this chunk should be split"
        )

    return flags
