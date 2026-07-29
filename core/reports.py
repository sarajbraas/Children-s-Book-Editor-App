"""Final assembly: combine every phase's output into the deliverables an
editor or production team actually receives.

1. build_tagged_manuscript() -- the single tagged JSON file, structurally
   matching the three fixture manuscripts exactly (same top-level shape,
   same per-chunk field names). Chunk-level emotion/secondary_emotion/
   intensity/volume/emphasis are the RESOLVED values (core.overrides.
   resolve_chunk applied) -- if a human has corrected a chunk, the JSON
   reflects that correction, with the original auto-tag preserved
   separately in corrections_log.jsonl (see core.overrides) rather than
   duplicated here, plus an "overrides" sub-object on the chunk itself so
   the correction is visible in context, not just buried in a log file.

2. generate_analysis_document() -- a plain-English Markdown report, no
   chunk IDs or tag jargon in the main body, aimed at a human collaborator
   (editor/author) rather than a production pipeline. Tone/structure is
   modeled on the reference Pharaoh_Duck_Analysis.docx. Picking a story's
   theme and deciding which jokes/threads/non-obvious reads matter enough
   to write up takes real synthesis, so -- like core.scenes -- this is one
   whole-manuscript call to Claude; only the "Tagged Chunk Reference"
   appendix is pure templating off the already-assembled manuscript dict.

3. generate_voice_actor_manual() -- optional Cold Read + Directed Read
   export for narration production.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic

from core.book_profile import BookProfile
from core.humor import ChunkHumor
from core.illustration import ChunkIllustration
from core.overrides import Override, resolve_chunk
from core.scenes import Scene
from core.tagger import ChunkTag
from core.verse import BookVerseProfile, VerseInfo

MODEL = "claude-opus-4-8"


def _build_scene_refs_by_chunk(scenes: list[Scene]) -> dict[str, list[dict]]:
    refs: dict[str, list[dict]] = {}
    for scene in scenes:
        for member in scene.members:
            refs.setdefault(member.chunk_id, []).append({"scene_id": scene.scene_id, "role": member.role})
    return refs


def _build_book_profile_dict(book_profile: BookProfile, verse_profile: BookVerseProfile) -> dict:
    er = book_profile.emotional_range
    result = {
        "genre": book_profile.genre,
        "style_era": book_profile.style_era,
        "narrator_stance": book_profile.narrator_stance,
        "emotional_range": {
            "intensity_ceiling": er.intensity_ceiling,
            "volume_ceiling": er.volume_ceiling,
            "disabled_emotions": er.disabled_emotions,
            "favored_humor_types": er.favored_humor_types,
        },
        "verse_form": verse_profile.verse_form,
        "rhyme_scheme": verse_profile.rhyme_scheme,
        "dominant_meter": verse_profile.dominant_meter,
    }
    if book_profile.flags_for_review:
        result["note"] = "\n".join(book_profile.flags_for_review)
    return result


def _build_scene_dict(scene: Scene) -> dict:
    result = {
        "scene_id": scene.scene_id,
        "type": scene.type,
        "description": scene.description,
        "members": [{"chunk_id": m.chunk_id, "role": m.role} for m in scene.members],
    }
    if scene.humor_type:
        result["humor_type"] = scene.humor_type
    return result


def _build_illustration_dict(illustration: ChunkIllustration) -> dict:
    result = {"implied_scene": illustration.implied_scene, "source": illustration.source}
    if illustration.visual_energy is not None:
        result["visual_energy"] = illustration.visual_energy
    return result


def _build_verse_dict(verse: VerseInfo) -> dict:
    result = {
        "syllables_per_line": verse.syllables_per_line,
        "meter_break": verse.meter_break,
        "rhyme_role": verse.rhyme_role,
    }
    if verse.rhymes_with_chunk:
        result["rhymes_with_chunk"] = verse.rhymes_with_chunk
    return result


def _build_override_dict(override: Override) -> dict:
    from core.overrides import OVERRIDABLE_FIELDS, UNSET

    result = {name: getattr(override, name) for name in OVERRIDABLE_FIELDS if getattr(override, name) is not UNSET}
    if override.pronunciation is not UNSET:
        result["pronunciation"] = [asdict(p) for p in override.pronunciation]
    result["reviewer"] = override.reviewer
    result["date"] = override.date
    result["note"] = override.note
    result["kind"] = override.kind
    return result


def _build_chunk_dict(
    chunk,
    tag: ChunkTag,
    override: Override | None,
    humor_type: str | None,
    illustration: ChunkIllustration | None,
    verse: VerseInfo | None,
    scene_refs: list[dict] | None,
) -> dict:
    resolved = resolve_chunk(chunk.id, tag, override)

    result = {"id": chunk.id, "text": chunk.text}
    result["emotion"] = resolved.emotion
    if resolved.secondary_emotion:
        result["secondary_emotion"] = resolved.secondary_emotion
    result["intensity"] = resolved.intensity
    result["volume"] = resolved.volume
    if resolved.emphasis:
        result["emphasis"] = resolved.emphasis

    if tag.type:
        result["type"] = tag.type
    if tag.speaker:
        result["speaker"] = tag.speaker
    if humor_type:
        result["humor_type"] = humor_type
    if scene_refs:
        result["scene_refs"] = scene_refs
    if illustration is not None:
        result["illustration_inference"] = _build_illustration_dict(illustration)

    note_parts = [p for p in (tag.note, tag.flag_reason if tag.flag_for_review else None) if p]
    if note_parts:
        result["note"] = " ".join(note_parts)
    if tag.flag_for_review:
        result["flag_for_review"] = True

    if verse is not None:
        result["verse"] = _build_verse_dict(verse)
    if override is not None:
        result["overrides"] = _build_override_dict(override)

    return result


def build_tagged_manuscript(
    title: str,
    author: str,
    note: str,
    chunks: list,
    tags_by_id: dict[str, ChunkTag],
    scenes: list[Scene],
    book_profile: BookProfile,
    verse_profile: BookVerseProfile,
    humor_by_id: dict[str, str] | None = None,
    illustration_by_id: dict[str, ChunkIllustration] | None = None,
    verse_by_id: dict[str, VerseInfo] | None = None,
    overrides_by_id: dict[str, Override] | None = None,
) -> dict:
    humor_by_id = humor_by_id or {}
    illustration_by_id = illustration_by_id or {}
    verse_by_id = verse_by_id or {}
    overrides_by_id = overrides_by_id or {}
    scene_refs_by_chunk = _build_scene_refs_by_chunk(scenes)

    return {
        "title": title,
        "author": author,
        "note": note,
        "book_profile": _build_book_profile_dict(book_profile, verse_profile),
        "scenes": [_build_scene_dict(s) for s in scenes],
        "chunks": [
            _build_chunk_dict(
                chunk,
                tags_by_id[chunk.id],
                overrides_by_id.get(chunk.id),
                humor_by_id.get(chunk.id),
                illustration_by_id.get(chunk.id),
                verse_by_id.get(chunk.id),
                scene_refs_by_chunk.get(chunk.id),
            )
            for chunk in chunks
        ],
    }


def write_tagged_manuscript(manuscript: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manuscript, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# generate_analysis_document() -- plain-English companion document.
#
# Tone/structure is modeled directly on the reference analysis document
# (Pharaoh_Duck_Analysis.docx): a title + one-line subtitle, then five
# sections -- "What This Story Is About", "The Voice of This Book", "The
# Humor in This Story", "Recurring Threads to Track Across the Story",
# "Where the Emotional Read Isn't What the Bare Words Suggest" -- followed by
# a fixed closing note and a "Tagged Chunk Reference" appendix.
#
# The appendix is pure templating straight off the already-assembled
# manuscript dict (no judgment involved). The five prose sections require
# real synthesis -- picking a story's theme, selecting which 1-3 jokes/
# threads/non-obvious reads actually matter enough to write up -- so, like
# core.scenes, this is one whole-manuscript call to Claude rather than
# something built out of templates.
# ---------------------------------------------------------------------------

_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}


def build_analysis_schema() -> dict:
    humor_section_schema = {
        "type": "object",
        "properties": {
            "heading": {"type": "string", "description": "short label for this humor mechanism, e.g. 'The royal-naming joke'"},
            "body": {"type": "string", "description": "plain-English explanation of how the joke works and why it's funny, no chunk IDs or jargon"},
            "example_quote": {
                **_NULLABLE_STRING,
                "description": "one line quoted EXACTLY as it appears in the manuscript that best exemplifies this joke's payoff, or null if none fits well",
            },
            "direction_note": {
                **_NULLABLE_STRING,
                "description": "a short, concrete note on how that quoted line should be performed, or null",
            },
        },
        "required": ["heading", "body", "example_quote", "direction_note"],
        "additionalProperties": False,
    }
    thread_schema = {
        "type": "object",
        "properties": {
            "heading": {"type": "string"},
            "body": {"type": "string", "description": "plain-English explanation of the thread and how it should be voiced consistently wherever it recurs"},
        },
        "required": ["heading", "body"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "theme": {
                "type": "string",
                "description": "1-3 short paragraphs of plain prose (no headings) answering: what is this story actually about underneath its plot -- its emotional or thematic core.",
            },
            "voice_of_book": {
                "type": "string",
                "description": "1-2 short paragraphs translating the book's overall narrator tone, energy range, and emotional ceiling into plain prose a narrator/director would understand -- no field names, no jargon, no enum values verbatim.",
            },
            "humor_intro": {
                **_NULLABLE_STRING,
                "description": "optional one-sentence lead-in naming how many humor mechanisms carry the book and their general character (e.g. 'quiet, sincere ones rather than broad jokes'). Null if not useful.",
            },
            "humor_sections": {
                "type": "array",
                "description": "One entry per distinct humor mechanism actually carrying the book. Favor 1-3 substantial mechanisms over an exhaustive catalogue of every joke.",
                "items": humor_section_schema,
            },
            "recurring_threads": {
                "type": "array",
                "description": "Motifs, running bits, or continuity threads a narrator/director should track across the book. Favor genuinely recurring threads over one-off moments.",
                "items": thread_schema,
            },
            "emotional_read_notes": {
                "type": "array",
                "description": "Self-contained plain-English paragraphs (no chunk IDs) covering specific places where the bare words alone would lead to the wrong performance. Favor the most genuinely non-obvious few over an exhaustive list.",
                "items": {"type": "string"},
            },
        },
        "required": ["theme", "voice_of_book", "humor_intro", "humor_sections", "recurring_threads", "emotional_read_notes"],
        "additionalProperties": False,
    }


def build_analysis_system_prompt() -> str:
    return """You are writing a plain-English companion document to a children's-book audio \
production's tagged emotion/prosody data. Your audience is the author, an editor, or a \
narration director -- NOT a developer. They will read this instead of the underlying tagged \
data file, so it needs to stand on its own.

Rules:

1. Never use chunk IDs, field names as jargon (like writing "the intensity is high" instead of \
just describing the feeling), tag enum values verbatim, or any mention of "the model," "the \
pipeline," "the system," "the algorithm," or code. Write as a human collaborator explaining \
their own reasoning to another human they respect.

2. Ground every claim in the tagged data you are given -- the book profile, detected scenes/ \
threads, humor assignments, illustration inferences, and per-chunk emotion/intensity/volume/ \
notes/flags. Do not invent plot details, character traits, or motifs that aren't supported by \
the manuscript text you were given.

3. When you quote a line, quote it EXACTLY as it appears in the manuscript text -- never \
paraphrase a quotation.

4. Be selective, not exhaustive. A good analysis document picks the 1-3 humor mechanisms, \
recurring threads, and non-obvious emotional-read notes that most matter for a narrator to \
know -- not a catalogue of every scene or every joke in the book. Cutting a minor or repetitive \
example is better than diluting the ones that matter.

5. Keep the tone warm, sincere, and specific to this particular story -- avoid generic praise, \
boilerplate, or hedging ("this could be interpreted as..."). Write with the same confidence a \
director would use explaining a choice they've already made.

6. "emotional_read_notes" exist specifically for places where the bare words on the page would \
lead a cold reader to the WRONG performance: a sound effect that's a creature's own voice and \
not a person's fear or excitement, a whispered line that's actually the most emotionally \
significant moment in the book, flat list-like sentences where the illustration is doing \
unstated visual work, or a chunk carrying two feelings at once that look contradictory on \
paper but aren't. Skip anything where the obvious read from the bare text is simply correct --\
 these notes are only for genuine gaps between the words and the intended performance."""


def build_analysis_user_message(manuscript: dict) -> str:
    bp = manuscript["book_profile"]
    er = bp["emotional_range"]

    scenes_block = "\n".join(
        f"- [{s['type']}] {s['scene_id']}: {s['description']}"
        + (f" (humor_type: {s['humor_type']})" if s.get("humor_type") else "")
        for s in manuscript["scenes"]
    ) or "(none detected)"

    chunk_lines = []
    for c in manuscript["chunks"]:
        parts = [f"[{c['id']}]", f'"{c["text"]}"', f"emotion={c['emotion']}"]
        if c.get("secondary_emotion"):
            parts.append(f"secondary_emotion={c['secondary_emotion']}")
        parts.append(f"intensity={c['intensity']}")
        parts.append(f"volume={c['volume']}")
        if c.get("humor_type"):
            parts.append(f"humor_type={c['humor_type']}")
        if c.get("illustration_inference"):
            parts.append(f"illustration_inference={c['illustration_inference']['implied_scene']}")
        if c.get("note"):
            parts.append(f"note={c['note']}")
        if c.get("flag_for_review"):
            parts.append("flagged_for_review=true")
        chunk_lines.append(" ".join(parts))
    chunk_block = "\n".join(chunk_lines)

    book_note = f"\nbook-level note: {bp['note']}" if bp.get("note") else ""

    return f"""BOOK: {manuscript['title']} by {manuscript['author']}

BOOK PROFILE:
genre: {bp['genre']}
style_era: {bp['style_era']}
narrator_stance: {bp['narrator_stance']}
intensity_ceiling: {er['intensity_ceiling']}
volume_ceiling: {er['volume_ceiling']}
disabled_emotions: {er['disabled_emotions']}
favored_humor_types: {er['favored_humor_types']}{book_note}

DETECTED SCENES/THREADS:
{scenes_block}

TAGGED CHUNKS (in reading order):
{chunk_block}

Write the plain-English analysis document content described in your instructions, grounded \
entirely in the data above."""


ANALYSIS_DOCUMENT_NOTE = (
    "This is meant to travel alongside the tagged data as a plain-English companion. Anyone "
    "recording, directing, or reviewing the audio should be able to read this instead of the "
    "underlying data file and understand not just what emotion was chosen for a given line, "
    "but why -- and see the shape of the whole story at a glance."
)

TAGGED_REFERENCE_INTRO = (
    "The same lines above, broken into the individual chunks used to tag the emotion, "
    "intensity, and volume for each one. This is the underlying data behind the analysis "
    "above, line by line."
)

EMOTIONAL_READ_INTRO = (
    "A few places where reading the words alone, without this context, would lead to the "
    "wrong performance:"
)


def _render_humor_section(entry: dict) -> str:
    parts = [f"### {entry['heading']}", "", entry["body"]]
    if entry.get("example_quote"):
        parts += ["", f'> "{entry["example_quote"]}"']
    if entry.get("direction_note"):
        parts += ["", entry["direction_note"]]
    return "\n".join(parts)


def _render_thread_section(entry: dict) -> str:
    return f"### {entry['heading']}\n\n{entry['body']}"


def _render_chunk_reference_entry(chunk: dict) -> str:
    emotion = chunk["emotion"]
    if chunk.get("secondary_emotion"):
        emotion = f"{emotion} + {chunk['secondary_emotion']}"
    lines = [
        f"**{chunk['id']}**",
        f'"{chunk["text"]}"',
        f"{emotion} | {chunk['intensity']} intensity | {chunk['volume']} volume",
    ]
    if chunk.get("note"):
        lines.append(f"Cue: {chunk['note']}")
    return "  \n".join(lines)


def render_analysis_markdown(manuscript: dict, data: dict) -> str:
    sections = [
        f"# {manuscript['title']} -- Narration Analysis",
        "",
        "A plain-English companion to the emotion/prosody tagging, for the author and for the "
        "person producing the voice-clone audio. No code or tags here -- just the reasoning "
        "behind the choices.",
        "",
        "## What This Story Is About",
        "",
        data["theme"],
        "",
        "## The Voice of This Book",
        "",
        data["voice_of_book"],
        "",
        "## The Humor in This Story",
        "",
    ]
    if data.get("humor_intro"):
        sections += [data["humor_intro"], ""]
    for entry in data["humor_sections"]:
        sections += [_render_humor_section(entry), ""]

    sections += ["## Recurring Threads to Track Across the Story", ""]
    for entry in data["recurring_threads"]:
        sections += [_render_thread_section(entry), ""]

    sections += [
        "## Where the Emotional Read Isn't What the Bare Words Suggest",
        "",
        EMOTIONAL_READ_INTRO,
        "",
    ]
    for note in data["emotional_read_notes"]:
        sections += [f"- {note}", ""]

    sections += [
        "## A Note on This Document",
        "",
        ANALYSIS_DOCUMENT_NOTE,
        "",
        "## Tagged Chunk Reference",
        "",
        TAGGED_REFERENCE_INTRO,
        "",
    ]
    chunk_entries = [_render_chunk_reference_entry(c) for c in manuscript["chunks"]]
    sections.append("\n\n---\n\n".join(chunk_entries))
    sections.append("")

    return "\n".join(sections)


@dataclass
class AnalysisDocumentResult:
    markdown: str
    validation_flags: list[str] = field(default_factory=list)


def _validate_quotes(data: dict, manuscript: dict) -> list[str]:
    """Every example_quote must be an exact substring of some chunk's text --
    the model is instructed to quote verbatim (never trust that blindly)."""
    flags = []
    full_text = "\n".join(c["text"] for c in manuscript["chunks"])
    for entry in data["humor_sections"]:
        quote = entry.get("example_quote")
        if quote and quote not in full_text:
            flags.append(
                f"humor section '{entry['heading']}' quotes a line not found verbatim in the "
                f"manuscript: {quote!r}"
            )
    return flags


def generate_analysis_document(
    client: anthropic.Anthropic,
    manuscript: dict,
    model: str = MODEL,
) -> AnalysisDocumentResult:
    schema = build_analysis_schema()
    system_prompt = build_analysis_system_prompt()
    user_message = build_analysis_user_message(manuscript)

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )

    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)

    markdown = render_analysis_markdown(manuscript, data)
    return AnalysisDocumentResult(markdown=markdown, validation_flags=_validate_quotes(data, manuscript))


def write_analysis_document(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
