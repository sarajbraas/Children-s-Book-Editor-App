"""Cross-manuscript scene/motif detection via the Claude API.

Unlike core.tagger (one chunk at a time, with a couple chunks of context),
this runs over the WHOLE manuscript in a single pass. The patterns it looks
for are specifically ones a chunk-local read can't see: a joke whose payoff
depends on a setup established dozens of chunks earlier, a running gag whose
punchline is landing the Nth repetition of an established phrase, a hideout
mentioned once early that becomes the emotional low point when it's invaded
later, a background thread (a phone call, say) that needs to stay
audio-subordinate everywhere it recurs.

Detection needs real thematic/narrative understanding across the whole book,
so -- like core.tagger -- this calls Claude rather than pattern-matching.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from core.book_profile import BookProfile

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "taxonomy" / "emotion_taxonomy.json"

MODEL = "claude-opus-4-8"

SCENE_TYPES = ["comedic_motif", "narrative_motif", "continuity_thread", "stylistic_device"]
MEMBER_ROLES = ["setup", "build", "payoff", "callback", "climax", "resolution", "background"]


@dataclass
class SceneMember:
    chunk_id: str
    role: str


@dataclass
class Scene:
    scene_id: str
    type: str
    description: str
    humor_type: str | None
    members: list[SceneMember]


@dataclass
class SceneDetectionResult:
    scenes: list[Scene] = field(default_factory=list)
    validation_flags: list[str] = field(default_factory=list)


def load_taxonomy(taxonomy_path: Path = TAXONOMY_PATH) -> dict:
    with open(taxonomy_path) as f:
        return json.load(f)


def _humor_type_names(taxonomy: dict) -> list[str]:
    return [h["name"] for h in taxonomy["humor_types"]["types"]]


def build_scene_schema(humor_type_names: list[str]) -> dict:
    member_schema = {
        "type": "object",
        "properties": {
            "chunk_id": {"type": "string"},
            "role": {"type": "string", "enum": MEMBER_ROLES},
        },
        "required": ["chunk_id", "role"],
        "additionalProperties": False,
    }
    scene_schema = {
        "type": "object",
        "properties": {
            "scene_id": {"type": "string"},
            "type": {"type": "string", "enum": SCENE_TYPES},
            "description": {"type": "string"},
            "humor_type": {"anyOf": [{"type": "string", "enum": humor_type_names}, {"type": "null"}]},
            "members": {"type": "array", "items": member_schema},
        },
        "required": ["scene_id", "type", "description", "humor_type", "members"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"scenes": {"type": "array", "items": scene_schema}},
        "required": ["scenes"],
        "additionalProperties": False,
    }


def build_system_prompt(taxonomy: dict) -> str:
    humor_block = "\n".join(
        f"- {h['name']}: {h['description']}" for h in taxonomy["humor_types"]["types"]
    )
    return f"""You are reading an ENTIRE children's audiobook manuscript, already split into \
sequential chunks, to find recurring threads that span multiple chunks. This is specifically \
the kind of pattern a single-chunk read cannot catch: a joke's payoff often depends on a setup \
established many chunks earlier, and a plot detail's significance is often invisible until it \
resurfaces later.

Find four kinds of thread:

1. comedic_motif -- a joke built on a setup-payoff structure spanning multiple chunks. Common \
shapes: an elevated or mismatched register applied consistently to something mundane \
(incongruity humor -- e.g. treating an ordinary pet like royalty throughout the book), or the \
same phrase/sound repeated across the book with a shifting meaning, speaker, or owner (a \
running gag -- e.g. a chant that starts as an antagonist's chaos and ends as the protagonists' \
victory cry). Assign a "humor_type" from this list, matching WHY it's funny:
{humor_block}

2. narrative_motif -- a plot or character detail introduced early that becomes significant \
later for GENUINE STORY reasons, not comedic ones -- a turning point, an emotional low point, a \
resolution. Example: a character's private hideout mentioned once in passing becomes the \
emotional low point of the book when it's invaded later. narrative_motif scenes get \
"humor_type": null.

3. continuity_thread -- a thread that recurs across the book but isn't a joke or a plot beat -- \
it just needs consistent treatment everywhere it appears (most commonly: a background/parallel \
action, like an adult's phone call happening in the background of the main scene, that should \
stay audio-subordinate/muffled every time it recurs, rather than competing with the main \
action). continuity_thread scenes also get "humor_type": null.

4. stylistic_device -- a recurring NARRATOR technique rather than a plot or joke thread: most \
commonly the narrator stepping outside the story to address the reader directly ("I think he \
might have gotten away...", "I am sorry to say..."). These moments want a distinct, knowing, \
slightly-outside-the-story voice each time they recur, separate from the in-scene urgency \
around them. stylistic_device scenes get "humor_type": null; use the "background" role for \
each recurring instance (the same way continuity_thread does).

TIE-BREAKER -- comedic_motif vs. narrative_motif: when a recurring thread has BOTH a comedic \
tone and real narrative substance, decide by STAKES, not by tone. If the scene has genuine \
narrative consequence -- a character could be caught, hurt, lose something that matters, or \
the plot actually turns on it -- classify it as narrative_motif, even when the narration \
delivering it is witty, dry, or ironic. That wit still lives at the individual chunk level \
(a later phase's humor_type/secondary_emotion tags) without making the SCENE ITSELF a \
comedic_motif. Reserve comedic_motif for threads that exist purely as a joke device with no \
real stakes attached -- nobody is actually at risk, and nothing of consequence is won or lost, \
if the gag doesn't land.

For each detected scene, provide:
- scene_id: a short, descriptive, kebab-case slug (e.g. "royal-naming-motif", "hideout-invaded")
- type: one of comedic_motif | narrative_motif | continuity_thread | stylistic_device
- description: plain-English explanation of the pattern and why it matters for narration \
(what changes about how a member chunk should be delivered because it belongs to this scene)
- humor_type: required (non-null) for comedic_motif, must be null for the other three types
- members: every chunk that's part of this thread, each with a "role" from this fixed set: \
setup, build, payoff, callback, climax, resolution, background. Typical usage: comedic_motif \
arcs use setup/build/payoff/callback (a callback is a later chunk that explicitly references \
the motif by name/phrase after it already paid off once); narrative_motif arcs use \
setup/build/climax/resolution; continuity_thread and stylistic_device arcs use background for \
every recurring instance. Pick whichever of the seven roles best fits each member's function \
in the arc -- these are the only allowed values, there is no free-form alternative.

STAY CONSERVATIVE. A real scene needs at least two genuinely connected members (a setup AND a \
payoff, or multiple recurring instances) -- do not invent a motif from a single chunk, and do \
not force a connection between chunks that merely share a topic without an actual structural \
link (repetition, callback, or payoff). It is much better to miss a borderline case than to \
report a scene that doesn't hold up. Only report threads you can point to specific member \
chunks for."""


def build_user_message(chunks: list, book_profile: BookProfile) -> str:
    chunk_block = "\n".join(f"[{c.id}] {c.text}" for c in chunks)
    return f"""BOOK PROFILE:
genre: {book_profile.genre}
style_era: {book_profile.style_era}
narrator_stance: {book_profile.narrator_stance}

FULL MANUSCRIPT (in reading order):
{chunk_block}

Identify every comedic_motif, narrative_motif, and continuity_thread scene in this manuscript."""


def detect_scenes(
    client: anthropic.Anthropic,
    chunks: list,
    book_profile: BookProfile,
    taxonomy: dict | None = None,
    model: str = MODEL,
) -> SceneDetectionResult:
    taxonomy = taxonomy or load_taxonomy()
    schema = build_scene_schema(_humor_type_names(taxonomy))
    system_prompt = build_system_prompt(taxonomy)
    user_message = build_user_message(chunks, book_profile)

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

    scenes = [
        Scene(
            scene_id=s["scene_id"],
            type=s["type"],
            description=s["description"],
            humor_type=s["humor_type"],
            members=[SceneMember(**m) for m in s["members"]],
        )
        for s in data["scenes"]
    ]

    return SceneDetectionResult(scenes=scenes, validation_flags=_validate(scenes, chunks))


def _validate(scenes: list[Scene], chunks: list) -> list[str]:
    flags = []
    valid_ids = {c.id for c in chunks}
    seen_scene_ids: set[str] = set()

    for scene in scenes:
        if scene.scene_id in seen_scene_ids:
            flags.append(f"duplicate scene_id '{scene.scene_id}'")
        seen_scene_ids.add(scene.scene_id)

        if scene.type not in SCENE_TYPES:
            flags.append(f"scene '{scene.scene_id}' has unknown type '{scene.type}'")

        if scene.type == "comedic_motif" and not scene.humor_type:
            flags.append(f"scene '{scene.scene_id}' is comedic_motif but has no humor_type")
        if scene.type != "comedic_motif" and scene.humor_type:
            flags.append(
                f"scene '{scene.scene_id}' is {scene.type} but has humor_type "
                f"'{scene.humor_type}' set -- should be null"
            )

        if len(scene.members) < 2:
            flags.append(
                f"scene '{scene.scene_id}' has fewer than 2 members -- a single chunk isn't "
                "a cross-manuscript thread, possible over-eager detection"
            )

        for member in scene.members:
            if member.chunk_id not in valid_ids:
                flags.append(
                    f"scene '{scene.scene_id}' references unknown chunk_id '{member.chunk_id}'"
                )

    return flags
