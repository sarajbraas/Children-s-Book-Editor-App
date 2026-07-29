import os
from pathlib import Path

import pytest

from core.scenes import (
    MEMBER_ROLES,
    Scene,
    SceneMember,
    _validate,
    build_scene_schema,
    load_taxonomy,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

live_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- live scene-detection tests need real API credentials",
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _chunk(chunk_id):
    return type("C", (), {"id": chunk_id})()


# ---- schema shape (no API needed) ----


def test_scene_schema_role_enum_matches_fixed_role_set():
    taxonomy = load_taxonomy()
    humor_names = [h["name"] for h in taxonomy["humor_types"]["types"]]
    schema = build_scene_schema(humor_names)
    member_schema = schema["properties"]["scenes"]["items"]["properties"]["members"]["items"]
    assert member_schema["properties"]["role"]["enum"] == MEMBER_ROLES


def test_scene_schema_humor_type_enum_matches_taxonomy():
    taxonomy = load_taxonomy()
    humor_names = [h["name"] for h in taxonomy["humor_types"]["types"]]
    schema = build_scene_schema(humor_names)
    scene_schema = schema["properties"]["scenes"]["items"]
    humor_type_field = scene_schema["properties"]["humor_type"]
    assert humor_type_field["anyOf"][0]["enum"] == humor_names
    assert len(humor_names) == 9


# ---- _validate() (no API needed) ----


def test_validate_flags_comedic_motif_missing_humor_type():
    scene = Scene(
        scene_id="x", type="comedic_motif", description="d", humor_type=None,
        members=[SceneMember("c-001", "setup"), SceneMember("c-002", "payoff")],
    )
    flags = _validate([scene], [_chunk("c-001"), _chunk("c-002")])
    assert any("no humor_type" in f for f in flags)


def test_validate_flags_non_comedic_scene_with_humor_type_set():
    scene = Scene(
        scene_id="x", type="narrative_motif", description="d", humor_type="Dry/Ironic (Deadpan)",
        members=[SceneMember("c-001", "setup"), SceneMember("c-002", "resolution")],
    )
    flags = _validate([scene], [_chunk("c-001"), _chunk("c-002")])
    assert any("should be null" in f for f in flags)


def test_validate_flags_scene_with_fewer_than_two_members():
    scene = Scene(
        scene_id="x", type="narrative_motif", description="d", humor_type=None,
        members=[SceneMember("c-001", "setup")],
    )
    flags = _validate([scene], [_chunk("c-001")])
    assert any("fewer than 2 members" in f for f in flags)


def test_validate_flags_unknown_chunk_reference():
    scene = Scene(
        scene_id="x", type="narrative_motif", description="d", humor_type=None,
        members=[SceneMember("c-001", "setup"), SceneMember("c-999", "resolution")],
    )
    flags = _validate([scene], [_chunk("c-001")])
    assert any("unknown chunk_id 'c-999'" in f for f in flags)


def test_validate_flags_duplicate_scene_id():
    scenes = [
        Scene(scene_id="dup", type="narrative_motif", description="d", humor_type=None,
              members=[SceneMember("c-001", "setup"), SceneMember("c-002", "resolution")]),
        Scene(scene_id="dup", type="continuity_thread", description="d2", humor_type=None,
              members=[SceneMember("c-003", "background"), SceneMember("c-004", "background")]),
    ]
    flags = _validate(scenes, [_chunk("c-001"), _chunk("c-002"), _chunk("c-003"), _chunk("c-004")])
    assert any("duplicate scene_id" in f for f in flags)


def test_validate_flags_stylistic_device_with_humor_type_set():
    scene = Scene(
        scene_id="x", type="stylistic_device", description="d", humor_type="Dry/Ironic (Deadpan)",
        members=[SceneMember("c-001", "background"), SceneMember("c-002", "background")],
    )
    flags = _validate([scene], [_chunk("c-001"), _chunk("c-002")])
    assert any("should be null" in f for f in flags)


def test_validate_clean_stylistic_device_scene_has_no_flags():
    scene = Scene(
        scene_id="x", type="stylistic_device", description="d", humor_type=None,
        members=[SceneMember("c-001", "background"), SceneMember("c-002", "background")],
    )
    flags = _validate([scene], [_chunk("c-001"), _chunk("c-002")])
    assert flags == []


def test_validate_clean_scene_has_no_flags():
    scene = Scene(
        scene_id="x", type="comedic_motif", description="d", humor_type="Incongruity/Absurd",
        members=[SceneMember("c-001", "setup"), SceneMember("c-002", "payoff")],
    )
    flags = _validate([scene], [_chunk("c-001"), _chunk("c-002")])
    assert flags == []


# ---- live API test against Pharaoh Duck ----


@live_api
def test_pharaoh_duck_scene_detection_matches_known_fixture_scenes():
    import anthropic

    from core.book_profile import compute_book_profile
    from core.chunker import chunk_narration
    from core.ingest import ingest
    from core.scenes import detect_scenes
    from core.verse import compute_verse

    result = ingest(FIXTURES / "pharaoh_duck" / "source.docx")
    chunks = chunk_narration(result.narration_lines, "pd")
    verse = compute_verse([(c.id, [c.text]) for c in chunks])
    profile = compute_book_profile(result.narration_lines, verse.book_profile.verse_form)

    client = anthropic.Anthropic()
    detection = detect_scenes(client, chunks, profile)

    scenes_by_type = {}
    for scene in detection.scenes:
        scenes_by_type.setdefault(scene.type, []).append(scene)

    # The royal-naming-motif: incongruity humor, several members, spanning
    # most of the book (setup near the start, payoff/callback near the end).
    royal_scenes = [
        s for s in scenes_by_type.get("comedic_motif", [])
        if any("pd-009" in m.chunk_id or "pd-016" in m.chunk_id for m in s.members)
    ]
    assert royal_scenes, "expected a comedic_motif covering the royal-naming bit"
    assert royal_scenes[0].humor_type == "Incongruity/Absurd"
    assert len(royal_scenes[0].members) >= 8

    # The background phone call: a continuity_thread with all "background" roles.
    phone_scenes = [
        s for s in scenes_by_type.get("continuity_thread", [])
        if any("pd-007" in m.chunk_id or "pd-008" in m.chunk_id for m in s.members)
    ]
    assert phone_scenes, "expected a continuity_thread covering the background phone call"
    assert all(m.role == "background" for m in phone_scenes[0].members)
    assert {m.chunk_id for m in phone_scenes[0].members} >= {"pd-007", "pd-008", "pd-011", "pd-014"}

    assert detection.validation_flags == []


@live_api
def test_peter_rabbit_scene_detection_uses_stakes_not_tone_for_type():
    import anthropic

    from core.book_profile import compute_book_profile
    from core.chunker import chunk_narration
    from core.ingest import ingest
    from core.scenes import detect_scenes
    from core.verse import compute_verse

    result = ingest(FIXTURES / "peter_rabbit" / "source.txt")
    chunks = chunk_narration(result.narration_lines, "pr")
    verse = compute_verse([(c.id, [c.text]) for c in chunks])
    profile = compute_book_profile(result.narration_lines, verse.book_profile.verse_form)

    client = anthropic.Anthropic()
    detection = detect_scenes(client, chunks, profile)

    # Real narrative consequence (Peter could lose his clothes/be caught) --
    # dry, ironic delivery doesn't make these comedic_motif; stakes do.
    lost_clothes = [
        s for s in detection.scenes
        if any(m.chunk_id == "pr-053" for m in s.members)  # "second little jacket... in a fortnight"
    ]
    assert lost_clothes, "expected a scene covering the lost jacket/shoes payoff"
    assert lost_clothes[0].type == "narrative_motif"
    assert lost_clothes[0].humor_type is None

    # The omniscient narrator's direct-address asides -- a stylistic_device,
    # not a plot or joke thread.
    narrator_asides = [s for s in detection.scenes if s.type == "stylistic_device"]
    assert narrator_asides, "expected a stylistic_device scene for narrator direct-address"
    assert len(narrator_asides[0].members) >= 2

    assert detection.validation_flags == []
