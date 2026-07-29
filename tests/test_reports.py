import json

from core.book_profile import BookProfile, EmotionalRange
from core.chunker import Chunk
from core.humor import ChunkHumor
from core.illustration import ChunkIllustration
from core.overrides import Override
from core.reports import (
    _validate_quotes,
    build_analysis_schema,
    build_analysis_user_message,
    build_tagged_manuscript,
    render_analysis_markdown,
    write_tagged_manuscript,
)
from core.scenes import Scene, SceneMember
from core.tagger import ChunkTag
from core.verse import BookVerseProfile, VerseInfo

EXPECTED_TOP_LEVEL_KEYS = {"title", "author", "note", "book_profile", "scenes", "chunks"}
EXPECTED_BOOK_PROFILE_KEYS = {
    "genre", "style_era", "narrator_stance", "emotional_range", "verse_form", "rhyme_scheme", "dominant_meter",
}
EXPECTED_EMOTIONAL_RANGE_KEYS = {"intensity_ceiling", "volume_ceiling", "disabled_emotions", "favored_humor_types"}
EXPECTED_SCENE_KEYS = {"scene_id", "type", "description", "members"}
REQUIRED_CHUNK_KEYS = {"id", "text", "emotion", "intensity", "volume"}
ALLOWED_CHUNK_KEYS = REQUIRED_CHUNK_KEYS | {
    "secondary_emotion", "emphasis", "type", "speaker", "humor_type", "scene_refs",
    "illustration_inference", "note", "flag_for_review", "verse", "overrides",
}


def _profile():
    return BookProfile(
        genre="fiction", style_era="contemporary", narrator_stance="close, warm, theatrical",
        emotional_range=EmotionalRange(
            intensity_ceiling="high", volume_ceiling="shout",
            disabled_emotions=[], favored_humor_types=["incongruity_absurd"],
        ),
    )


def _verse_profile(verse_form=False):
    return BookVerseProfile(verse_form=verse_form, rhyme_scheme=None, dominant_meter=None, meter_break_count=0)


def _tag(**overrides):
    base = dict(
        emotion="Curious/Wondering", secondary_emotion=None, intensity="medium", volume="normal",
        emphasis=[], type=None, speaker=None, note=None, flag_for_review=False, flag_reason=None,
    )
    base.update(overrides)
    return ChunkTag(**base)


def _minimal_manuscript(**kwargs):
    defaults = dict(
        title="Test Book", author="Test Author", note="a test note",
        chunks=[Chunk(id="x-001", text="Hello there.")],
        tags_by_id={"x-001": _tag()},
        scenes=[],
        book_profile=_profile(),
        verse_profile=_verse_profile(),
    )
    defaults.update(kwargs)
    return build_tagged_manuscript(**defaults)


# ---- top-level and book_profile shape ----


def test_top_level_keys_match_fixture_schema():
    manuscript = _minimal_manuscript()
    assert set(manuscript.keys()) == EXPECTED_TOP_LEVEL_KEYS


def test_book_profile_keys_match_fixture_schema_without_flags():
    manuscript = _minimal_manuscript()
    assert set(manuscript["book_profile"].keys()) == EXPECTED_BOOK_PROFILE_KEYS  # no flags -> no "note" key
    assert set(manuscript["book_profile"]["emotional_range"].keys()) == EXPECTED_EMOTIONAL_RANGE_KEYS


def test_book_profile_flags_for_review_become_the_note_field():
    profile = _profile()
    profile.flags_for_review = ["antagonist type ambiguous"]
    manuscript = _minimal_manuscript(book_profile=profile)
    assert manuscript["book_profile"]["note"] == "antagonist type ambiguous"


def test_verse_profile_fields_are_folded_into_book_profile():
    manuscript = _minimal_manuscript(verse_profile=BookVerseProfile(
        verse_form=True, rhyme_scheme="AABB couplets", dominant_meter="~11 syllables", meter_break_count=3
    ))
    bp = manuscript["book_profile"]
    assert bp["verse_form"] is True
    assert bp["rhyme_scheme"] == "AABB couplets"
    assert bp["dominant_meter"] == "~11 syllables"


# ---- scenes ----


def test_scene_dict_matches_fixture_schema():
    scene = Scene(
        scene_id="test-motif", type="comedic_motif", description="a running gag",
        humor_type="Running Gag/Callback",
        members=[SceneMember("x-001", "setup"), SceneMember("x-002", "payoff")],
    )
    manuscript = _minimal_manuscript(
        chunks=[Chunk(id="x-001", text="a"), Chunk(id="x-002", text="b")],
        tags_by_id={"x-001": _tag(), "x-002": _tag()},
        scenes=[scene],
    )
    scene_dict = manuscript["scenes"][0]
    assert EXPECTED_SCENE_KEYS <= set(scene_dict.keys())
    assert scene_dict["humor_type"] == "Running Gag/Callback"
    assert scene_dict["members"] == [{"chunk_id": "x-001", "role": "setup"}, {"chunk_id": "x-002", "role": "payoff"}]


def test_non_comedic_scene_omits_humor_type_key():
    scene = Scene(
        scene_id="test-arc", type="narrative_motif", description="d", humor_type=None,
        members=[SceneMember("x-001", "setup"), SceneMember("x-002", "resolution")],
    )
    manuscript = _minimal_manuscript(
        chunks=[Chunk(id="x-001", text="a"), Chunk(id="x-002", text="b")],
        tags_by_id={"x-001": _tag(), "x-002": _tag()},
        scenes=[scene],
    )
    assert "humor_type" not in manuscript["scenes"][0]


def test_scene_membership_populates_scene_refs_on_member_chunks():
    scene = Scene(
        scene_id="test-motif", type="narrative_motif", description="d", humor_type=None,
        members=[SceneMember("x-001", "setup")],
    )
    manuscript = _minimal_manuscript(scenes=[scene])
    chunk = manuscript["chunks"][0]
    assert chunk["scene_refs"] == [{"scene_id": "test-motif", "role": "setup"}]


# ---- chunk shape: only expected keys, required ones always present ----


def test_chunk_only_contains_allowed_keys():
    manuscript = _minimal_manuscript()
    assert set(manuscript["chunks"][0].keys()) <= ALLOWED_CHUNK_KEYS


def test_chunk_required_keys_always_present():
    manuscript = _minimal_manuscript()
    assert REQUIRED_CHUNK_KEYS <= set(manuscript["chunks"][0].keys())


def test_optional_chunk_fields_omitted_when_not_applicable():
    manuscript = _minimal_manuscript()  # bare tag, no secondary/emphasis/type/etc.
    chunk = manuscript["chunks"][0]
    for key in ("secondary_emotion", "emphasis", "type", "speaker", "humor_type", "scene_refs", "illustration_inference", "note", "flag_for_review", "verse", "overrides"):
        assert key not in chunk


def test_sound_effect_type_and_speaker_included_when_present():
    manuscript = _minimal_manuscript(tags_by_id={"x-001": _tag(type="sound_effect", speaker="Duckling")})
    chunk = manuscript["chunks"][0]
    assert chunk["type"] == "sound_effect"
    assert chunk["speaker"] == "Duckling"


def test_humor_type_included_from_humor_by_id():
    manuscript = _minimal_manuscript(humor_by_id={"x-001": "Dry/Ironic (Deadpan)"})
    assert manuscript["chunks"][0]["humor_type"] == "Dry/Ironic (Deadpan)"


def test_illustration_inference_included_when_present():
    illustration = ChunkIllustration(chunk_id="x-001", implied_scene="a busy scene", source="inferred", visual_energy="high")
    manuscript = _minimal_manuscript(illustration_by_id={"x-001": illustration})
    assert manuscript["chunks"][0]["illustration_inference"] == {
        "implied_scene": "a busy scene", "source": "inferred", "visual_energy": "high",
    }


def test_author_sourced_illustration_omits_visual_energy():
    illustration = ChunkIllustration(chunk_id="x-001", implied_scene="the author's own note", source="author")
    manuscript = _minimal_manuscript(illustration_by_id={"x-001": illustration})
    assert manuscript["chunks"][0]["illustration_inference"] == {
        "implied_scene": "the author's own note", "source": "author",
    }


def test_verse_included_when_present():
    verse = VerseInfo(syllables_per_line=[11, 11], meter_break=False, rhyme_role="self_contained_couplet")
    manuscript = _minimal_manuscript(verse_by_id={"x-001": verse})
    assert manuscript["chunks"][0]["verse"] == {
        "syllables_per_line": [11, 11], "meter_break": False, "rhyme_role": "self_contained_couplet",
    }


def test_flagged_chunk_combines_note_and_flag_reason():
    manuscript = _minimal_manuscript(
        tags_by_id={"x-001": _tag(flag_for_review=True, flag_reason="opposing emotions, chunk boundary looks wrong")}
    )
    chunk = manuscript["chunks"][0]
    assert chunk["flag_for_review"] is True
    assert "opposing emotions" in chunk["note"]


# ---- resolve_chunk() folding: the core new requirement ----


def test_chunk_with_no_override_has_no_overrides_key():
    manuscript = _minimal_manuscript()
    assert "overrides" not in manuscript["chunks"][0]


def test_chunk_top_level_fields_reflect_the_override_not_the_raw_auto_tag():
    tag = _tag(emotion="Curious/Wondering", intensity="medium")
    override = Override(reviewer="Sara", date="2026-08-01", note="felt too big", kind="reviewer_preference", intensity="low")
    manuscript = _minimal_manuscript(tags_by_id={"x-001": tag}, overrides_by_id={"x-001": override})
    chunk = manuscript["chunks"][0]
    assert chunk["intensity"] == "low"  # resolved value, not the raw auto-tag's "medium"
    assert chunk["emotion"] == "Curious/Wondering"  # untouched field falls back to auto-tag


def test_chunk_overrides_subobject_has_reviewer_date_note_kind_and_only_changed_fields():
    tag = _tag(emotion="Curious/Wondering", intensity="medium")
    override = Override(reviewer="Sara", date="2026-08-01", note="felt too big", kind="reviewer_preference", intensity="low")
    manuscript = _minimal_manuscript(tags_by_id={"x-001": tag}, overrides_by_id={"x-001": override})
    overrides_dict = manuscript["chunks"][0]["overrides"]
    assert overrides_dict["intensity"] == "low"
    assert overrides_dict["reviewer"] == "Sara"
    assert overrides_dict["date"] == "2026-08-01"
    assert overrides_dict["note"] == "felt too big"
    assert overrides_dict["kind"] == "reviewer_preference"
    assert "emotion" not in overrides_dict  # untouched field not listed in the override sub-object


def test_original_auto_tag_object_is_never_mutated_by_assembly():
    tag = _tag(emotion="Curious/Wondering", intensity="medium")
    override = Override(reviewer="Sara", date="2026-08-01", note="x", kind="reviewer_preference", intensity="low", emotion="Playful/Silly")
    _minimal_manuscript(tags_by_id={"x-001": tag}, overrides_by_id={"x-001": override})
    assert tag.emotion == "Curious/Wondering"
    assert tag.intensity == "medium"


# ---- write_tagged_manuscript ----


def test_write_tagged_manuscript_produces_valid_json(tmp_path):
    manuscript = _minimal_manuscript()
    output_path = tmp_path / "book" / "tagged.json"
    write_tagged_manuscript(manuscript, output_path)
    assert output_path.exists()
    with open(output_path) as f:
        loaded = json.load(f)
    assert loaded == manuscript


# ---- generate_analysis_document: schema + user message + markdown rendering ----


def _analysis_data(**overrides):
    base = dict(
        theme="A story about imagination.",
        voice_of_book="Warm and close-in.",
        humor_intro="One quiet joke carries the book.",
        humor_sections=[{
            "heading": "The royal-naming joke",
            "body": "Treating an ordinary duck like royalty.",
            "example_quote": "The court has to go home for dinner.",
            "direction_note": "Read it dry.",
        }],
        recurring_threads=[{"heading": "The royal-naming game", "body": "Carried throughout."}],
        emotional_read_notes=["The duckling's sound is not a person's fear."],
    )
    base.update(overrides)
    return base


def test_analysis_schema_marks_nullable_fields_as_required_but_nullable():
    schema = build_analysis_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "theme", "voice_of_book", "humor_intro", "humor_sections", "recurring_threads", "emotional_read_notes",
    }
    humor_intro_schema = schema["properties"]["humor_intro"]
    assert {"type": "null"} in humor_intro_schema["anyOf"]


def test_analysis_user_message_includes_book_profile_and_chunk_text():
    manuscript = _minimal_manuscript(tags_by_id={"x-001": _tag(emotion="Playful/Silly")})
    message = build_analysis_user_message(manuscript)
    assert "Test Book" in message
    assert "Playful/Silly" in message
    assert "Hello there." in message


def test_analysis_user_message_reports_no_scenes_when_none_detected():
    manuscript = _minimal_manuscript()
    message = build_analysis_user_message(manuscript)
    assert "(none detected)" in message


def test_render_analysis_markdown_includes_all_five_sections_and_appendix():
    manuscript = _minimal_manuscript()
    markdown = render_analysis_markdown(manuscript, _analysis_data())
    for heading in (
        "## What This Story Is About",
        "## The Voice of This Book",
        "## The Humor in This Story",
        "## Recurring Threads to Track Across the Story",
        "## Where the Emotional Read Isn't What the Bare Words Suggest",
        "## A Note on This Document",
        "## Tagged Chunk Reference",
    ):
        assert heading in markdown


def test_render_analysis_markdown_never_leaks_chunk_ids_before_the_appendix():
    manuscript = _minimal_manuscript()
    markdown = render_analysis_markdown(manuscript, _analysis_data())
    body, _, appendix = markdown.partition("## Tagged Chunk Reference")
    assert "x-001" not in body
    assert "x-001" in appendix


def test_render_analysis_markdown_omits_humor_intro_when_null():
    manuscript = _minimal_manuscript()
    markdown = render_analysis_markdown(manuscript, _analysis_data(humor_intro=None))
    assert "One quiet joke carries the book." not in markdown


def test_render_analysis_markdown_quotes_example_and_includes_direction_note():
    manuscript = _minimal_manuscript()
    markdown = render_analysis_markdown(manuscript, _analysis_data())
    assert '> "The court has to go home for dinner."' in markdown
    assert "Read it dry." in markdown


def test_render_analysis_markdown_appendix_combines_secondary_emotion_and_cue():
    manuscript = _minimal_manuscript(
        tags_by_id={"x-001": _tag(emotion="Whining/Pleading", secondary_emotion="Sad/Melancholic", note="pleading, with real tears")}
    )
    markdown = render_analysis_markdown(manuscript, _analysis_data())
    assert "Whining/Pleading + Sad/Melancholic | medium intensity | normal volume" in markdown
    assert "Cue: pleading, with real tears" in markdown


# ---- _validate_quotes: never trust the model's quoting blindly ----


def test_validate_quotes_passes_when_quote_is_exact_substring():
    manuscript = _minimal_manuscript()
    data = _analysis_data(humor_sections=[{
        "heading": "h", "body": "b", "example_quote": "Hello there.", "direction_note": None,
    }])
    assert _validate_quotes(data, manuscript) == []


def test_validate_quotes_flags_a_quote_not_found_verbatim_in_the_manuscript():
    manuscript = _minimal_manuscript()
    data = _analysis_data(humor_sections=[{
        "heading": "h", "body": "b", "example_quote": "This line does not exist.", "direction_note": None,
    }])
    flags = _validate_quotes(data, manuscript)
    assert len(flags) == 1
    assert "This line does not exist." in flags[0]


def test_validate_quotes_ignores_sections_with_no_example_quote():
    manuscript = _minimal_manuscript()
    data = _analysis_data(humor_sections=[{
        "heading": "h", "body": "b", "example_quote": None, "direction_note": None,
    }])
    assert _validate_quotes(data, manuscript) == []
