import os

import pytest

from core.book_profile import BookProfile, EmotionalRange
from core.tagger import (
    ChunkTag,
    _validate,
    build_tag_schema,
    load_taxonomy,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

live_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- live tagging tests need real API credentials",
)


def _profile(intensity_ceiling="high", volume_ceiling="shout", disabled_emotions=None, flags=None):
    return BookProfile(
        genre="fiction",
        style_era="contemporary",
        narrator_stance="close, warm, theatrical",
        emotional_range=EmotionalRange(
            intensity_ceiling=intensity_ceiling,
            volume_ceiling=volume_ceiling,
            disabled_emotions=disabled_emotions or [],
            favored_humor_types=[],
        ),
        flags_for_review=flags or [],
    )


def _tag(**overrides):
    base = dict(
        emotion="Curious/Wondering",
        secondary_emotion=None,
        intensity="medium",
        volume="normal",
        emphasis=[],
        type=None,
        speaker=None,
        note=None,
        flag_for_review=False,
        flag_reason=None,
    )
    base.update(overrides)
    return ChunkTag(**base)


# ---- schema shape (no API needed) ----


def test_tag_schema_includes_all_emotions_as_enum():
    taxonomy = load_taxonomy()
    names = [e["name"] for e in taxonomy["emotions"]]
    schema = build_tag_schema(names)
    assert schema["properties"]["emotion"]["enum"] == names
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())


# ---- _validate() (no API needed) ----


def test_validate_flags_intensity_exceeding_ceiling():
    profile = _profile(intensity_ceiling="medium")
    tag = _tag(intensity="high")
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=False)
    assert any("intensity" in f and "exceeds" in f for f in flags)


def test_validate_flags_volume_exceeding_ceiling():
    profile = _profile(volume_ceiling="normal")
    tag = _tag(volume="shout")
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=False)
    assert any("volume" in f and "exceeds" in f for f in flags)


def test_validate_allows_values_at_the_ceiling():
    profile = _profile(intensity_ceiling="high", volume_ceiling="loud")
    tag = _tag(intensity="high", volume="loud")
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=False)
    assert flags == []


def test_validate_flags_disabled_emotion():
    profile = _profile(disabled_emotions=["Villainous/Menacing"])
    tag = _tag(emotion="Villainous/Menacing")
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=False)
    assert any("disabled" in f for f in flags)


def test_validate_flags_disabled_secondary_emotion():
    profile = _profile(disabled_emotions=["Angry/Frustrated"])
    tag = _tag(secondary_emotion="Angry/Frustrated")
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=False)
    assert any("secondary_emotion" in f and "disabled" in f for f in flags)


def test_validate_flags_sound_effect_mismatch():
    # The duckling-vs-narrator-emotion failure mode: chunker already knows
    # this chunk is a sound effect, but the model's tag doesn't mark it.
    profile = _profile()
    tag = _tag(type=None)
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=True)
    assert any("narrator-emotion misattribution" in f for f in flags)


def test_validate_does_not_flag_correctly_tagged_sound_effect():
    profile = _profile()
    tag = _tag(type="sound_effect", speaker="Duckling")
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=True)
    assert flags == []


def test_validate_surfaces_model_self_flag():
    profile = _profile()
    tag = _tag(flag_for_review=True, flag_reason="opposing emotions, chunk boundary looks wrong")
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=False)
    assert any("model self-flagged" in f and "opposing emotions" in f for f in flags)


def test_validate_flags_known_incompatible_emotion_pair_even_without_self_flag():
    # Tender/Loving + Humorous/Funny (dry-irony register) don't share the
    # same energy/direction -- this must be caught as a code-side safety
    # net even when the model itself didn't flag the pairing as a conflict.
    profile = _profile()
    tag = _tag(emotion="Tender/Loving", secondary_emotion="Humorous/Funny", flag_for_review=False)
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=False)
    assert any("different energetic directions" in f for f in flags)


def test_validate_does_not_flag_compatible_secondary_emotion():
    profile = _profile()
    tag = _tag(emotion="Fearful/Scared", secondary_emotion="Excited/Energetic")
    flags = _validate("x-001", tag, profile, is_sound_effect_chunk=False)
    assert flags == []


# ---- live API tests against real hard cases from the fixtures ----


@live_api
class TestLiveTagging:
    @staticmethod
    @pytest.fixture(scope="class")
    def client():
        import anthropic

        return anthropic.Anthropic()

    @staticmethod
    @pytest.fixture(scope="class")
    def pharaoh_duck():
        from pathlib import Path

        from core.book_profile import compute_book_profile
        from core.chunker import chunk_narration
        from core.ingest import ingest
        from core.verse import compute_verse

        fixtures = Path(__file__).parent.parent / "fixtures"
        result = ingest(fixtures / "pharaoh_duck" / "source.docx")
        chunks = chunk_narration(result.narration_lines, "pd")
        verse = compute_verse([(c.id, [c.text]) for c in chunks])
        profile = compute_book_profile(result.narration_lines, verse.book_profile.verse_form)
        return chunks, profile

    @staticmethod
    @pytest.fixture(scope="class")
    def turkey_takeover():
        from pathlib import Path

        from core.book_profile import compute_book_profile
        from core.chunker import chunk_narration
        from core.ingest import ingest
        from core.verse import compute_verse

        fixtures = Path(__file__).parent.parent / "fixtures"
        result = ingest(fixtures / "turkey_takeover" / "source.docx")
        chunks = chunk_narration(result.narration_lines, "tt")
        verse = compute_verse([(c.id, [c.text]) for c in chunks])
        profile = compute_book_profile(result.narration_lines, verse.book_profile.verse_form)
        return chunks, profile

    @staticmethod
    @pytest.fixture(scope="class")
    def peter_rabbit():
        from pathlib import Path

        from core.book_profile import compute_book_profile
        from core.chunker import chunk_narration
        from core.ingest import ingest
        from core.verse import compute_verse

        fixtures = Path(__file__).parent.parent / "fixtures"
        result = ingest(fixtures / "peter_rabbit" / "source.txt")
        chunks = chunk_narration(result.narration_lines, "pr")
        verse = compute_verse([(c.id, [c.text]) for c in chunks])
        profile = compute_book_profile(result.narration_lines, verse.book_profile.verse_form)
        return chunks, profile

    @staticmethod
    def _tag_by_id(client, chunks, profile, chunk_id):
        from core.tagger import CONTEXT_WINDOW, tag_chunk

        idx = next(i for i, c in enumerate(chunks) if c.id == chunk_id)
        chunk = chunks[idx]
        before = [(c.id, c.text) for c in chunks[max(0, idx - CONTEXT_WINDOW):idx]]
        after = [(c.id, c.text) for c in chunks[idx + 1:idx + 1 + CONTEXT_WINDOW]]
        return tag_chunk(
            client, chunk.id, chunk.text, before, after, profile,
            is_sound_effect_chunk=chunk.type == "sound_effect",
        )

    def test_all_emotion_values_are_from_the_allowed_list(self, client, pharaoh_duck):
        chunks, profile = pharaoh_duck
        taxonomy = load_taxonomy()
        allowed = {e["name"] for e in taxonomy["emotions"]}
        result = self._tag_by_id(client, chunks, profile, "pd-002")
        assert result.tag.emotion in allowed
        if result.tag.secondary_emotion:
            assert result.tag.secondary_emotion in allowed

    def test_duckling_sound_is_not_tagged_as_a_human_emotion(self, client, pharaoh_duck):
        # pd-001: "Meep Meep Meep. Meep Meep Meep." -- the animal's own sound,
        # not the narrator's emotional state.
        chunks, profile = pharaoh_duck
        result = self._tag_by_id(client, chunks, profile, "pd-001")
        assert result.tag.type == "sound_effect"
        assert result.tag.speaker is not None
        assert "duck" in result.tag.speaker.lower()
        assert result.validation_flags == []

    def test_bodily_sound_narrated_in_plain_words_is_still_a_sound_effect(self, client, turkey_takeover):
        # tt-012: "A fart..." -- narrated rather than spelled-out onomatopoeia,
        # but still the sound itself, belonging to Tom.
        chunks, profile = turkey_takeover
        result = self._tag_by_id(client, chunks, profile, "tt-012")
        assert result.tag.type == "sound_effect"
        assert result.validation_flags == []

    def test_inline_onomatopoeia_does_not_split_into_a_sound_effect_chunk(self, client, peter_rabbit):
        # pr-033: 'Presently Peter sneezed "Kertyschoo!"' -- the sneeze is one
        # moment inside a normal narrated sentence, not the whole chunk.
        chunks, profile = peter_rabbit
        result = self._tag_by_id(client, chunks, profile, "pr-033")
        assert result.tag.type is None

    def test_ceilings_are_respected_on_a_high_stakes_line(self, client, peter_rabbit):
        # pr-017: McGregor's "Stop thief!" chase moment -- classic_literary
        # ceiling is intensity=high, volume=loud.
        chunks, profile = peter_rabbit
        result = self._tag_by_id(client, chunks, profile, "pr-017")
        assert not any("exceeds book ceiling" in f for f in result.validation_flags)

    def test_antagonist_ambiguity_flag_reaches_the_tagger(self, client, turkey_takeover):
        # tt-047: the turkey chorus. Turkey Takeover's book_profile carries a
        # flags_for_review entry about whether the turkeys are a real threat
        # or just chaos -- the tag should either avoid Villainous/Menacing as
        # the primary emotion, or self-flag for review if it leans that way,
        # rather than silently committing to one reading.
        chunks, profile = turkey_takeover
        assert profile.flags_for_review  # sanity: the book-level flag exists
        result = self._tag_by_id(client, chunks, profile, "tt-047")
        assert result.tag.type == "sound_effect"
        if result.tag.emotion == "Villainous/Menacing":
            assert result.tag.flag_for_review is True
