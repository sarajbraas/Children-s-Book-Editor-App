import json

import pytest

from core.overrides import (
    UNSET,
    Override,
    PronunciationOverride,
    append_correction_log,
    attach_override,
    load_current_overrides,
    resolve_chunk,
)
from core.tagger import ChunkTag


def _tag(**overrides):
    base = dict(
        emotion="Curious/Wondering", secondary_emotion=None, intensity="medium", volume="normal",
        emphasis=[], type=None, speaker=None, note=None, flag_for_review=False, flag_reason=None,
    )
    base.update(overrides)
    return ChunkTag(**base)


# ---- Override validation ----


def test_override_requires_valid_kind():
    with pytest.raises(ValueError):
        Override(reviewer="Sara", date="2026-08-01", note="x", kind="vibes")


def test_override_requires_reviewer_date_note():
    with pytest.raises(ValueError):
        Override(reviewer="", date="2026-08-01", note="x", kind="error_correction")
    with pytest.raises(ValueError):
        Override(reviewer="Sara", date="", note="x", kind="error_correction")
    with pytest.raises(ValueError):
        Override(reviewer="Sara", date="2026-08-01", note="", kind="error_correction")


# ---- resolve_chunk: additive, never destructive ----


def test_resolve_chunk_with_no_override_returns_original_tag_values():
    tag = _tag(emotion="Tender/Loving", intensity="high")
    resolved = resolve_chunk("pd-013", tag, override=None)
    assert resolved.emotion == "Tender/Loving"
    assert resolved.intensity == "high"


def test_resolve_chunk_uses_override_value_where_set():
    tag = _tag(intensity="medium")
    override = Override(reviewer="Sara", date="2026-08-01", note="cold read felt smaller", kind="reviewer_preference", intensity="low")
    resolved = resolve_chunk("pd-013", tag, override)
    assert resolved.intensity == "low"


def test_resolve_chunk_falls_back_to_original_for_untouched_fields():
    tag = _tag(emotion="Tender/Loving", volume="soft")
    override = Override(reviewer="Sara", date="2026-08-01", note="x", kind="reviewer_preference", intensity="low")
    resolved = resolve_chunk("pd-013", tag, override)
    assert resolved.emotion == "Tender/Loving"  # untouched by override
    assert resolved.volume == "soft"  # untouched by override
    assert resolved.intensity == "low"  # overridden


def test_resolve_chunk_can_explicitly_clear_secondary_emotion():
    # None as an override value must be distinguishable from "not touched".
    tag = _tag(secondary_emotion="Humorous/Funny")
    override = Override(
        reviewer="Sara", date="2026-08-01", note="no real secondary feeling here",
        kind="error_correction", secondary_emotion=None,
    )
    resolved = resolve_chunk("pd-013", tag, override)
    assert resolved.secondary_emotion is None


def test_resolve_chunk_original_tag_object_is_never_mutated():
    tag = _tag(emotion="Tender/Loving", intensity="medium")
    override = Override(reviewer="Sara", date="2026-08-01", note="x", kind="reviewer_preference", emotion="Humorous/Funny", intensity="low")
    resolve_chunk("pd-013", tag, override)
    # The original tag is completely unaffected by resolving with an override.
    assert tag.emotion == "Tender/Loving"
    assert tag.intensity == "medium"


def test_resolve_chunk_includes_pronunciation_override_when_present():
    tag = _tag()
    override = Override(
        reviewer="Sara", date="2026-08-01", note="TTS said this wrong once",
        kind="error_correction", pronunciation=[PronunciationOverride(word="niche", respelling="NEESH")],
    )
    resolved = resolve_chunk("tt-006", tag, override)
    assert resolved.pronunciation_overrides == [PronunciationOverride(word="niche", respelling="NEESH")]


def test_resolve_chunk_pronunciation_defaults_to_empty_list():
    tag = _tag()
    resolved = resolve_chunk("tt-006", tag, override=None)
    assert resolved.pronunciation_overrides == []


# ---- corrections_log.jsonl: append-only, full context ----


def test_append_correction_log_writes_expected_fields(tmp_path):
    log_path = tmp_path / "corrections_log.jsonl"
    tag = _tag(emotion="Tender/Loving", intensity="medium")
    override = Override(reviewer="Sara", date="2026-08-01", note="felt too big", kind="reviewer_preference", intensity="low")

    append_correction_log(
        "pd-013", "Ava wonders if the royal duck knows how to swim yet.", tag, override, log_path,
        context_before=["prev chunk text"], context_after=["next chunk text"],
    )

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["chunk_id"] == "pd-013"
    assert entry["chunk_text"] == "Ava wonders if the royal duck knows how to swim yet."
    assert entry["context_before"] == ["prev chunk text"]
    assert entry["context_after"] == ["next chunk text"]
    assert entry["original_tag"]["emotion"] == "Tender/Loving"
    assert entry["original_tag"]["intensity"] == "medium"
    assert entry["override"] == {"intensity": "low"}
    assert entry["reviewer"] == "Sara"
    assert entry["date"] == "2026-08-01"
    assert entry["note"] == "felt too big"
    assert entry["kind"] == "reviewer_preference"
    assert "logged_at" in entry


def test_append_correction_log_is_append_only(tmp_path):
    log_path = tmp_path / "corrections_log.jsonl"
    tag = _tag()
    override1 = Override(reviewer="Sara", date="2026-08-01", note="first pass", kind="reviewer_preference", intensity="low")
    override2 = Override(reviewer="Sara", date="2026-08-02", note="changed my mind", kind="reviewer_preference", intensity="high")

    append_correction_log("pd-013", "text", tag, override1, log_path)
    append_correction_log("pd-013", "text", tag, override2, log_path)

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2  # both entries preserved, nothing overwritten
    assert json.loads(lines[0])["note"] == "first pass"
    assert json.loads(lines[1])["note"] == "changed my mind"


def test_attach_override_logs_and_returns_the_override(tmp_path):
    log_path = tmp_path / "corrections_log.jsonl"
    tag = _tag(emotion="Curious/Wondering")

    override = attach_override(
        chunk_id="pd-013", chunk_text="some chunk text", original_tag=tag,
        reviewer="Sara", date="2026-08-01", note="misread the tone", kind="error_correction",
        emotion="Playful/Silly", log_path=log_path,
    )

    assert override.emotion == "Playful/Silly"
    assert log_path.read_text().strip() != ""
    resolved = resolve_chunk("pd-013", tag, override)
    assert resolved.emotion == "Playful/Silly"


# ---- load_current_overrides: latest entry per chunk wins, log stays intact ----


def test_load_current_overrides_returns_latest_per_chunk(tmp_path):
    log_path = tmp_path / "corrections_log.jsonl"
    tag = _tag()
    append_correction_log(
        "pd-013", "text", tag,
        Override(reviewer="Sara", date="2026-08-01", note="v1", kind="reviewer_preference", intensity="low"),
        log_path,
    )
    append_correction_log(
        "pd-013", "text", tag,
        Override(reviewer="Sara", date="2026-08-02", note="v2, changed my mind", kind="reviewer_preference", intensity="high"),
        log_path,
    )

    current = load_current_overrides(log_path)
    assert current["pd-013"].intensity == "high"
    assert current["pd-013"].note == "v2, changed my mind"
    # both entries remain on disk even though only the latest is "current"
    assert len(log_path.read_text().splitlines()) == 2


def test_load_current_overrides_empty_when_no_log_file(tmp_path):
    assert load_current_overrides(tmp_path / "does_not_exist.jsonl") == {}


def test_load_current_overrides_round_trips_cleared_secondary_emotion(tmp_path):
    log_path = tmp_path / "corrections_log.jsonl"
    tag = _tag(secondary_emotion="Humorous/Funny")
    append_correction_log(
        "pd-013", "text", tag,
        Override(reviewer="Sara", date="2026-08-01", note="no secondary feeling", kind="error_correction", secondary_emotion=None),
        log_path,
    )
    current = load_current_overrides(log_path)
    resolved = resolve_chunk("pd-013", tag, current["pd-013"])
    assert resolved.secondary_emotion is None


def test_load_current_overrides_round_trips_pronunciation(tmp_path):
    log_path = tmp_path / "corrections_log.jsonl"
    tag = _tag()
    append_correction_log(
        "tt-006", "text", tag,
        Override(
            reviewer="Sara", date="2026-08-01", note="one-off exception", kind="error_correction",
            pronunciation=[PronunciationOverride(word="niche", respelling="NEESH")],
        ),
        log_path,
    )
    current = load_current_overrides(log_path)
    assert current["tt-006"].pronunciation == [PronunciationOverride(word="niche", respelling="NEESH")]
