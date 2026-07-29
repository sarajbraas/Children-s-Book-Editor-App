"""Manual override layer -- the human-in-the-loop correction step.

Per the taxonomy's "manual_override" section: overrides are ADDITIVE, never
destructive. The original auto-tagged value on a chunk is never edited or
deleted in place -- a correction is layered on top, and resolve_chunk()
computes the effective value per field (the override if one exists,
otherwise the original tag) without ever mutating the original.

Every override is appended to corrections_log.jsonl with enough context
(original tag, corrected value, reviewer's note, surrounding text) to be
useful later for the roadmap's planned in-context-learning feature (not
built here -- this module is purely the groundwork: log everything, lose
nothing). Because the log is append-only, it also doubles as the "current
state" store: the most recent entry per chunk_id is the active override,
while every earlier entry stays in the file as history.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.tagger import ChunkTag

CORRECTIONS_LOG_PATH = Path(__file__).parent.parent / "data" / "corrections_log.jsonl"

VALID_KINDS = ("error_correction", "reviewer_preference")
OVERRIDABLE_FIELDS = ("emotion", "secondary_emotion", "intensity", "volume", "emphasis")


class _Unset:
    """Sentinel distinguishing 'field not touched by this override' from an
    explicit override value of None (e.g. clearing a secondary_emotion)."""

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET: Any = _Unset()


@dataclass
class PronunciationOverride:
    word: str
    respelling: str


@dataclass
class Override:
    reviewer: str
    date: str
    note: str
    kind: str  # "error_correction" | "reviewer_preference"
    emotion: Any = UNSET
    secondary_emotion: Any = UNSET
    intensity: Any = UNSET
    volume: Any = UNSET
    emphasis: Any = UNSET
    pronunciation: Any = UNSET  # list[PronunciationOverride], only for a one-off exception

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got {self.kind!r}")
        if not self.reviewer:
            raise ValueError("reviewer is required")
        if not self.date:
            raise ValueError("date is required")
        if not self.note:
            raise ValueError("note is required")


@dataclass
class ResolvedChunk:
    chunk_id: str
    emotion: str | None
    secondary_emotion: str | None
    intensity: str | None
    volume: str | None
    emphasis: list
    pronunciation_overrides: list


def resolve_chunk(chunk_id: str, tag: ChunkTag, override: Override | None = None) -> ResolvedChunk:
    """The effective value per field: the override if one exists, otherwise
    the original auto-tag. Never mutates `tag` or `override`."""

    def pick(field_name: str, original):
        if override is None:
            return original
        value = getattr(override, field_name)
        return original if value is UNSET else value

    pronunciation = []
    if override is not None and override.pronunciation is not UNSET:
        pronunciation = override.pronunciation

    return ResolvedChunk(
        chunk_id=chunk_id,
        emotion=pick("emotion", tag.emotion),
        secondary_emotion=pick("secondary_emotion", tag.secondary_emotion),
        intensity=pick("intensity", tag.intensity),
        volume=pick("volume", tag.volume),
        emphasis=pick("emphasis", tag.emphasis),
        pronunciation_overrides=pronunciation,
    )


def _changed_fields(override: Override) -> dict:
    changed = {name: getattr(override, name) for name in OVERRIDABLE_FIELDS if getattr(override, name) is not UNSET}
    if override.pronunciation is not UNSET:
        changed["pronunciation"] = [
            asdict(p) if isinstance(p, PronunciationOverride) else p for p in override.pronunciation
        ]
    return changed


def append_correction_log(
    chunk_id: str,
    chunk_text: str,
    original_tag: ChunkTag,
    override: Override,
    log_path: Path = CORRECTIONS_LOG_PATH,
    context_before: list[str] | None = None,
    context_after: list[str] | None = None,
) -> dict:
    """Append one line to the corrections log. Never overwrites or removes
    any prior entry -- this file only ever grows."""
    entry = {
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "context_before": context_before or [],
        "context_after": context_after or [],
        "original_tag": asdict(original_tag),
        "override": _changed_fields(override),
        "reviewer": override.reviewer,
        "date": override.date,
        "note": override.note,
        "kind": override.kind,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def attach_override(
    chunk_id: str,
    chunk_text: str,
    original_tag: ChunkTag,
    reviewer: str,
    date: str,
    note: str,
    kind: str,
    emotion: Any = UNSET,
    secondary_emotion: Any = UNSET,
    intensity: Any = UNSET,
    volume: Any = UNSET,
    emphasis: Any = UNSET,
    pronunciation: Any = UNSET,
    context_before: list[str] | None = None,
    context_after: list[str] | None = None,
    log_path: Path = CORRECTIONS_LOG_PATH,
) -> Override:
    """The main entry point a review interface (CLI or web form) calls:
    build the override, log it, return it so the caller can show the
    reviewer the newly-effective values via resolve_chunk()."""
    override = Override(
        reviewer=reviewer, date=date, note=note, kind=kind,
        emotion=emotion, secondary_emotion=secondary_emotion, intensity=intensity,
        volume=volume, emphasis=emphasis, pronunciation=pronunciation,
    )
    append_correction_log(chunk_id, chunk_text, original_tag, override, log_path, context_before, context_after)
    return override


def load_current_overrides(log_path: Path = CORRECTIONS_LOG_PATH) -> dict[str, Override]:
    """The active override per chunk_id: the most recent log entry wins,
    but nothing is ever deleted from the log itself."""
    if not log_path.exists():
        return {}

    latest_entry_by_chunk: dict[str, dict] = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            latest_entry_by_chunk[entry["chunk_id"]] = entry  # later lines overwrite earlier ones

    result = {}
    for chunk_id, entry in latest_entry_by_chunk.items():
        fields = entry["override"]
        pronunciation = fields.get("pronunciation", UNSET)
        if pronunciation is not UNSET:
            pronunciation = [PronunciationOverride(**p) for p in pronunciation]
        result[chunk_id] = Override(
            reviewer=entry["reviewer"],
            date=entry["date"],
            note=entry["note"],
            kind=entry["kind"],
            emotion=fields.get("emotion", UNSET),
            secondary_emotion=fields.get("secondary_emotion", UNSET),
            intensity=fields.get("intensity", UNSET),
            volume=fields.get("volume", UNSET),
            emphasis=fields.get("emphasis", UNSET),
            pronunciation=pronunciation,
        )
    return result
