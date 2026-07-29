#!/usr/bin/env python3
"""Minimal CLI review tool: show a chunk's current auto-tag, let a reviewer
attach an override, and log the correction.

Usage:
    python review_cli.py <book> <chunk_id>

<book> is one of: pharaoh_duck, turkey_takeover, peter_rabbit
"""

import argparse
import sys
from dataclasses import asdict
from datetime import date as date_cls
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic

from core.book_profile import compute_book_profile
from core.chunker import chunk_narration
from core.ingest import ingest
from core.overrides import CORRECTIONS_LOG_PATH, UNSET, attach_override, load_current_overrides, resolve_chunk
from core.tagger import CONTEXT_WINDOW, tag_chunk
from core.verse import compute_verse

FIXTURES = Path(__file__).parent / "fixtures"
BOOKS = {
    "pharaoh_duck": ("source.docx", "pd"),
    "turkey_takeover": ("source.docx", "tt"),
    "peter_rabbit": ("source.txt", "pr"),
}


def load_book(book: str):
    fname, code = BOOKS[book]
    result = ingest(FIXTURES / book / fname)
    chunks = chunk_narration(result.narration_lines, code)
    verse = compute_verse([(c.id, [c.text]) for c in chunks])
    profile = compute_book_profile(result.narration_lines, verse.book_profile.verse_form)
    return chunks, profile


def prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else " [leave blank to keep as-is]"
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", choices=sorted(BOOKS))
    parser.add_argument("chunk_id")
    parser.add_argument(
        "--log-path", type=Path, default=CORRECTIONS_LOG_PATH,
        help="Path to the corrections log (default: data/corrections_log.jsonl)",
    )
    args = parser.parse_args()

    chunks, profile = load_book(args.book)
    idx = next((i for i, c in enumerate(chunks) if c.id == args.chunk_id), None)
    if idx is None:
        print(f"No chunk '{args.chunk_id}' found in {args.book}.", file=sys.stderr)
        sys.exit(1)
    chunk = chunks[idx]
    before = [(c.id, c.text) for c in chunks[max(0, idx - CONTEXT_WINDOW):idx]]
    after = [(c.id, c.text) for c in chunks[idx + 1:idx + 1 + CONTEXT_WINDOW]]

    print(f"\n[{chunk.id}] {chunk.text}\n")

    existing = load_current_overrides(args.log_path).get(chunk.id)
    if existing:
        print(f"NOTE: this chunk already has an override from {existing.reviewer} on {existing.date}:")
        print(f"  note: {existing.note}\n")

    print("Tagging current auto-read (live call)...")
    client = anthropic.Anthropic()
    result = tag_chunk(
        client, chunk.id, chunk.text, before, after, profile,
        is_sound_effect_chunk=chunk.type == "sound_effect",
    )
    tag = result.tag

    print("\nCURRENT AUTO-TAG:")
    for field_name, value in asdict(tag).items():
        print(f"  {field_name}: {value}")
    if result.validation_flags:
        print("  ! validation flags:", result.validation_flags)

    resolved_before = resolve_chunk(chunk.id, tag, existing)
    if existing:
        print("\nEFFECTIVE VALUES (with existing override applied):")
        for field_name in ("emotion", "secondary_emotion", "intensity", "volume", "emphasis"):
            print(f"  {field_name}: {getattr(resolved_before, field_name)}")

    print("\n--- Attach a correction (leave a field blank to keep the auto-tag value) ---")
    reviewer = prompt("Reviewer name")
    while not reviewer:
        reviewer = prompt("Reviewer name (required)")

    kind = ""
    while kind not in ("error_correction", "reviewer_preference"):
        kind = prompt("Kind ('error_correction' or 'reviewer_preference')")

    overrides = {}
    emotion = prompt(f"emotion (auto: {tag.emotion})")
    if emotion:
        overrides["emotion"] = emotion
    secondary = prompt(f"secondary_emotion (auto: {tag.secondary_emotion}; type 'none' to clear)")
    if secondary:
        overrides["secondary_emotion"] = None if secondary.lower() == "none" else secondary
    intensity = prompt(f"intensity (auto: {tag.intensity})")
    if intensity:
        overrides["intensity"] = intensity
    volume = prompt(f"volume (auto: {tag.volume})")
    if volume:
        overrides["volume"] = volume
    emphasis = prompt(f"emphasis, comma-separated (auto: {tag.emphasis})")
    if emphasis:
        overrides["emphasis"] = [w.strip() for w in emphasis.split(",") if w.strip()]

    note = ""
    while not note:
        note = prompt("Note -- why is this being corrected? (required)")

    today = date_cls.today().isoformat()
    date_str = prompt("Date", default=today)

    override = attach_override(
        chunk_id=chunk.id, chunk_text=chunk.text, original_tag=tag,
        reviewer=reviewer, date=date_str, note=note, kind=kind,
        context_before=[t for _, t in before], context_after=[t for _, t in after],
        log_path=args.log_path,
        **{k: v for k, v in overrides.items()},
    )

    resolved = resolve_chunk(chunk.id, tag, override)
    print(f"\nLogged correction to {args.log_path}.")
    print("EFFECTIVE VALUES NOW:")
    for field_name in ("emotion", "secondary_emotion", "intensity", "volume", "emphasis"):
        print(f"  {field_name}: {getattr(resolved, field_name)}")


if __name__ == "__main__":
    main()
