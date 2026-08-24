# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An 11-phase pipeline that turns a children's book manuscript (`.docx`/`.txt`)
into a fully tagged production script (emotion, intensity, volume, speaker,
humor mechanism, illustration inference, pronunciation) for a voice-cloned
audiobook narrator. See README.md for the full phase-by-phase description
and setup instructions.

## Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt

pytest                                    # deterministic unit tests (no API calls)
pytest tests/test_chunker.py              # single test file
pytest tests/test_chunker.py::test_name -v  # single test

ANTHROPIC_API_KEY=sk-... pytest           # also runs live-API end-to-end tests (auto-skipped without a key)
ANTHROPIC_API_KEY=sk-... pytest tests/test_end_to_end.py -k pharaoh_duck

python review_cli.py <book> <chunk_id>    # book: pharaoh_duck | turkey_takeover | peter_rabbit
```

`ANTHROPIC_API_KEY` comes from `.env` (via `python-dotenv`), or export it
directly. Modules that call Claude (`tagger`, `scenes`, `humor`,
`illustration`, `reports.generate_analysis_document`) need it; the rest of
the pipeline is pure deterministic/heuristic Python and needs no API key.

## Architecture

**Data flow is strictly one-directional through `core/`**, orchestrated by
whatever caller runs the phases in sequence (see
`tests/test_end_to_end.py::_run_pipeline` for the canonical call order —
there is no standalone pipeline-runner script yet, only the test suite and
`review_cli.py` invoke phases directly):

```
ingest → chunk → verse → book_profile → scenes → tag → humor →
illustration → pronunciation → overrides (human review) → reports
```

- **`core/ingest.py`** — strips front/back matter and Gutenberg boilerplate; produces `narration_lines` and separately captured `illustration_notes`. Nothing downstream re-reads the raw file.
- **`core/chunker.py`** — the only place `narration_lines` become chunk objects with IDs (`<bookcode>-NNN`). Every later phase operates on chunk IDs, never raw line numbers.
- **`core/book_profile.py`** — computed *before* tagging and used to cap it: `intensity_ceiling`/`volume_ceiling`/`disabled_emotions` are enforced (not just suggested) by `core/tagger.py::_validate()`. Two axes are deliberately independent — narrator delivery style (composed vs. theatrical → volume ceiling) vs. actual story stakes (→ intensity ceiling). A restrained narrator voice does not imply low stakes (see the Peter Rabbit fixture, which pairs `intensity_ceiling: high` with a sub-shout volume ceiling).
- **`core/scenes.py`** — the one whole-manuscript Claude call (everything else is per-chunk with 1-2 chunks of context, or a single manuscript-wide call for the final analysis doc). Finds cross-chunk patterns a local read can't see (delayed joke payoffs, running gags, recurring background threads). Its output feeds both `core/humor.py` (motif-scoped jokes inherit their humor_type from the scene) and `core/reports.py` (`scene_refs` on chunks).
- **`core/tagger.py`** — per-chunk `emotion`/`secondary_emotion`/`intensity`/`volume` via Claude, validated post-hoc against the book profile's ceilings rather than trusting the prompt alone.
- **`core/humor.py`** — a separate axis from `emotion="Humorous/Funny"`: *why* a line is funny (one of 9 mechanisms in the taxonomy), not whether it should read as amused. Inherited from a scene's `comedic_motif` when the chunk is scene-linked; freshly classified otherwise.
- **`core/illustration.py`** — uses an author-provided illustration note verbatim when Phase 1 captured one; otherwise gates a Claude call behind heuristic signals from the taxonomy (sparse/parallel anaphoric structure, concrete noun/verb density) rather than invoking it on every chunk.
- **`core/pronunciation.py`** — three-tier resolution per proper noun / OOV word: CMU dictionary (`pronouncing`) → `data/lexicon/pronunciation_lexicon.json` (global or `book:<title>`-scoped) → flagged for human review. Never silently guesses. Note: a word present in *both* the CMU dictionary and the lexicon resolves at the dictionary tier (dictionary is checked first) — this is intentional, not a bug, and is asserted in `test_end_to_end.py`.
- **`core/overrides.py`** — corrections are strictly additive; the auto-tag is never mutated in place. `resolve_chunk()` computes the effective value per field (override if present, else original). The corrections log (`data/corrections_log.jsonl`, git-ignored) is append-only and doubles as the "current state" store — the latest entry per `chunk_id` is the active override, earlier entries remain as history. This log is also the planned foundation for an in-context-learning feature (see the taxonomy's `roadmap` section) — not yet built.
- **`core/reports.py`** — final assembly only, no new judgment calls except one whole-manuscript Claude call for the analysis write-up. `build_tagged_manuscript()` applies `overrides.resolve_chunk` so the emitted JSON reflects corrections while `corrections_log.jsonl` keeps the original auto-tag; `generate_analysis_document()` produces the human-facing Markdown report (no chunk IDs/tag jargon).

**`data/taxonomy/emotion_taxonomy.json` is the single source of truth** for
the 17-emotion enum, the 9 humor types, the book-profile schema (with
worked example profiles), illustration-detection signals, and verse/prosody
rules. Every module reads it at runtime rather than duplicating any of it
— when changing allowed emotions, humor types, or ceiling logic, edit this
file, not the Python modules that consume it.

**Fixtures double as the test corpus.** `fixtures/{pharaoh_duck,
turkey_takeover, peter_rabbit}/` each have a `source.docx`/`source.txt`
plus a hand-tagged `expected.json`. `test_end_to_end.py` treats tagging as
inherently subjective: it asserts hard *invariants* (valid enum values,
ceilings respected, cross-references resolve to real chunk/scene IDs) but
reports emotion divergences from `expected.json` as a "needs human review"
JSON file under `tests/e2e_output/`, never a hard failure. When adding a
fixture-dependent test, follow this pattern rather than asserting exact
equality with the hand-tagged fixture.

## Conventions worth preserving

- Manuscript text is reproduced verbatim end-to-end — no paraphrasing, no reordering, even when a chunk is split. This matters most for verse/rhyme fixtures where reordering breaks meter.
- Raw URLs are never voiced; strip them from `TEXT` fields rather than reading them character-by-character.
- A sound-cue-only chunk (a bark, a gobble, no real dialogue) defaults `EMOTION` to `Neutral/Narrator`; a character with real quoted dialogue always gets the full emotional range regardless of species.
