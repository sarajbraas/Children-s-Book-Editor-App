# Children's Audiobook Prosody Engine

MVP pipeline that turns a children's book manuscript into a fully tagged,
production-ready script for a voice-cloned audiobook narrator.

Given a raw manuscript (`.docx` or `.txt`), the engine reads it the way a
human narrator would prepare a script: who's speaking, what they're
feeling, how loud, how intense, where the jokes land, how a rhyme should
scan, how proper nouns should be pronounced. That output is the reference
material a TTS voice-clone step (not part of this repo) needs to perform
the story rather than just read it aloud.

## Pipeline

Each manuscript runs through eleven phases, most deterministic/heuristic,
a few backed by a Claude API call for real language-understanding judgment
calls (tagging, scene/motif detection, humor mechanism, illustration
inference, final analysis write-up):

1. **Ingest** (`core/ingest.py`) — extract narration text from `.docx`/`.txt`, strip front/back matter and Gutenberg boilerplate, separate out author illustration notes.
2. **Chunk** (`core/chunker.py`) — group narration into sequential, narrator-sized chunks (roughly one spoken breath/beat each).
3. **Verse** (`core/verse.py`) — syllable counts, rhyme detection, and meter breaks for rhyming/verse manuscripts.
4. **Book profile** (`core/book_profile.py`) — sets the book's emotional ceiling and narrator stance (genre, style era, intensity/volume ceilings) before any chunk-level tagging.
5. **Scenes** (`core/scenes.py`) — whole-manuscript pass over Claude to detect cross-chunk motifs: running gags, delayed payoffs, recurring threads.
6. **Tag** (`core/tagger.py`) — per-chunk emotion/intensity/volume tagging against the fixed 17-emotion taxonomy, validated against the book profile's ceilings.
7. **Humor** (`core/humor.py`) — classifies *why* a "Humorous/Funny" chunk is funny (one of 9 humor mechanisms), inherited from a scene's motif or judged fresh if self-contained.
8. **Illustration** (`core/illustration.py`) — infers what the (unseen) companion illustration likely shows, from an author note when present, otherwise heuristically gated Claude reasoning.
9. **Pronunciation** (`core/pronunciation.py`) — three-tier resolution (CMU dictionary → custom lexicon → flagged for human review) for every proper noun and out-of-vocabulary word.
10. **Overrides** (`core/overrides.py`) — additive human-in-the-loop correction layer; original auto-tags are never mutated, corrections are logged and layered on top.
11. **Reports** (`core/reports.py`) — assembles the final tagged manuscript JSON plus a plain-English analysis Markdown doc for an editor/author.

## Data

- `data/taxonomy/emotion_taxonomy.json` — the single source of truth for the 17 allowed emotions, 9 humor types, book-profile schema, illustration-inference signals, and verse/prosody rules. Every module reads this rather than duplicating it.
- `data/lexicon/pronunciation_lexicon.json` — custom pronunciations, scoped globally or per book.
- `data/corrections_log.jsonl` — append-only log of every human override (generated at runtime, not checked in).

## Fixtures & output

Three manuscripts are used as fixtures throughout development and testing:
`pharaoh_duck` and `turkey_takeover` (original rhyming/prose picture books,
with illustration notes), and `peter_rabbit` (public-domain Gutenberg
prose, no illustration notes, license boilerplate to strip). Each has a
hand-tagged `expected.json` under `fixtures/<book>/` used to report emotion
divergences for human review, not as a hard pass/fail bar. Pipeline runs
write `tagged.json` and `analysis.md` per book under `output/<book>/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with:

```
ANTHROPIC_API_KEY=sk-...
```

## Running

The full pipeline is exercised end-to-end by the test suite (see
`tests/test_end_to_end.py` for the phase-by-phase call sequence to follow
if you're wiring up a standalone runner):

```bash
pytest                          # deterministic unit tests only
ANTHROPIC_API_KEY=sk-... pytest # include live-API end-to-end tests (skipped without a key)
```

To review and correct a single tagged chunk by hand:

```bash
python review_cli.py <book> <chunk_id>   # book: pharaoh_duck | turkey_takeover | peter_rabbit
```

## Status

MVP / actively evolving — see the `roadmap` section of
`data/taxonomy/emotion_taxonomy.json` for features discussed and agreed on
in principle but not yet built (in-context learning from manual overrides
is the main one).
