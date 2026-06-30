# RAG Project Report — Apollo corpus

Course: Module 4 · Context Management. Deliverable for "Your job (in order):
run indexer → test app → show successes & failures".

## Setup verified

- **Indexer ran:** `index.pkl` built over `documents/` (20 Wikipedia articles).
  `load_index()` → **993 chunks**.
- **Chunking:** sliding window, `target_chars=1000`, `overlap_chars=100`,
  cut snapped to the last `\n\n` paragraph break in the back third of the window
  (`indexer.py:chunk_text`).
- **Retrieval:** `search(q, INDEX, k=5)` — cosine over
  `sentence-transformers/all-MiniLM-L6-v2` (same model index + query time).
- **Citations:** `_build_citations` parses `[n]` markers, drops out-of-range
  numbers, maps each valid `n` → source filename.
- **App tested** via `test_rag.py` (same retrieve→answer→cite path as
  `backend/app.py`), Claude `claude-sonnet-4-6`.

## 5 test questions

| # | Question | Angle | Verdict |
|---|----------|-------|---------|
| 1 | What was the cause of the Apollo 1 fire? | single-doc factual | **answered well** |
| 2 | Which Apollo missions landed on the Moon? | cross-cutting enumeration | **weak** (incomplete) |
| 3 | Compare moonwalk durations of Apollo 11 vs 17 | cross-doc comparison | **failed** (retrieval miss) |
| 4 | List Apollo missions that used the Saturn V | cross-doc reasoning | **partial** |
| 5 | What is the Artemis program? | "out-of-corpus" probe | **answered** (corpus has it) |

### Q1 — Apollo 1 fire (success)
- Answer: ignition source **electrical** `[3]`; spread by pure-O2 cabin + nylon.
- Cited: `[3] apollo-01.md`, `[1] apollo-01.md`, `[5] apollo-04.md`.
- **Verified:** `apollo-01.md` — "The ignition source of the fire was determined
  to be electrical, and the fire spread rapidly due to combustible nylon material
  and the high-pressure pure oxygen cabin atmosphere." Citation correct.

### Q2 — Which missions landed (weak)
- Answer named only Apollo 11 and 12; noted "five subsequent Apollo missions
  also" but said the sentence was "cut off".
- **Root cause:** the answer-bearing sentence in `apollo-program.md` ("Five
  subsequent Apollo missions also landed astronauts on the Moon, the last,
  Apollo 17 ... twelve people walked on the Moon") was split by chunking, and the
  per-mission docs (14/15/16/17) didn't make top-5. The full answer
  (11, 12, 14, 15, 16, 17) exists in the corpus but retrieval didn't assemble it.

### Q3 — Moonwalk durations (failure)
- Answer gave Apollo 17 (~23h total EVA) but said "no information about Apollo 11".
- **But the data exists:** `apollo-11.md` — "spent around two and a half hours
  walking on the surface". top-5 simply never retrieved `apollo-11.md` for this
  query. Classic retrieval miss on a cross-doc comparison; one verbose doc
  (Apollo 16/17 EVA text) crowded out the Apollo 11 chunk.

### Q4 — Saturn V missions (partial)
- Correctly listed Apollo 4, 6, 8, 16; weakly pulled 14/15 from a retro-rocket
  passage rather than launch-vehicle text. Enumeration incomplete vs the real set.

### Q5 — Artemis (notable)
- README predicted this is **out-of-corpus** and the model "should say it doesn't
  know". In practice the corpus *does* mention Artemis (program est. 2017,
  Artemis II 2026 in `apollo-program.md` / `apollo-17.md`), so the model answered
  with citations rather than refusing. Correct behavior given the actual corpus —
  the "gotcha" no longer holds because the source text changed.

## 2 strengths

1. **Grounded single-doc answers with correct citations.** Q1 is exact and the
   `[n]`→filename mapping is verifiable against the source. Citation parser drops
   invented/out-of-range numbers, so the `Sources:` line is trustworthy.
2. **Honest refusal on missing context.** Q2/Q3 said "not in the provided
   context" instead of hallucinating durations/lists. The system prompt's
   "use ONLY the provided context" rule holds.

## 2 weaknesses

1. **Coverage on cross-cutting / enumeration queries (Q2, Q4).** `k=5` flat
   top-K lets one document dominate; answer-bearing sentences split across chunk
   boundaries never reassemble. "List all X" systematically under-answers.
2. **Retrieval misses on multi-entity comparison (Q3).** Embedding the whole
   comparison question retrieves chunks for the louder entity (Apollo 17) and
   drops the other (Apollo 11) even though it's in the corpus.

## What I would change

- **Chunking:** keep 1000/100 but never split a sentence-final enumeration;
  or index a document-level summary chunk so "list all" queries hit one coherent
  source. Tune chunking before retrieval.
- **Retrieval:** raise `k` (8–10) + per-document budget / MMR diversification so
  Apollo 11 and Apollo 17 both land for comparison queries. For "list all X",
  fan out a retrieval per candidate doc instead of one flat top-K.
- **Query rewrite:** split comparison questions ("A vs B") into per-entity
  sub-queries, retrieve each, then merge context — fixes Q3 directly.
