# RAG-Chat - cited Q&A over FAA regulations (14 CFR)

Retrieval-augmented chat that answers questions about US federal aviation regulations with per-claim citations. The corpus is the official GPO PDFs of 14 CFR (aircraft certification, pilot licensing part 61, medical part 67, airspace parts 71/73, operating rules part 91).

Started from a course starter (RAG over Apollo Wikipedia articles - see `REPORT.md` for that first iteration and its failure analysis), then rebuilt on a harder corpus: legal PDFs with running page headers, section cross-references, and deep paragraph nesting.

## Pipeline

```
question
  → query rewrite (LLM, only when the heuristic says it helps; cached)
  → hybrid retrieval (MiniLM cosine + lexical) over §-level chunks
  → §-dedup → cross-encoder rerank
  → answer generation (Claude) with [n] citation markers, SSE streaming
  → citation builder + unverified-section detector
```

- **Structure-aware chunking** (`chunk_legal.py`) - strips GPO page headers/numbers (which embed fake `§` headings), de-hyphenates, then splits on real `§ NN.NNN Title` headings so every chunk is a legal section; long sections split at paragraph markers `(a) (1) (i)`
- **Citations** - `[n]` markers in the answer are mapped back to the retrieved § sections; sections the answer cites but retrieval can't support are flagged
- **Refusal** - questions the corpus can't ground get a refusal in the user's language instead of a guess

## Evaluation

RAGAS-style metrics implemented locally (`backend/eval_ragas.py`, no `ragas` dependency - reuses the app's Anthropic judge + MiniLM embedder), scored on the live pipeline. Full table in `EVAL_RAGAS.md`:

| Faithfulness | Answer relevancy | Context precision | Context recall | Answer correctness |
|:--:|:--:|:--:|:--:|:--:|
| 0.95 | 0.75 | 0.50 | 0.94 | 0.91 |

Additional eval harnesses in `backend/` (`eval_live.py`, `eval_sheet*.py`, `eval_embed.py`) cover retrieval-only scoring and embedding comparisons.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env   # set ANTHROPIC_API_KEY

python backend/app.py  # Flask, http://localhost:5001 (first run extracts PDFs and builds the index)
```

Frontend (React + Vite):

```bash
cd frontend
npm install
npm run dev
```

## Layout

```
documents/        14 CFR volumes (GPO PDFs)
chunk_legal.py    structure-aware CFR chunker
indexer.py        original corpus indexer (first iteration)
backend/app.py    Flask API: rewrite → retrieve → rerank → generate → cite
backend/eval_*    RAGAS-style + retrieval/embedding eval harnesses
frontend/         chat UI with a Sources panel
EVAL_RAGAS.md     evaluation results
REPORT.md         first-iteration report (Apollo corpus) with failure analysis
```
