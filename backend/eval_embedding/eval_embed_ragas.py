"""Per-embedding-model RAGAS comparison for the FAA 14 CFR RAG pipeline.

Runs the SAME lightweight RAGAS harness as `backend/eval_ragas.py`, but swaps the
corpus + query embedder for each candidate model (on-disk `index.pkl` is untouched —
the corpus is re-embedded in memory). This lifts the question `eval_embed.py` asked at
the retrieval level ("does a stronger embedder fix recall?") up to the ANSWER-QUALITY
level: faithfulness / relevancy / precision / recall / correctness per model.

Everything downstream of embedding — query rewrite, BM25, cross-encoder rerank, the
generator — is held fixed, so score deltas isolate the embedder.

Run: python backend/eval_embedding/eval_embed_ragas.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find app / eval_ragas
import app as A
import eval_ragas as R


def _robust_json(system, user, max_tokens=500):
    """Drop-in for eval_ragas._json that survives a judge that emits prose or a second
    array after the first. The stock _json slices first-'[' .. last-']' and json.loads
    the whole span, which crashes on "Extra data" (sonnet judge is chattier than haiku).
    raw_decode parses exactly the first JSON array and ignores anything trailing."""
    t = R._llm(system, user, max_tokens)
    lo = t.find("[")
    if lo == -1:
        return []
    try:
        val, _ = json.JSONDecoder().raw_decode(t[lo:])
        return val
    except Exception:
        return []


R._json = _robust_json   # metrics resolve `_json` as a module global → picks this up

# (name, query_prefix, passage_prefix) — bge/e5 want instruction prefixes; asymmetric
# models (e5) use a different prefix for the query vs the passage.
CANDIDATES = [
    ("sentence-transformers/all-MiniLM-L6-v2", "", ""),   # current baseline (384-dim)
    ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    ("thenlper/gte-small", "", ""),
    ("intfloat/e5-small-v2", "query: ", "passage: "),
    ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    ("thenlper/gte-base", "", ""),
]

REPORT_PATH = Path(__file__).resolve().parent / "EVAL_EMBEDDING_RAGAS.md"


def set_embedder(name, qpre, ppre):
    """Swap the corpus + query embedder inside the live `app` module. Re-embeds the whole
    corpus in memory (index.pkl on disk stays MiniLM) and clears every embedding-derived
    cache so no baseline vector leaks across models."""
    model = SentenceTransformer(name)
    texts = [r["text"] for r in A.INDEX]
    corpus = model.encode([ppre + t for t in texts], normalize_embeddings=True,
                          batch_size=64, show_progress_bar=False)
    A._EMB = np.asarray(corpus, dtype="float32")
    A._DEF_EMB = A._EMB[A._DEF_IDX] if A._DEF_IDX else None

    def _embed(qs):
        return model.encode([qpre + q for q in qs], normalize_embeddings=True,
                            show_progress_bar=False)
    A.embed = _embed                 # retrieve() / _embed_one() resolve `embed` as a global
    A._embed_one.cache_clear()       # drop the previous model's query vectors
    # Clear every embedding-derived cache that this app build happens to define (the
    # working tree is under active edit — some caches, e.g. _ANSWER_CACHE, may be absent).
    for cache_name in ("_RETRIEVE_CACHE", "_ANSWER_CACHE"):
        c = getattr(A, cache_name, None)
        if c is not None:
            c.clear()


def eval_model():
    """Run the full RAGAS suite over the ground-truth set, return per-metric averages."""
    fs, rs, ps, cr, co = [], [], [], [], []
    for q in R.QUESTIONS:
        try:
            m = R.run(q, R.GROUND_TRUTH[q])
        except Exception as e:
            print(f"    [skip] question failed: {type(e).__name__}: {e}", flush=True)
            continue
        fs.append(m["faithfulness"][0])   # (score, supported, total) -> score
        rs.append(m["relevancy"]); ps.append(m["precision"])
        cr.append(m["recall"]); co.append(m["correctness"])
    return {
        "faithfulness": R._avg(fs), "relevancy": R._avg(rs),
        "precision": R._avg(ps), "recall": R._avg(cr), "correctness": R._avg(co),
    }


def main():
    results = []
    for name, qpre, ppre in CANDIDATES:
        print(f"\n=== {name} ===", flush=True)
        try:
            set_embedder(name, qpre, ppre)
        except Exception as e:
            print(f"  load/embed failed: {e}", flush=True)
            continue
        m = eval_model()
        m["model"] = name
        results.append(m)
        print(f"  faith={m['faithfulness']:.2f} relev={m['relevancy']:.2f} "
              f"prec={m['precision']:.2f} recall={m['recall']:.2f} corr={m['correctness']:.2f}",
              flush=True)

    # Markdown comparison report.
    lines = [
        "# Embedding-model RAGAS comparison — FAA 14 CFR RAG",
        "",
        "Same lightweight RAGAS harness as `backend/eval_ragas.py`, run once per candidate "
        "embedder. The corpus is re-embedded in memory per model (on-disk `index.pkl` stays "
        "MiniLM); everything downstream of embedding — query rewrite, BM25, cross-encoder "
        "rerank, generator — is held fixed, so score deltas isolate the embedder. Ground "
        "truth = 5 well-established 14 CFR provisions (see `eval_ragas.GROUND_TRUTH`).",
        "",
        "| Embedding model | Dim | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |",
        "|---|:--:|:--:|:--:|:--:|:--:|:--:|",
    ]
    base = results[0] if results else None
    dims = {"sentence-transformers/all-MiniLM-L6-v2": 384, "thenlper/gte-small": 384,
            "intfloat/e5-small-v2": 384, "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768, "thenlper/gte-base": 768}
    for m in results:
        tag = " (baseline)" if base and m["model"] == base["model"] else ""
        lines.append(f"| `{m['model']}`{tag} | {dims.get(m['model'], '?')} "
                     f"| {m['faithfulness']:.2f} | {m['relevancy']:.2f} | {m['precision']:.2f} "
                     f"| {m['recall']:.2f} | {m['correctness']:.2f} |")
    lines += [
        "",
        f"Generated by `backend/eval_embedding/eval_embed_ragas.py`. Judge model: "
        f"`{R.JUDGE}`. Metric definitions in `eval_ragas.py`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n→ wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
