"""Embedding-model A/B test — does a stronger embedder fix the retrieval-recall gap?

Re-embeds the corpus in memory with each candidate model (index.pkl untouched) and
measures, per model:
  - Context Recall@k : does the ground-truth § section appear in the dense top-k?
  - Definition hit   : for the cross-country question, is the exact "Cross-country
                       time means …" chunk in the top-k? (the known hard miss)

Run: python eval_embed.py   (downloads each candidate once; embeds ~10k chunks/model)
"""
import numpy as np
from sentence_transformers import SentenceTransformer

import app as A
from eval_ragas import GROUND_TRUTH

# (name, query_prefix, passage_prefix) — bge/e5 want instruction prefixes.
CANDIDATES = [
    ("sentence-transformers/all-MiniLM-L6-v2", "", ""),          # current baseline
    ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    ("thenlper/gte-small", "", ""),
    ("intfloat/e5-small-v2", "query: ", "passage: "),
]
K = 8
TEXTS = [r["text"] for r in A.INDEX]
XC_Q = "For private-pilot airplane aeronautical experience, what makes flight time count as cross-country time?"
XC_DEF = [i for i, r in enumerate(A.INDEX)
          if r.get("section") == "61.1" and "cross-country time means" in r["text"].lower()]


def embed_corpus(model, prefix):
    return np.asarray(model.encode([prefix + t for t in TEXTS], batch_size=64,
                                   normalize_embeddings=True, show_progress_bar=False),
                      dtype="float32")


def topk(emb, model, qprefix, q, k=K):
    qv = np.asarray(model.encode([qprefix + q], normalize_embeddings=True), dtype="float32")[0]
    return list(np.argsort(-(emb @ qv))[:k])


def main():
    questions = list(GROUND_TRUTH)
    print(f"{'model':40} {'recall@8':>9} {'xc-def@8':>9}")
    for name, qpre, ppre in CANDIDATES:
        try:
            model = SentenceTransformer(name)
        except Exception as e:
            print(f"{name:40}  load failed: {e}")
            continue
        emb = embed_corpus(model, ppre)
        recs = []
        for q in questions:
            exp = set(GROUND_TRUTH[q]["sections"])
            got = {A.INDEX[i].get("section") for i in topk(emb, model, qpre, q)}
            recs.append(len(exp & got) / len(exp))
        xc_hit = any(i in XC_DEF for i in topk(emb, model, qpre, XC_Q)) if XC_DEF else False
        print(f"{name:40} {sum(recs)/len(recs):9.2f} {('YES' if xc_hit else 'no'):>9}")


if __name__ == "__main__":
    main()
