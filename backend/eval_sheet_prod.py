"""RAGAS over 10-Q sheet, mirroring the PRODUCTION path:
retrieve → rerank → definition_boost → expand_sections → generate.
(the built-in eval_ragas run() skips the last two — this measures what ships.)"""
import eval_ragas as R
import app as A
from eval_sheet_ragas import GT

print("JUDGE =", R.JUDGE)

# --- robustness: retry the judge on flaky/non-JSON responses (real RAGAS has this) ---
_json_orig, _llm_orig = R._json, R._llm
def _json_retry(system, user, max_tokens=500):
    for _ in range(3):
        try:
            v = _json_orig(system, user, max_tokens)
            if v:
                return v
        except Exception:
            pass
    return []
def _llm_retry(system, user, max_tokens=500):
    for _ in range(3):
        try:
            return _llm_orig(system, user, max_tokens)
        except Exception:
            pass
    return ""
R._json, R._llm = _json_retry, _llm_retry


def run_prod(q, gt):
    do_rewrite = A.should_rewrite(q)
    queries, _ = A.rewrite_cached(q, None) if do_rewrite else ([q], None)
    k_final = A.K_FINAL if do_rewrite else A.K_SIMPLE
    hits = A.rerank(queries, A.retrieve(queries, k=A.RERANK_POOL), k_final)
    if hits and max(h["score"] for h in hits) >= A.OOS_FLOOR:
        hits = A.expand_sections(queries, A.definition_boost(q, queries, hits))  # <-- prod steps
    context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    uc = f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"
    gen_model = A.MODEL_HQ if A.is_hard(q) else A.MODEL
    r = A.client.messages.create(model=gen_model, max_tokens=1200, system=A.SYSTEM_PROMPT,
                                 messages=[{"role": "user", "content": uc}])
    answer = r.content[0].text
    return {
        "faithfulness": R.faithfulness(answer, context),
        "relevancy": R.answer_relevancy(q, answer),
        "precision": R.context_precision(q, hits),
        "recall": R.context_recall(hits, gt.get("sections")),
        "correctness": R.answer_correctness(answer, gt.get("reference")),
        "nsec": sorted({h.get("section") for h in hits}),
    }


print(f"{'faith':>7} {'relev':>7} {'prec':>7} {'recall':>7} {'corr':>7}  question")
rows = []
for q, gt in GT.items():
    m = run_prod(q, gt)
    f, sup, n = m["faithfulness"]
    rows.append((q, f, m["relevancy"], m["precision"], m["recall"], m["correctness"]))
    print(f"{f:7.2f} {m['relevancy']:7.2f} {m['precision']:7.2f} "
          f"{(m['recall'] or 0):7.2f} {(m['correctness'] or 0):7.2f}  {q[:40]} (c={sup}/{n})")

def avg(i):
    xs=[r[i] for r in rows if r[i] is not None]; return sum(xs)/len(xs) if xs else 0
print("-"*72)
print(f"{avg(1):7.2f} {avg(2):7.2f} {avg(3):7.2f} {avg(4):7.2f} {avg(5):7.2f}  AVERAGE (prod path)")
