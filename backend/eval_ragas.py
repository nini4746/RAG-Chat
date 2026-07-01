"""RAGAS-style answer-quality metrics for the FAA 14 CFR RAG pipeline.

A lightweight, self-contained implementation of the core RAGAS metrics (no heavy
`ragas`/langchain dependency — reuses the app's Anthropic client + MiniLM embedder):

  - Faithfulness       : share of the answer's atomic claims supported by context
  - Answer Relevancy   : how on-topic the answer is (question ↔ back-generated qs)
  - Context Precision  : are the retrieved passages relevant, and ranked well

Runs the SAME pipeline /api/chat/stream uses, then scores it. Prints a table +
averages so the deck's "검증" step has real numbers.

Run: python eval_ragas.py
"""
import json
from pathlib import Path

import numpy as np

import app as A

JUDGE = A.MODEL_HQ  # sonnet — judge accuracy over cost (offline batch, not latency-bound)


def _llm(system, user, max_tokens=500):
    r = A.client.messages.create(
        model=JUDGE, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return r.content[0].text


def _json(system, user, max_tokens=500):
    t = _llm(system, user, max_tokens)
    lo, hi = t.find("["), t.rfind("]")
    return json.loads(t[lo:hi + 1]) if lo != -1 else []


def _cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def faithfulness(answer, context):
    claims = _json(
        "Extract the atomic factual claims made in the ANSWER. Ignore citation "
        "markers like [1]. Output ONLY a JSON array of short claim strings.",
        f"ANSWER:\n{answer}")
    if not claims:
        return 1.0, 0, 0
    verdicts = _json(
        "For each CLAIM, decide if it is directly supported by the CONTEXT. "
        "Output ONLY a JSON array of booleans (true/false), same order/length as CLAIMS.",
        f"CONTEXT:\n{context}\n\nCLAIMS:\n{json.dumps(claims, ensure_ascii=False)}")
    verdicts = (verdicts + [False] * len(claims))[:len(claims)]
    supported = sum(1 for v in verdicts if v)
    return supported / len(claims), supported, len(claims)


def answer_relevancy(question, answer):
    gen = _json(
        "Generate 3 distinct questions that the ANSWER directly and fully answers. "
        "Output ONLY a JSON array of 3 question strings.",
        f"ANSWER:\n{answer}")
    if not gen:
        return 0.0
    qv = A._embed_one(question)
    sims = [_cos(qv, A._embed_one(g)) for g in gen if isinstance(g, str) and g.strip()]
    return sum(sims) / len(sims) if sims else 0.0


def context_precision(question, hits):
    passages = [h["text"][:600] for h in hits]
    verdicts = _json(
        "For each PASSAGE, decide if it is relevant to answering the QUESTION. "
        "Output ONLY a JSON array of booleans, same order/length as PASSAGES.",
        f"QUESTION:\n{question}\n\nPASSAGES:\n{json.dumps(passages, ensure_ascii=False)}")
    rel = [1 if (i < len(verdicts) and verdicts[i]) else 0 for i in range(len(passages))]
    if not any(rel):
        return 0.0
    # RAGAS context precision = mean of precision@k over the relevant positions.
    running, score = 0, 0.0
    for i, r in enumerate(rel):
        if r:
            running += 1
            score += running / (i + 1)
    return score / sum(rel)


def context_recall(hits, expected_sections):
    """Fraction of the expected ground-truth § sections that retrieval surfaced."""
    if not expected_sections:
        return None
    got = {h.get("section") for h in hits}
    return sum(1 for s in expected_sections if s in got) / len(expected_sections)


def answer_correctness(answer, reference):
    """LLM-graded factual overlap of the answer against a ground-truth reference."""
    if not reference:
        return None
    t = _llm(
        "Compare the ANSWER to the REFERENCE (ground truth). Output ONLY a JSON "
        'object {"score": x} where x in 0.0-1.0 = fraction of the reference\'s key '
        "facts the answer states correctly (penalize contradictions).",
        f"REFERENCE:\n{reference}\n\nANSWER:\n{answer}", max_tokens=120)
    lo, hi = t.find("{"), t.rfind("}")
    try:
        return float(json.loads(t[lo:hi + 1]).get("score"))
    except Exception:
        return None


def run(q, gt=None):
    gt = gt or {}
    do_rewrite = A.should_rewrite(q)
    queries, _ = A.rewrite_cached(q, None) if do_rewrite else ([q], None)
    k_final = A.K_FINAL if do_rewrite else A.K_SIMPLE
    hits = A.rerank(queries, A.retrieve(queries, k=A.RERANK_POOL), k_final)
    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    uc = f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"
    gen_model = A.MODEL_HQ if A.is_hard(q) else A.MODEL   # mirror production escalation
    r = A.client.messages.create(model=gen_model, max_tokens=1200, system=A.SYSTEM_PROMPT,
                                 messages=[{"role": "user", "content": uc}])
    answer = r.content[0].text
    return {
        "faithfulness": faithfulness(answer, context),
        "relevancy": answer_relevancy(q, answer),
        "precision": context_precision(q, hits),
        "recall": context_recall(hits, gt.get("sections")),
        "correctness": answer_correctness(answer, gt.get("reference")),
    }


# Ground truth: expected § sections (Context Recall) + a concise reference answer
# (Answer Correctness). Sections/facts are well-established 14 CFR provisions.
GROUND_TRUTH = {
    "What aeronautical experience is required for a private pilot certificate with an airplane single-engine rating?": {
        "sections": ["61.109"],
        "reference": "At least 40 hours of flight time including 20 hours of flight training and 10 hours of solo flight, with specified cross-country, night, and test-preparation time (section 61.109).",
    },
    "Which medical conditions disqualify an applicant for a first-class airman medical certificate?": {
        "sections": ["67.101", "67.103", "67.105", "67.107", "67.109", "67.111", "67.113"],
        "reference": "Conditions such as diabetes requiring insulin, angina pectoris, myocardial infarction, coronary heart disease requiring treatment, cardiac valve replacement, permanent pacemaker, heart replacement, and disqualifying mental/neurological conditions (part 67 subpart B, sections 67.101–67.115).",
    },
    "What are the fuel-reserve requirements for VFR flight, day versus night?": {
        "sections": ["91.151"],
        "reference": "Enough fuel to fly to the first point of intended landing plus at least 30 minutes (day) or 45 minutes (night) at normal cruising speed (section 91.151).",
    },
    "How do operating requirements differ between Class B and Class C airspace?": {
        "sections": ["91.130", "91.131"],
        "reference": "Class B requires an ATC clearance to enter (section 91.131); Class C requires establishing two-way radio communication with ATC before entering (section 91.130). Both require an operable transponder.",
    },
    "What must a pilot do before operating in an active restricted area?": {
        "sections": ["73.13", "91.133"],
        "reference": "Obtain authorization/permission from the using or controlling agency before operating in an active restricted area (sections 91.133 and 73.13).",
    },
}
QUESTIONS = list(GROUND_TRUTH)


REPORT_PATH = Path(__file__).resolve().parent.parent / "EVAL_RAGAS.md"


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main():
    print(f"{'faith':>7} {'relev':>7} {'prec':>7} {'recall':>7} {'corr':>7}  question")
    rows = []
    fs, rs, ps, cr, co = [], [], [], [], []
    for q in QUESTIONS:
        m = run(q, GROUND_TRUTH[q])
        f, sup, n = m["faithfulness"]
        fs.append(f); rs.append(m["relevancy"]); ps.append(m["precision"])
        cr.append(m["recall"]); co.append(m["correctness"])
        print(f"{f:7.2f} {m['relevancy']:7.2f} {m['precision']:7.2f} "
              f"{(m['recall'] or 0):7.2f} {(m['correctness'] or 0):7.2f}  {q[:44]} (claims={n})")
        rows.append((q, f, m["relevancy"], m["precision"], m["recall"], m["correctness"], sup, n))
    af, ar, ap, acr, aco = _avg(fs), _avg(rs), _avg(ps), _avg(cr), _avg(co)
    print("-" * 72)
    print(f"{af:7.2f} {ar:7.2f} {ap:7.2f} {acr:7.2f} {aco:7.2f}  AVERAGE (RAGAS-style)")

    # Markdown report artifact (deck's "검증" evidence).
    lines = [
        "# RAGAS-style Evaluation — FAA 14 CFR RAG",
        "",
        "Lightweight local implementation of the core RAGAS metrics (no `ragas` "
        "dependency — reuses the app's Anthropic judge + MiniLM embedder). Scores the "
        "live pipeline (rewrite → hybrid retrieve → §-dedup → rerank → generate).",
        "",
        "| Metric | What it measures |",
        "|---|---|",
        "| **Faithfulness** | share of the answer's claims supported by retrieved context |",
        "| **Answer Relevancy** | how on-topic the answer is (question ↔ back-generated questions) |",
        "| **Context Precision** | are retrieved passages relevant, and well-ranked |",
        "| **Context Recall** | share of the expected ground-truth § sections that were retrieved |",
        "| **Answer Correctness** | factual overlap of the answer vs a ground-truth reference |",
        "",
        "## Results",
        "",
        "| # | Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |",
        "|---|---|:--:|:--:|:--:|:--:|:--:|",
    ]
    for i, (q, f, r, p, rc, co, sup, n) in enumerate(rows, 1):
        qshort = (q[:56] + "…") if len(q) > 56 else q
        lines.append(f"| {i} | {qshort} | {f:.2f} ({sup}/{n}) | {r:.2f} | {p:.2f} "
                     f"| {rc:.2f} | {(co if co is not None else 0):.2f} |")
    lines += [
        f"| | **Average** | **{af:.2f}** | **{ar:.2f}** | **{ap:.2f}** | **{acr:.2f}** | **{aco:.2f}** |",
        "",
        "Faithfulness shown as score (supported/total claims). Context Recall uses "
        "ground-truth § sections; Answer Correctness is LLM-graded vs a reference "
        "answer. Generated by `backend/eval_ragas.py`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n→ wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
