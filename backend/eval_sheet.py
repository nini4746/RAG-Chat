"""Run the 10 answer-sheet questions through the live pipeline, print
answer + cited sections + unverified sections for grading."""
import json
import app as A

QS = [
    "What aeronautical experience is required for a private pilot certificate with an airplane single-engine rating?",
    "For private-pilot airplane aeronautical experience, what makes flight time count as “cross-country time”?",
    "What recent flight experience must a pilot have to carry passengers, and what changes at night?",
    "What does a flight review require, and how often must a pilot complete one to act as PIC?",
    "Which conditions can prevent an applicant from meeting first-class airman medical standards?",
    "What are the fuel-reserve requirements for VFR flight by day versus night, and how do rotorcraft differ?",
    "Under IFR, when is an alternate airport required, and what fuel must be carried?",
    "If an IFR flight plan needs an alternate, what weather minima must be forecast at the alternate?",
    "How do operating requirements differ between Class B and Class C airspace?",
    "What must a pilot do before operating in an active restricted area, and how is that different from a prohibited area?",
]


def run(q):
    if A._INJECT.search(q):
        return {"q": q, "blocked": "prompt-injection"}
    do_rewrite = A.should_rewrite(q)
    queries, rw = (A.rewrite_query(q) if do_rewrite else ([q], {"input": 0, "output": 0}))
    k_final = A.K_FINAL if do_rewrite else A.K_SIMPLE
    hits = A.retrieve(queries, k=A.RERANK_POOL)
    hits = A.rerank(queries, hits, k_final)
    if not hits or max(h["score"] for h in hits) < A.OOS_FLOOR:
        return {"q": q, "blocked": "out-of-corpus",
                "best": max((h["score"] for h in hits), default=0)}
    context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"
    resp = A.client.messages.create(model=A.MODEL, max_tokens=1200,
                                    system=A.SYSTEM_PROMPT,
                                    messages=[{"role": "user", "content": user_content}])
    answer = resp.content[0].text
    cites = A._build_citations(answer, hits)
    return {
        "q": q,
        "rewrite": do_rewrite,
        "retrieved_sections": [h.get("section") for h in hits],
        "cited": sorted({(c.get("section"), c["source"]) for c in cites}),
        "unverified": A._unverified_sections(answer, hits),
        "answer": answer,
    }


out = []
for i, q in enumerate(QS, 1):
    print(f"\n{'='*70}\nQ{i}: {q}", flush=True)
    r = run(q)
    out.append(r)
    if "blocked" in r:
        print("BLOCKED:", r["blocked"]); continue
    print("retrieved §:", r["retrieved_sections"])
    print("cited §:", r["cited"])
    print("unverified §:", r["unverified"])
    print("ANSWER:\n" + r["answer"])

with open("eval_sheet_out.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n\nsaved eval_sheet_out.json")
