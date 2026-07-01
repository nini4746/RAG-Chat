"""RAGAS over the 10-Q answer sheet, mirroring the PRODUCTION retrieval path
EXACTLY: retrieve → rerank → OOS guard → definition_boost → expand_sections → generate.
Prints per-question diagnostics (rewrite?/def_boost fired?/expand added?/final chunks/
retrieved §) so the run is auditable. Self-contained: imports only metric fns from
eval_ragas (which is __main__-guarded, so no side-run)."""
import eval_ragas as R
import app as A

print("JUDGE =", R.JUDGE, "| GEN =", A.MODEL, "/ hard→", A.MODEL_HQ)

# retry the judge on flaky/non-JSON responses (real RAGAS ships this)
_json_orig, _llm_orig = R._json, R._llm
def _json_retry(s, u, max_tokens=500):
    for _ in range(3):
        try:
            v = _json_orig(s, u, max_tokens)
            if v:
                return v
        except Exception:
            pass
    return []
def _llm_retry(s, u, max_tokens=500):
    for _ in range(3):
        try:
            return _llm_orig(s, u, max_tokens)
        except Exception:
            pass
    return ""
R._json, R._llm = _json_retry, _llm_retry

GT = {
 "What aeronautical experience is required for a private pilot certificate with an airplane single-engine rating?": {
   "sections": ["61.109", "61.107"],
   "reference": "At least 40 hours flight time: 20 hours dual + 10 hours solo. Includes 3 hours cross-country, 3 hours night (one XC over 100 NM, 10 full-stop landings), 3 hours instrument, 3 hours test prep within 2 calendar months; solo includes 5 hours XC, one solo XC of 150 NM total with landings at 3 points and one 50 NM segment, 3 full-stop landings at a towered airport (61.109(a); 61.107(b)(1))."},
 "For private-pilot airplane aeronautical experience, what makes flight time count as “cross-country time”?": {
   "sections": ["61.1"],
   "reference": "Cross-country time: flight by a certificated pilot, a landing at a point other than departure, using pilotage/dead reckoning/electronic navigation. For private-pilot airplane, the landing point must be more than 50 NM straight-line from departure (61.1)."},
 "What recent flight experience must a pilot have to carry passengers, and what changes at night?": {
   "sections": ["61.57"],
   "reference": "To carry passengers as PIC: 3 takeoffs and 3 landings within the preceding 90 days as sole manipulator, same category/class/type; tailwheel landings full stop. At night: 3 takeoffs and 3 landings to a full stop between 1 hour after sunset and 1 hour before sunrise within 90 days (61.57(a)-(b))."},
 "What does a flight review require, and how often must a pilot complete one to act as PIC?": {
   "sections": ["61.56"],
   "reference": "Flight review: minimum 1 hour ground + 1 hour flight training, review of part 91 rules and maneuvers. Must be completed since the beginning of the 24th calendar month before the month of flight to act as PIC (61.56(a),(c))."},
 "Which conditions can prevent an applicant from meeting first-class airman medical standards?": {
   "sections": ["67.107", "67.109", "67.111", "67.113"],
   "reference": "Mental (psychosis, bipolar, severe personality disorder, substance dependence/abuse - 67.107), neurologic (epilepsy, unexplained loss of consciousness - 67.109), cardiovascular (MI, angina, treated coronary disease, valve replacement, pacemaker, heart replacement - 67.111), general (insulin-requiring diabetes, other unsafe disease - 67.113)."},
 "What are the fuel-reserve requirements for VFR flight by day versus night, and how do rotorcraft differ?": {
   "sections": ["91.151"],
   "reference": "Airplane VFR: fuel to first intended landing plus 30 minutes by day or 45 minutes at night. Rotorcraft: plus 20 minutes at normal cruising speed (91.151(a)-(b))."},
 "Under IFR, when is an alternate airport required, and what fuel must be carried?": {
   "sections": ["91.167", "91.169"],
   "reference": "IFR fuel: to first airport, then to alternate if required, then 45 minutes (30 for helicopters). Alternate required unless destination has published/special approach and from 1 hour before to 1 hour after ETA ceiling >=2000 ft and visibility >=3 SM (91.167, 91.169)."},
 "If an IFR flight plan needs an alternate, what weather minima must be forecast at the alternate?": {
   "sections": ["91.169"],
   "reference": "At ETA use published alternate minima; if none, non-helicopter standard: precision 600 ft/2 SM, nonprecision 800 ft/2 SM. Helicopter 200 ft above approach min and 1 SM. No approach: descent from MEA, approach, land under basic VFR (91.169(c))."},
 "How do operating requirements differ between Class B and Class C airspace?": {
   "sections": ["91.130", "91.131"],
   "reference": "Class B: ATC clearance before entry, pilot minimums, two-way radio, transponder + ADS-B Out (91.131). Class C: establish two-way radio before entry and maintain it, transponder per 91.215 and ADS-B per 91.225 after 2020 (91.130)."},
 "What must a pilot do before operating in an active restricted area, and how is that different from a prohibited area?": {
   "sections": ["73.13", "91.133"],
   "reference": "Restricted area: advance permission from using or controlling agency before operating (73.13, 91.133). Prohibited area: authorization from the using agency; no operation otherwise (73.83, 91.133)."},
}


def run_prod(q, gt):
    do_rewrite = A.should_rewrite(q)
    queries, _ = A.rewrite_cached(q, None) if do_rewrite else ([q], None)
    k_final = A.K_FINAL if do_rewrite else A.K_SIMPLE
    core = A.rerank(queries, A.retrieve(queries, k=A.RERANK_POOL), k_final)
    oos = not core or max(h["score"] for h in core) < A.OOS_FLOOR
    hits = core
    n_after_boost = len(core)
    if not oos:
        boosted = A.definition_boost(q, queries, core)
        n_after_boost = len(boosted)
        hits = A.expand_sections(queries, boosted)
    diag = {
        "rewrite": do_rewrite, "n_queries": len(queries), "oos": oos,
        "n_core": len(core), "def_added": n_after_boost - len(core),
        "expand_added": len(hits) - n_after_boost, "n_final": len(hits),
        "secs": sorted({h.get("section") for h in hits}),
    }
    context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    uc = f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"
    gen_model = A.MODEL_HQ if A.is_hard(q) else A.MODEL
    diag["gen"] = gen_model
    r = A.client.messages.create(model=gen_model, max_tokens=1200, system=A.SYSTEM_PROMPT,
                                 messages=[{"role": "user", "content": uc}])
    answer = r.content[0].text
    m = {
        "faithfulness": R.faithfulness(answer, context),
        "relevancy": R.answer_relevancy(q, answer),
        "precision": R.context_precision(q, hits),
        "recall": R.context_recall(hits, gt.get("sections")),
        "correctness": R.answer_correctness(answer, gt.get("reference")),
    }
    return m, diag


rows = []
for i, (q, gt) in enumerate(GT.items(), 1):
    m, d = run_prod(q, gt)
    f, sup, n = m["faithfulness"]
    rows.append((q, f, m["relevancy"], m["precision"], m["recall"], m["correctness"]))
    print(f"\nQ{i}: {q[:60]}")
    print(f"  [audit] rewrite={d['rewrite']}({d['n_queries']}q) gen={d['gen']} "
          f"core={d['n_core']} +def={d['def_added']} +expand={d['expand_added']} "
          f"final={d['n_final']} oos={d['oos']}")
    print(f"  [audit] §={d['secs']}")
    print(f"  faith={f:.2f}({sup}/{n}) relev={m['relevancy']:.2f} prec={m['precision']:.2f} "
          f"recall={(m['recall'] or 0):.2f} corr={(m['correctness'] or 0):.2f}")

def avg(i):
    xs = [r[i] for r in rows if r[i] is not None]
    return sum(xs)/len(xs) if xs else 0.0
print("\n" + "="*60)
print(f"AVERAGE  faith={avg(1):.2f} relev={avg(2):.2f} prec={avg(3):.2f} "
      f"recall={avg(4):.2f} corr={avg(5):.2f}  (prod path, n={len(rows)})")
