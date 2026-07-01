"""RAGAS (Sonnet judge) over the full 10-question answer sheet."""
import eval_ragas as R
import app as A

print("JUDGE =", R.JUDGE)

# Ground truth from the user's answer sheet: expected § sections + concise reference.
GT = {
 "What aeronautical experience is required for a private pilot certificate with an airplane single-engine rating?": {
   "sections": ["61.109", "61.107"],
   "reference": "At least 40 hours flight time: 20 hours dual + 10 hours solo. Includes 3 hours cross-country, 3 hours night (one XC over 100 NM, 10 full-stop landings), 3 hours instrument, 3 hours test prep within 2 calendar months; solo includes 5 hours XC, one solo XC of 150 NM total with landings at 3 points and one 50 NM segment, 3 full-stop landings at a towered airport (section 61.109(a); areas of operation 61.107(b)(1))."},
 "For private-pilot airplane aeronautical experience, what makes flight time count as “cross-country time”?": {
   "sections": ["61.1"],
   "reference": "Cross-country time: flight by a certificated pilot, a landing at a point other than departure, using pilotage/dead reckoning/electronic navigation. For private-pilot airplane, the landing point must be more than 50 NM straight-line from departure (section 61.1)."},
 "What recent flight experience must a pilot have to carry passengers, and what changes at night?": {
   "sections": ["61.57"],
   "reference": "To carry passengers as PIC: 3 takeoffs and 3 landings within the preceding 90 days as sole manipulator, same category/class/type; tailwheel landings must be full stop. At night: 3 takeoffs and 3 landings to a full stop between 1 hour after sunset and 1 hour before sunrise within 90 days (section 61.57(a)-(b))."},
 "What does a flight review require, and how often must a pilot complete one to act as PIC?": {
   "sections": ["61.56"],
   "reference": "Flight review: minimum 1 hour ground + 1 hour flight training, review of part 91 rules and maneuvers to show safe exercise of privileges. Must be completed since the beginning of the 24th calendar month before the month of flight to act as PIC (section 61.56(a),(c))."},
 "Which conditions can prevent an applicant from meeting first-class airman medical standards?": {
   "sections": ["67.107", "67.109", "67.111", "67.113"],
   "reference": "Mental (psychosis, bipolar, severe personality disorder, substance dependence/abuse - 67.107), neurologic (epilepsy, unexplained loss of consciousness/nervous-system control - 67.109), cardiovascular (myocardial infarction, angina, treated/significant coronary disease, valve replacement, pacemaker, heart replacement - 67.111), and general (insulin-requiring diabetes, other unsafe disease - 67.113)."},
 "What are the fuel-reserve requirements for VFR flight by day versus night, and how do rotorcraft differ?": {
   "sections": ["91.151"],
   "reference": "Airplane VFR: fuel to first intended landing plus 30 minutes by day or 45 minutes at night at normal cruising speed. Rotorcraft: plus 20 minutes at normal cruising speed (section 91.151(a)-(b))."},
 "Under IFR, when is an alternate airport required, and what fuel must be carried?": {
   "sections": ["91.167", "91.169"],
   "reference": "IFR fuel: complete flight to first airport, then to alternate if required, then 45 minutes (30 for helicopters) at normal cruising speed. Alternate required unless destination has a published/special approach and from 1 hour before to 1 hour after ETA ceiling >=2000 ft and visibility >=3 SM (non-helicopter) (sections 91.167, 91.169)."},
 "If an IFR flight plan needs an alternate, what weather minima must be forecast at the alternate?": {
   "sections": ["91.169"],
   "reference": "At ETA, use published alternate minima; if none specified, non-helicopter standard: precision 600 ft ceiling/2 SM, nonprecision 800 ft ceiling/2 SM. Helicopter: 200 ft above approach minimum and 1 SM. No approach: descent from MEA, approach, land under basic VFR (section 91.169(c))."},
 "How do operating requirements differ between Class B and Class C airspace?": {
   "sections": ["91.130", "91.131"],
   "reference": "Class B: ATC clearance required before entry, pilot certificate minimums, two-way radio, transponder + ADS-B Out (91.131). Class C: establish two-way radio communication before entry and maintain it, transponder per 91.215 and ADS-B Out per 91.225 after 2020 (91.130)."},
 "What must a pilot do before operating in an active restricted area, and how is that different from a prohibited area?": {
   "sections": ["73.13", "91.133"],
   "reference": "Restricted area: obtain advance permission from the using agency or controlling agency before operating between designated altitudes and time (73.13, 91.133). Prohibited area: authorization must come from the using agency; no operation otherwise (73.83, 91.133)."},
}

hdr = f"{'faith':>7} {'relev':>7} {'prec':>7} {'recall':>7} {'corr':>7}  question"
print(hdr)
rows = []
for q, gt in GT.items():
    m = R.run(q, gt)
    f, sup, n = m["faithfulness"]
    rows.append((q, f, m["relevancy"], m["precision"], m["recall"], m["correctness"], sup, n))
    print(f"{f:7.2f} {m['relevancy']:7.2f} {m['precision']:7.2f} "
          f"{(m['recall'] or 0):7.2f} {(m['correctness'] or 0):7.2f}  {q[:44]} (c={sup}/{n})")

def avg(i):
    xs = [r[i] for r in rows if r[i] is not None]
    return sum(xs)/len(xs) if xs else 0.0
print("-"*72)
print(f"{avg(1):7.2f} {avg(2):7.2f} {avg(3):7.2f} {avg(4):7.2f} {avg(5):7.2f}  AVERAGE")

import json
json.dump([{"q":r[0],"faith":r[1],"relev":r[2],"prec":r[3],"recall":r[4],"corr":r[5],
            "sup":r[6],"claims":r[7]} for r in rows],
          open("eval_sheet_ragas_out.json","w"), ensure_ascii=False, indent=2)
