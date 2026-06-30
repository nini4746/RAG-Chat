"""Structure-aware chunker for 14 CFR (FAA regs). Self-contained."""
import re
from typing import Callable, Optional

# Running page-header lines emitted by the GPO CFR layout. Each printed page
# starts with: <page-number>  then a header line that is EITHER
#   "Federal Aviation Administration, DOT  § 61.23"      (right pages)
#   "14 CFR Ch. I (1-1-25 Edition)  § 61.23"             (left pages)
# Both carry an embedded "§ NN.NNN" that would otherwise be mis-read as a
# section heading, so they MUST be stripped before parsing.
_HDR = re.compile(r"(Federal Aviation Administration, DOT|CFR Ch\.\s*I.*?Edition\))")
_PAGENUM = re.compile(r"^\s*\d{1,4}\s*$")

# A real section heading after cleanup: "§ 91.151 Fuel requirements ..."
#   - § then number, optional trailing lowercase letter (e.g. 121.99a is rare)
#   - then whitespace then a Capital letter (the Title) and NOT "(" (cross-ref)
_HEADING = re.compile(r"§+\s*(\d+\.\d+[a-z]?)\s+(?=[A-Z])")
# Cross-references look like "§ 61.107(b)" or "§ 61.110 of" -> never match above
# because they are followed by "(" or a lowercase word.

# Split points inside a section body: before any paragraph marker
#   (a) (b) ... (1) (2) ... (i) (ii) ...  -- used only to find pack boundaries.
_MARKER = re.compile(r"(?=\((?:[a-z]{1,3}|\d{1,2})\)\s)")
_SENT = re.compile(r"(?<=[.;:])\s+")


def clean_pages(pages: list[str]) -> str:
    """Strip page numbers + running headers per page, de-hyphenate, normalize."""
    out = []
    for raw in pages:
        lines = raw.split("\n")
        kept = []
        for j, ln in enumerate(lines):
            if j < 3 and _PAGENUM.match(ln):
                continue                      # leading page-number line
            if _HDR.search(ln):
                continue                      # running header line (any position)
            kept.append(ln)
        out.append("\n".join(kept))
    text = "\n".join(out)
    text = text.replace("­", "")          # soft hyphen
    text = re.sub(r"-\n", "", text)            # join hyphenated line breaks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)  # newlines -> spaces (titles wrap)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _token_split(body: str, prefix: str, max_tok: int,
                 tok: Callable[[str], int]) -> list[str]:
    """Guaranteed: split body so prefix+piece <= max_tok tokens. Packs whole
    words; if a single word blows the budget it is char-sliced as last resort."""
    out, cur = [], ""
    for w in body.split():
        cand = (cur + " " + w).strip()
        if cur and tok(prefix + cand) > max_tok:
            out.append(cur)
            cur = w
        else:
            cur = cand
        while tok(prefix + cur) > max_tok and " " not in cur:
            out.append(cur[: len(cur) // 2])      # pathological long token
            cur = cur[len(cur) // 2:]
    if cur:
        out.append(cur)
    return out


def _pack(units: list[str], heading: str, target: int, hard: int,
          tok: Optional[Callable[[str], int]], max_tok: int) -> list[str]:
    """Greedily pack units into chunks <= hard chars (and <= max_tok tokens),
    each prefixed with the section heading. Never crosses the section."""
    chunks, cur = [], ""
    prefix = heading + "\n" if heading else ""
    budget = hard - len(prefix)

    def emit(body):
        body = body.strip()
        if not body:
            return
        chunk = prefix + body
        if tok and tok(chunk) > max_tok:
            # token guard (authoritative): word-level token-bounded split
            for piece in _token_split(body, prefix, max_tok, tok):
                chunks.append(prefix + piece)
        else:
            chunks.append(chunk)

    for u in units:
        u = u.strip()
        if not u:
            continue
        if len(u) > budget:                    # oversized single unit
            if cur:
                emit(cur); cur = ""
            for piece in _hard_split(u, budget, target):
                emit(piece)
            continue
        if not cur:
            cur = u
        elif len(cur) + 1 + len(u) <= target or (
                len(cur) < target // 2 and len(cur) + 1 + len(u) <= budget):
            cur += " " + u
        else:
            emit(cur); cur = u
    if cur:
        emit(cur)
    return chunks


def _hard_split(text: str, budget: int, target: int) -> list[str]:
    """Sentence-aware fallback for a unit with no usable paragraph markers."""
    out, cur = [], ""
    for s in _SENT.split(text):
        if len(s) > budget:                    # a single monster sentence
            if cur:
                out.append(cur); cur = ""
            for k in range(0, len(s), target):
                out.append(s[k:k + target])
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= target:
            cur += " " + s
        else:
            out.append(cur); cur = s
    if cur:
        out.append(cur)
    return out


def chunk_legal(pages: list[str], *, source: str = "",
                target_chars: int = 600, max_chars: int = 700,
                min_chars: int = 120,
                token_counter: Optional[Callable[[str], int]] = None,
                max_tokens: int = 250,
                drop_front_matter: bool = True) -> list[dict]:
    """Chunk a 14 CFR PDF (list of per-page extracted text) into section-aware
    records. Returns dicts: {source, section, title, text}.

    - Splits on § section headings; never crosses a § boundary.
    - Packs paragraph units to ~target_chars, hard ceiling max_chars
      (and <= max_tokens via token_counter if supplied -> respects 256-tok cap).
    - Strips repeated page numbers + running headers, de-hyphenates line breaks.
    - Prepends "§ NN.NNN Title" to every chunk for self-describing citation.
    """
    text = clean_pages(pages)

    # Find heading positions.
    marks = [(m.start(), m.end(), m.group(1)) for m in _HEADING.finditer(text)]
    records: list[dict] = []
    if not marks:
        return records

    if not drop_front_matter and marks[0][0] > 0:
        marks = [(0, 0, None)] + marks

    for idx, (hs, he, sec) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(text)
        seg = text[hs:end].strip()
        if sec is None:
            heading, body = "", seg
        else:
            # heading line = "§ NN.NNN Title."  (title runs to first period)
            mtitle = re.match(r"(§+\s*\d+\.\d+[a-z]?\s+[^.]{0,120}?\.)", seg)
            if mtitle:
                heading = re.sub(r"\s+", " ", mtitle.group(1)).strip()
                body = seg[mtitle.end():].strip()
            else:
                heading = re.sub(r"\s+", " ", seg[:60]).strip()
                body = seg[60:].strip()

        full = (heading + " " + body).strip()
        if token_counter is None and len(full) <= max_chars or \
           (token_counter is not None and len(full) <= max_chars and
                token_counter(full) <= max_tokens):
            records.append({"source": source, "section": sec,
                            "title": heading, "text": full})
            continue

        units = [u for u in _MARKER.split(body) if u.strip()]
        if not units:
            units = [body]
        for ch in _pack(units, heading, target_chars, max_chars,
                        token_counter, max_tokens):
            records.append({"source": source, "section": sec,
                            "title": heading, "text": ch})

    # merge a too-small chunk into its previous sibling within the SAME section
    merged: list[dict] = []
    for r in records:
        if (merged and r["section"] is not None and
                merged[-1]["section"] == r["section"] and
                len(r["text"]) < min_chars and
                len(merged[-1]["text"]) + len(r["text"]) <= max_chars):
            cand = merged[-1]["text"] + " " + r["text"].split("\n", 1)[-1]
            if token_counter is None or token_counter(cand) <= max_tokens:
                merged[-1]["text"] = cand
                continue
        merged.append(r)
    return merged
