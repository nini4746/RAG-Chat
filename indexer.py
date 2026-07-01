"""Context Management RAG starter — indexer.

Walks documents/, chunks each file, embeds chunks, persists the index to disk
so the chat backend can load it without re-indexing.

TODO: implement chunk_text(). The embedding and storage code is provided so
you can focus on the structure.
"""
import pickle
import re
import threading
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from chunk_legal import chunk_legal

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = Path(__file__).parent / "index.pkl"
# Corpus: the 14 CFR (FAA) PDFs live one level up in documents/.
DOCS_DIR = Path(__file__).parent / "documents"


def extract_pdf(path: Path) -> str:
    """Concatenate text from every page, collapsing whitespace so the chunker has
    clean boundaries to snap to. Same extraction the doc-viewer uses, so cited
    previews line up with what's shown."""
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    text = "\n\n".join(t for t in pages if t.strip())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ════════════════════════════════════════════════════════════════
# TODO — implement chunk_text
#
# Split `text` into overlapping chunks. A reasonable default:
#   - ~1000 characters per chunk
#   - ~100 characters of overlap
#   - try to break on paragraph boundaries (\n\n) when possible
#
# Return a list of non-empty strings.
# See the lecture slide on chunking for one working implementation.
# ════════════════════════════════════════════════════════════════

def chunk_text(text: str, target_chars: int = 500, overlap_chars: int = 60) -> list[str]:
    """Split text into ~target_chars chunks, cutting on a natural boundary.

    Smaller, sentence-bounded chunks embed more sharply (the retrieval signal isn't
    diluted by a 1000-char grab-bag) and cost fewer tokens per hit. Each window aims
    for `target_chars`; within the back half it snaps to the last paragraph break,
    else line break, else sentence end — never mid-sentence when a boundary exists.
    The next window starts `overlap_chars` before the cut so context carries over.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + target_chars, n)
        if end < n:
            window_start = start + target_chars // 2
            for sep in ("\n\n", "\n", ". "):
                idx = text.rfind(sep, window_start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


# ════════════════════════════════════════════════════════════════
# Provided: embedding (sentence-transformers, no API key required)
# ════════════════════════════════════════════════════════════════

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    # Double-checked locking so the embedder loads exactly once even under the
    # threaded dev server (two concurrent first calls would otherwise both load it).
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print(f"Loading embedding model ({MODEL_NAME})... (one-time download ~80MB)")
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns unit-normalized 384-dim vectors."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def extract_pages(path: Path) -> list[str]:
    """Per-page extracted text — what the section-aware legal chunker needs to
    strip running page headers correctly."""
    return [p.extract_text() or "" for p in PdfReader(str(path)).pages]


def _ntok(s: str) -> int:
    """Token count under the actual MiniLM tokenizer, so chunks respect the
    embedder's 256-token cap (char caps alone can't bound dense table text)."""
    return len(get_model().tokenizer.encode(s, add_special_tokens=True))


# ════════════════════════════════════════════════════════════════
# Provided: build / save / load / search
# ════════════════════════════════════════════════════════════════

def build_index() -> list[dict]:
    """Walk DOCS_DIR, chunk each file, embed, return list of records.

    PDFs (14 CFR) use the section-aware legal chunker so each chunk is one
    self-contained passage tagged with its § section. Plain text falls back to
    the generic sliding-window chunker.
    """
    records: list[dict] = []
    chunk_id = 0
    for path in sorted(DOCS_DIR.glob("*")):
        suffix = path.suffix.lower()
        if path.is_dir() or suffix not in (".md", ".txt", ".pdf"):
            continue
        if suffix == ".pdf":
            recs = chunk_legal(extract_pages(path), source=path.name, token_counter=_ntok)
            items = [(r["text"], r.get("section"), r.get("title")) for r in recs]
        else:
            items = [(c, None, None) for c in chunk_text(path.read_text())]
        if not items:
            continue
        vectors = embed([text for text, _, _ in items])
        for i, ((text, section, title), vec) in enumerate(zip(items, vectors)):
            records.append({
                "chunk_id": chunk_id,
                "source": path.name,
                "chunk_index": i,
                "section": section,
                "title": title,
                "text": text,
                "embedding": vec,
            })
            chunk_id += 1
        print(f"  {path.name}: {len(items)} chunks")
    return records


def save_index(records: list[dict]) -> None:
    with INDEX_PATH.open("wb") as f:
        pickle.dump(records, f)


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run `python indexer.py` from the project root first."
        )
    with INDEX_PATH.open("rb") as f:
        return pickle.load(f)


def cosine_distance(a: list[float], b: list[float]) -> float:
    # Both vectors are unit-normalized, so cosine distance == 1 - dot product.
    return 1.0 - sum(x * y for x, y in zip(a, b))


def search(query: str, records: list[dict], k: int = 5) -> list[dict]:
    """Embed the query, return top-k records by cosine distance."""
    [query_vec] = embed([query])
    scored = [(cosine_distance(r["embedding"], query_vec), r) for r in records]
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]


def main() -> None:
    print(f"Indexing documents from {DOCS_DIR}/")
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")


if __name__ == "__main__":
    main()
