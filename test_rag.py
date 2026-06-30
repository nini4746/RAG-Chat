"""Quick harness: exercises the same retrieval + answer + citation path as
backend/app.py, without running Flask. Produces report material."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from anthropic import Anthropic
from dotenv import load_dotenv

from indexer import load_index, search
from backend.app import SYSTEM_PROMPT, _build_citations

load_dotenv()
client = Anthropic()
INDEX = load_index()

QUESTIONS = [
    "What was the cause of the Apollo 1 fire?",
    "Which Apollo missions landed on the Moon?",
    "Compare the moonwalk durations of Apollo 11 and Apollo 17.",
    "List Apollo missions that used the Saturn V rocket.",
    "What is the Artemis program?",
]


def ask(q: str):
    hits = search(q, INDEX, k=5)
    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    answer = resp.content[0].text
    citations = _build_citations(answer, hits)
    return answer, citations, hits


for q in QUESTIONS:
    print("=" * 70)
    print("Q:", q)
    answer, citations, hits = ask(q)
    print("\nA:", answer)
    print("\nRetrieved sources (top-5):", [h["source"] for h in hits])
    print("Cited:", [(c["n"], c["source"]) for c in citations])
    print()
