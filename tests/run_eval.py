"""Measures retrieval quality with the REAL embedder.   python -m tests.run_eval
Prints each question's similarity so you can tune CONFIDENCE_THRESHOLD in .env."""
from backend.config import settings
from backend.rag import get_retriever
from tests.eval_questions import QUESTIONS


def topic(source: str) -> str:
    return source.rsplit("_", 1)[0]  # thread_breaking_te.md -> thread_breaking


def main() -> None:
    retriever = get_retriever()
    if not retriever.ids:
        raise SystemExit("Nothing indexed. Run:  python -m backend.ingest")
    passed = 0
    for question, expected in QUESTIONS:
        hits = retriever.search(question)
        confidence = max((h.similarity for h in hits), default=0.0)
        answered = confidence >= settings.confidence_threshold
        if expected is None:
            ok = not answered
        else:
            ok = answered and topic(hits[0].meta["source"]) == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  sim={confidence:.2f}  {question[:60]}")
    print(f"\n{passed}/{len(QUESTIONS)} passed (threshold {settings.confidence_threshold}, "
          f"backend {settings.embed_backend})")
    print("If unrelated questions FAIL, raise the threshold. If real questions FAIL, lower it.")


if __name__ == "__main__":
    main()
