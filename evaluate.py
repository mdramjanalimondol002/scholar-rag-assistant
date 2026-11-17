"""Compare TF-IDF and dense-embedding retrieval on a set of questions.

Usage:
    python evaluate.py --pdf paper.pdf --questions eval/questions.json
    python evaluate.py --sample            # tiny smoke test on the built-in sample text

questions.json is a list of items like:
    {"question": "How many original images were collected?", "answer": "4,267"}

"answer" is a short piece of text copied exactly from the paper. A retrieved chunk
counts as correct if it contains that text. Metrics, per retrieval method:
    Hit@1 / Hit@3 - share of questions with a correct chunk in the top 1 / top 3
    MRR           - mean of 1/rank of the first correct chunk (0 if not in the top 10)
"""
import argparse
import json
import re
from pathlib import Path

from app.rag_engine import METHODS, ScholarRAGEngine
from app.sample import SAMPLE_PAPER

RANK_DEPTH = 10
LABELS = {"tfidf": "TF-IDF (keyword baseline)", "dense": "Dense embeddings (MiniLM + FAISS)"}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def first_correct_rank(results, answer):
    """1-based rank of the first retrieved chunk containing the answer, or None."""
    target = normalise(answer)
    for rank, item in enumerate(results, start=1):
        if target in normalise(item["content"]):
            return rank
    return None


def evaluate(engine, questions):
    scores = {}
    per_question = []
    for method in METHODS:
        ranks = []
        for q in questions:
            results = engine.retrieve(q["question"], top_k=RANK_DEPTH, method=method, min_score=-1)
            ranks.append(first_correct_rank(results, q["answer"]))
        n = len(ranks)
        scores[method] = {
            "hit@1": sum(r is not None and r <= 1 for r in ranks) / n,
            "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r is not None) / n,
        }
        per_question.append(ranks)
    return scores, list(zip(*per_question))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", help="Path to the paper to index")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--sample", action="store_true", help="Smoke test on the built-in sample text")
    parser.add_argument("--out", default="eval/results.md")
    args = parser.parse_args()

    engine = ScholarRAGEngine()
    if args.sample:
        source = "built-in sample text"
        n_chunks = engine.process_text(SAMPLE_PAPER, chunk_size=45)
        questions_path = Path("eval/questions.example.json")
    elif args.pdf:
        source = Path(args.pdf).name
        n_chunks = engine.load_pdf(args.pdf)
        questions_path = Path(args.questions)
    else:
        parser.error("give --pdf PATH or --sample")

    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    # An answer that appears in no chunk can never be retrieved; that is a mistake in the
    # questions file (typo, or text split across lines in the PDF), not a retrieval failure.
    all_text = [normalise(c) for c in engine.chunks]
    usable, unusable = [], []
    for q in questions:
        (usable if any(normalise(q["answer"]) in c for c in all_text) else unusable).append(q)
    if unusable:
        print(f"WARNING: {len(unusable)} question(s) skipped - answer text not found in the document:")
        for q in unusable:
            print(f"  - {q['question']!r} (answer: {q['answer']!r})")
    if not usable:
        raise SystemExit("No usable questions. Fix the answers in the questions file.")

    scores, per_question = evaluate(engine, usable)

    lines = [
        f"Document: {source} ({n_chunks} chunks). Questions: {len(usable)}.",
        "",
        "| Retrieval method | Hit@1 | Hit@3 | MRR |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for method in METHODS[::-1]:
        s = scores[method]
        lines.append(f"| {LABELS[method]} | {s['hit@1']:.0%} | {s['hit@3']:.0%} | {s['mrr']:.2f} |")
    table = "\n".join(lines)
    print("\n" + table + "\n")

    print("Rank of the first correct chunk per question (- = not in top 10):")
    print("  " + " / ".join(METHODS))
    for q, ranks in zip(usable, per_question):
        shown = " / ".join(str(r) if r else "-" for r in ranks)
        print(f"  {shown:>9}   {q['question']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved table to {args.out}")


if __name__ == "__main__":
    main()
