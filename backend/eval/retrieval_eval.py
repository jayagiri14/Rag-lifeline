"""Retrieval quality evaluation for the medical RAG embedding model.

Measures whether embedding the *question* half of a Q&A pair retrieves the
correct *answer* out of a batch of distractor answers drawn from the same
medical corpus (`app/medical_data.py`). Reports Recall@1/3/5 and MRR for the
PubMedBERT model currently used in the app, compared against a lightweight
general-purpose baseline (all-MiniLM-L6-v2) referenced in `app/config.py` but
never actually benchmarked.

Usage:
    cd backend
    python -m eval.retrieval_eval --n 200 --seed 42

Known limitation: this is a closed-world benchmark (correct answer is always
present among the sampled distractors) using the corpus's own Q&A pairs, not
independently authored paraphrased queries. It measures whether the embedding
space clusters matching Q/A pairs correctly relative to same-domain distractors
-- a reasonable proxy for retrieval quality, not a full IR benchmark.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.medical_data import get_medical_documents  # noqa: E402


def _cosine_sim_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-9)
    c = corpus / np.linalg.norm(corpus, axis=1, keepdims=True).clip(min=1e-9)
    return q @ c.T


def _metrics(sim: np.ndarray, k_values: list[int]) -> dict:
    """sim[i][j] = similarity of query i to corpus item j; correct match is j == i."""
    n = sim.shape[0]
    ranks = (-sim).argsort(axis=1)  # descending similarity -> index order
    correct_rank = np.empty(n, dtype=int)
    for i in range(n):
        correct_rank[i] = int(np.where(ranks[i] == i)[0][0]) + 1  # 1-indexed

    result = {"mrr": float(np.mean(1.0 / correct_rank))}
    for k in k_values:
        result[f"recall@{k}"] = float(np.mean(correct_rank <= k))
    return result


def _sample_pairs(n: int, seed: int) -> tuple[list[str], list[str]]:
    docs = get_medical_documents(limit=None)
    rng = random.Random(seed)
    sample = rng.sample(docs, min(n, len(docs)))
    questions = [d["metadata"]["question"] for d in sample]
    answers = [d["content"].split("\n\nA: ", 1)[1] for d in sample]
    return questions, answers


def run_pubmedbert(questions: list[str], answers: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    from app.embeddings import get_embeddings

    start = time.time()
    q_emb = np.array(get_embeddings(questions))
    a_emb = np.array(get_embeddings(answers))
    return q_emb, a_emb, time.time() - start


def run_minilm(questions: list[str], answers: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    start = time.time()
    q_emb = model.encode(questions, show_progress_bar=False)
    a_emb = model.encode(answers, show_progress_bar=False)
    return np.array(q_emb), np.array(a_emb), time.time() - start


def run_pubmedbert_msmarco(questions: list[str], answers: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    """PubMedBERT fine-tuned on MS MARCO for retrieval -- same domain vocabulary as
    the app's current model, but with an actual similarity/retrieval training
    objective (unlike raw PubMedBERT, which was only trained for masked-LM)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    start = time.time()
    q_emb = model.encode(questions, show_progress_bar=False)
    a_emb = model.encode(answers, show_progress_bar=False)
    return np.array(q_emb), np.array(a_emb), time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Number of Q&A pairs to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=str, default="1,3,5", help="Comma-separated k values for Recall@k")
    parser.add_argument("--output", type=str, default="eval/results.json")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k.split(",")]
    questions, answers = _sample_pairs(args.n, args.seed)
    print(f"Sampled {len(questions)} Q&A pairs (seed={args.seed}) from the medical corpus.\n")

    report = {"n": len(questions), "seed": args.seed, "models": {}}

    runners = [
        ("pubmedbert", run_pubmedbert),
        ("minilm", run_minilm),
        ("pubmedbert-msmarco", run_pubmedbert_msmarco),
    ]
    for name, runner in runners:
        print(f"Embedding with {name}...")
        try:
            q_emb, a_emb, elapsed = runner(questions, answers)
        except Exception as exc:
            print(f"  skipped ({exc})")
            continue
        sim = _cosine_sim_matrix(q_emb, a_emb)
        metrics = _metrics(sim, k_values)
        metrics["embedding_seconds"] = round(elapsed, 2)
        report["models"][name] = metrics
        print(f"  {metrics}\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote results to {args.output}")

    print("\n| model | " + " | ".join(f"recall@{k}" for k in k_values) + " | mrr | embed time (s) |")
    print("|---" * (len(k_values) + 3) + "|")
    for name, m in report["models"].items():
        cells = [f"{m[f'recall@{k}']:.3f}" for k in k_values]
        print(f"| {name} | " + " | ".join(cells) + f" | {m['mrr']:.3f} | {m['embedding_seconds']} |")


if __name__ == "__main__":
    main()
