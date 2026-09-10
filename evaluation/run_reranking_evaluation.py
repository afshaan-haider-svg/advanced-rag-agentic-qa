"""Chunk-level before vs after Cross-Encoder reranking evaluation.

This evaluation does not call the LLM.

For every question:
1. Retrieve 8 candidate chunks.
2. Use the expected answer as evaluation ground truth.
3. Measure lexical overlap between each candidate chunk and expected answer.
4. Identify the most answer-relevant candidate chunk.
5. Compare that same chunk's rank before and after Cross-Encoder reranking.

Metrics:
- Best relevant chunk average rank
- MRR before vs after
- Relevant chunk Recall@4 before vs after
- Improved / unchanged / worsened queries
- Retrieval/reranking latency
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "eval_dataset.json"
RESULTS_JSON_PATH = BASE_DIR / "reranking_chunk_results.json"
RESULTS_CSV_PATH = BASE_DIR / "reranking_chunk_results.csv"
SUMMARY_PATH = BASE_DIR / "reranking_chunk_summary.json"

API_URL = "http://localhost:8000/rag/search"

TOP_K = 8

# Important:
# We rerank ALL retrieved candidates so before/after ranking
# is compared over exactly the same candidate set.
RERANK_TOP_N = 8

SIMILARITY_THRESHOLD = 0.30
MMR_LAMBDA = 0.7

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on",
    "for", "is", "are", "was", "were", "be", "been", "being",
    "with", "that", "this", "these", "those", "it", "its",
    "as", "at", "by", "from", "can", "could", "may", "might",
    "will", "would", "should", "has", "have", "had", "how",
    "what", "why", "when", "where", "which", "who", "about",
    "into", "than", "their", "them", "they", "also",
}


def tokenize(text: str) -> list[str]:
    """Normalize text into meaningful lexical tokens."""
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())

    return [
        word
        for word in words
        if word not in STOPWORDS and len(word) > 1
    ]


def lexical_f1(reference: str, candidate: str) -> float:
    """Calculate token-level lexical F1 overlap."""
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)

    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_set = set(reference_tokens)
    candidate_set = set(candidate_tokens)

    common = reference_set.intersection(candidate_set)

    if not common:
        return 0.0

    precision = len(common) / len(candidate_set)
    recall = len(common) / len(reference_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def post_rag_search(question: str) -> dict[str, Any]:
    """Call local retrieval/reranking endpoint."""
    payload = {
        "query": question,
        "top_k": TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "rerank_top_n": RERANK_TOP_N,
        "use_mmr": True,
        "mmr_lambda": MMR_LAMBDA,
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=240,
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def get_chunk_key(chunk: dict[str, Any]) -> str:
    """Create stable chunk identifier."""
    chunk_id = str(chunk.get("chunk_id", "")).strip()

    if chunk_id:
        return chunk_id

    return (
        f"{chunk.get('document_id', '')}:"
        f"{chunk.get('chunk_index', '')}"
    )


def find_best_relevant_chunk(
    candidates: list[dict[str, Any]],
    expected_answer: str,
    expected_document: str,
) -> tuple[dict[str, Any] | None, float]:
    """Find candidate chunk with strongest expected-answer overlap.

    Preference is given to chunks from the expected source document.
    """

    expected_document_lower = expected_document.lower().strip()

    source_candidates = [
        chunk
        for chunk in candidates
        if expected_document_lower
        in str(chunk.get("filename", "")).lower()
    ]

    pool = source_candidates if source_candidates else candidates

    if not pool:
        return None, 0.0

    scored = []

    for chunk in pool:
        score = lexical_f1(
            expected_answer,
            str(chunk.get("content", "")),
        )

        scored.append(
            (score, chunk)
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return scored[0][1], scored[0][0]


def find_rank(
    chunks: list[dict[str, Any]],
    target_chunk: dict[str, Any],
) -> int | None:
    """Find rank of same chunk inside another ranking."""
    target_key = get_chunk_key(target_chunk)

    for rank, chunk in enumerate(chunks, start=1):
        if get_chunk_key(chunk) == target_key:
            return rank

    return None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0

    return 1.0 / rank


def average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def main() -> None:
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    print()
    print("=" * 76)
    print("Advanced RAG - Chunk-Level Reranking Evaluation")
    print("=" * 76)
    print(f"Questions            : {len(dataset)}")
    print(f"Retrieved candidates : {TOP_K}")
    print(f"Candidates reranked  : {RERANK_TOP_N}")
    print(f"Similarity threshold : {SIMILARITY_THRESHOLD}")
    print(f"MMR lambda           : {MMR_LAMBDA}")
    print("=" * 76)
    print()

    results: list[dict[str, Any]] = []

    before_rr_values = []
    after_rr_values = []

    before_ranks = []
    after_ranks = []

    before_recall4 = 0
    after_recall4 = 0

    improved = 0
    unchanged = 0
    worsened = 0

    successful = 0
    failed = 0

    total_latencies = []
    retrieval_latencies = []
    reranking_latencies = []

    for item in dataset:
        question_id = item["id"]
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_document = item["expected_document"]

        print(f"Q{question_id}: {question}")

        start = time.perf_counter()

        try:
            response = post_rag_search(question)

            wall_latency = time.perf_counter() - start

            before = response.get(
                "retrieved_candidates",
                [],
            )

            after = response.get(
                "reranked_documents",
                [],
            )

            stats = response.get(
                "execution_stats",
                {},
            )

            best_chunk, overlap_score = (
                find_best_relevant_chunk(
                    before,
                    expected_answer,
                    expected_document,
                )
            )

            if best_chunk is None:
                raise RuntimeError(
                    "No candidate chunk available."
                )

            before_rank = find_rank(
                before,
                best_chunk,
            )

            after_rank = find_rank(
                after,
                best_chunk,
            )

            before_rr = reciprocal_rank(
                before_rank
            )

            after_rr = reciprocal_rank(
                after_rank
            )

            before_rr_values.append(before_rr)
            after_rr_values.append(after_rr)

            if before_rank is not None:
                before_ranks.append(
                    float(before_rank)
                )

            if after_rank is not None:
                after_ranks.append(
                    float(after_rank)
                )

            before_top4 = (
                before_rank is not None
                and before_rank <= 4
            )

            after_top4 = (
                after_rank is not None
                and after_rank <= 4
            )

            if before_top4:
                before_recall4 += 1

            if after_top4:
                after_recall4 += 1

            if (
                before_rank is not None
                and after_rank is not None
            ):
                if after_rank < before_rank:
                    status = "improved"
                    improved += 1

                elif after_rank > before_rank:
                    status = "worsened"
                    worsened += 1

                else:
                    status = "unchanged"
                    unchanged += 1

            else:
                status = "missing"
                worsened += 1

            retrieval_ms = float(
                stats.get(
                    "retrieval_ms",
                    0.0,
                )
            )

            reranking_ms = float(
                stats.get(
                    "reranking_ms",
                    0.0,
                )
            )

            total_ms = float(
                stats.get(
                    "total_latency_ms",
                    wall_latency * 1000,
                )
            )

            retrieval_latencies.append(
                retrieval_ms
            )

            reranking_latencies.append(
                reranking_ms
            )

            total_latencies.append(
                total_ms
            )

            result = {
                "id": question_id,
                "question": question,
                "expected_document": expected_document,
                "category": item.get(
                    "category",
                    "",
                ),
                "status": "success",
                "relevant_chunk_id": get_chunk_key(
                    best_chunk
                ),
                "relevant_filename": best_chunk.get(
                    "filename",
                    "",
                ),
                "relevant_chunk_index": best_chunk.get(
                    "chunk_index",
                    "",
                ),
                "answer_overlap_f1": round(
                    overlap_score,
                    4,
                ),
                "before_rank": before_rank,
                "after_rank": after_rank,
                "before_rr": round(
                    before_rr,
                    4,
                ),
                "after_rr": round(
                    after_rr,
                    4,
                ),
                "before_top4": before_top4,
                "after_top4": after_top4,
                "rank_change": (
                    before_rank - after_rank
                    if before_rank is not None
                    and after_rank is not None
                    else None
                ),
                "rank_status": status,
                "retrieval_ms": round(
                    retrieval_ms,
                    2,
                ),
                "reranking_ms": round(
                    reranking_ms,
                    2,
                ),
                "total_latency_ms": round(
                    total_ms,
                    2,
                ),
                "error": "",
            }

            successful += 1

            print(
                f"   Relevant chunk: "
                f"{best_chunk.get('chunk_index')}"
            )

            print(
                f"   Answer overlap: "
                f"{overlap_score:.4f}"
            )

            print(
                f"   Rank: "
                f"{before_rank} -> {after_rank} "
                f"| {status.upper()}"
            )

            print(
                f"   Retrieval: {retrieval_ms:.2f} ms | "
                f"Reranking: {reranking_ms:.2f} ms"
            )

        except urllib.error.HTTPError as exc:
            failed += 1

            try:
                error_text = (
                    exc.read()
                    .decode("utf-8")
                )
            except Exception:
                error_text = str(exc)

            result = {
                "id": question_id,
                "question": question,
                "expected_document": expected_document,
                "category": item.get(
                    "category",
                    "",
                ),
                "status": "failed",
                "error": (
                    f"HTTP {exc.code}: "
                    f"{error_text}"
                ),
            }

            print(
                f"   FAILED: HTTP {exc.code}"
            )

        except Exception as exc:
            failed += 1

            result = {
                "id": question_id,
                "question": question,
                "expected_document": expected_document,
                "category": item.get(
                    "category",
                    "",
                ),
                "status": "failed",
                "error": str(exc),
            }

            print(
                f"   FAILED: {exc}"
            )

        results.append(result)

        print()

    before_mrr = average(
        before_rr_values
    )

    after_mrr = average(
        after_rr_values
    )

    before_avg_rank = average(
        before_ranks
    )

    after_avg_rank = average(
        after_ranks
    )

    before_recall4_rate = (
        before_recall4 / successful
        if successful
        else 0.0
    )

    after_recall4_rate = (
        after_recall4 / successful
        if successful
        else 0.0
    )

    summary = {
        "evaluation_type": (
            "chunk_level_expected_answer_overlap"
        ),
        "total_questions": len(dataset),
        "successful_requests": successful,
        "failed_requests": failed,
        "configuration": {
            "top_k": TOP_K,
            "rerank_top_n": RERANK_TOP_N,
            "similarity_threshold": (
                SIMILARITY_THRESHOLD
            ),
            "use_mmr": True,
            "mmr_lambda": MMR_LAMBDA,
        },
        "before_reranking": {
            "mrr": round(
                before_mrr,
                4,
            ),
            "average_relevant_chunk_rank": round(
                before_avg_rank,
                4,
            ),
            "relevant_chunk_recall_at_4": round(
                before_recall4_rate,
                4,
            ),
        },
        "after_reranking": {
            "mrr": round(
                after_mrr,
                4,
            ),
            "average_relevant_chunk_rank": round(
                after_avg_rank,
                4,
            ),
            "relevant_chunk_recall_at_4": round(
                after_recall4_rate,
                4,
            ),
        },
        "comparison": {
            "mrr_change": round(
                after_mrr - before_mrr,
                4,
            ),
            "average_rank_change": round(
                before_avg_rank - after_avg_rank,
                4,
            ),
            "recall_at_4_change": round(
                after_recall4_rate
                - before_recall4_rate,
                4,
            ),
            "queries_improved": improved,
            "queries_unchanged": unchanged,
            "queries_worsened": worsened,
        },
        "latency": {
            "average_retrieval_ms": round(
                average(
                    retrieval_latencies
                ),
                2,
            ),
            "average_reranking_ms": round(
                average(
                    reranking_latencies
                ),
                2,
            ),
            "average_total_ms": round(
                average(
                    total_latencies
                ),
                2,
            ),
        },
    }

    with RESULTS_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    csv_fields = [
        "id",
        "question",
        "expected_document",
        "category",
        "status",
        "relevant_chunk_id",
        "relevant_filename",
        "relevant_chunk_index",
        "answer_overlap_f1",
        "before_rank",
        "after_rank",
        "before_rr",
        "after_rr",
        "before_top4",
        "after_top4",
        "rank_change",
        "rank_status",
        "retrieval_ms",
        "reranking_ms",
        "total_latency_ms",
        "error",
    ]

    with RESULTS_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 76)
    print("FINAL CHUNK-LEVEL RERANKING COMPARISON")
    print("=" * 76)

    print(
        f"Successful Requests      : "
        f"{successful}/{len(dataset)}"
    )

    print(
        f"Failed Requests          : "
        f"{failed}/{len(dataset)}"
    )

    print()

    print(
        f"Before MRR               : "
        f"{before_mrr:.4f}"
    )

    print(
        f"After MRR                : "
        f"{after_mrr:.4f}"
    )

    print(
        f"MRR Change               : "
        f"{after_mrr - before_mrr:+.4f}"
    )

    print()

    print(
        f"Before Avg Relevant Rank : "
        f"{before_avg_rank:.2f}"
    )

    print(
        f"After Avg Relevant Rank  : "
        f"{after_avg_rank:.2f}"
    )

    print(
        f"Average Rank Improvement : "
        f"{before_avg_rank - after_avg_rank:+.2f}"
    )

    print()

    print(
        f"Before Recall@4          : "
        f"{before_recall4_rate:.2%}"
    )

    print(
        f"After Recall@4           : "
        f"{after_recall4_rate:.2%}"
    )

    print()

    print(
        f"Queries Improved         : {improved}"
    )

    print(
        f"Queries Unchanged        : {unchanged}"
    )

    print(
        f"Queries Worsened         : {worsened}"
    )

    print()

    print(
        f"Avg Retrieval Latency    : "
        f"{average(retrieval_latencies):.2f} ms"
    )

    print(
        f"Avg Reranking Latency    : "
        f"{average(reranking_latencies):.2f} ms"
    )

    print(
        f"Avg Total Latency        : "
        f"{average(total_latencies):.2f} ms"
    )

    print("=" * 76)

    print()
    print("Results saved:")
    print(f"  {RESULTS_JSON_PATH}")
    print(f"  {RESULTS_CSV_PATH}")
    print(f"  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()