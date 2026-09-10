import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset.json"

RESULTS_JSON_PATH = BASE_DIR / "evaluation_results.json"
RESULTS_CSV_PATH = BASE_DIR / "evaluation_results.csv"
SUMMARY_PATH = BASE_DIR / "evaluation_summary.json"

API_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_URL}/chat"

REQUEST_TIMEOUT_SECONDS = 240

SESSION_PREFIX = "evaluation-session"


# ---------------------------------------------------------
# Text utilities
# ---------------------------------------------------------

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "by",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "from",
    "at",
    "into",
    "can",
    "may",
    "such",
    "than",
    "then",
    "their",
    "they",
    "them",
    "which",
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def meaningful_tokens(text: str) -> set[str]:
    tokens = normalize_text(text).split()

    return {
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    }


def lexical_scores(
    generated_answer: str,
    expected_answer: str,
) -> dict[str, float]:
    generated_tokens = meaningful_tokens(generated_answer)
    expected_tokens = meaningful_tokens(expected_answer)

    if not generated_tokens or not expected_tokens:
        return {
            "answer_precision": 0.0,
            "answer_recall": 0.0,
            "answer_f1": 0.0,
        }

    overlap = generated_tokens.intersection(expected_tokens)

    precision = len(overlap) / len(generated_tokens)
    recall = len(overlap) / len(expected_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "answer_precision": round(precision, 4),
        "answer_recall": round(recall, 4),
        "answer_f1": round(f1, 4),
    }


# ---------------------------------------------------------
# API helpers
# ---------------------------------------------------------

def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    encoded_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=encoded_data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")

        raise RuntimeError(
            f"HTTP {exc.code}: {body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to API: {exc}"
        ) from exc


# ---------------------------------------------------------
# Response extraction
# ---------------------------------------------------------

def extract_answer(response: dict[str, Any]) -> str:
    possible_fields = [
        "verified_answer",
        "answer",
        "response",
        "message",
        "content",
    ]

    for field in possible_fields:
        value = response.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    data = response.get("data")

    if isinstance(data, dict):
        for field in possible_fields:
            value = data.get(field)

            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def extract_citations(response: dict[str, Any]) -> list[dict[str, Any]]:
    citations = response.get("citations")

    if isinstance(citations, list):
        return [
            citation
            for citation in citations
            if isinstance(citation, dict)
        ]

    data = response.get("data")

    if isinstance(data, dict):
        citations = data.get("citations")

        if isinstance(citations, list):
            return [
                citation
                for citation in citations
                if isinstance(citation, dict)
            ]

    return []


def citation_source_text(citation: dict[str, Any]) -> str:
    possible_fields = [
        "filename",
        "source",
        "document_name",
        "document",
        "file_name",
        "title",
    ]

    parts = []

    for field in possible_fields:
        value = citation.get(field)

        if value is not None:
            parts.append(str(value))

    return " ".join(parts)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_source_hit(
    citations: list[dict[str, Any]],
    expected_document: str,
) -> bool:
    expected = normalize_text(expected_document)

    expected_without_extension = re.sub(
        r"\b(docx|pdf|txt)\b",
        "",
        expected,
    ).strip()

    for citation in citations:
        source_text = normalize_text(
            citation_source_text(citation)
        )

        if expected in source_text:
            return True

        if (
            expected_without_extension
            and expected_without_extension in source_text
        ):
            return True

    return False


def calculate_citation_score(
    answer: str,
    citations: list[dict[str, Any]],
) -> float:
    """
    Lightweight RAG faithfulness proxy.

    This is NOT full RAGAS faithfulness.
    It checks whether the generated answer is accompanied
    by retrieved/cited evidence.
    """

    if not answer.strip():
        return 0.0

    if not citations:
        return 0.0

    inline_citations = re.findall(r"\[\d+\]", answer)

    if inline_citations:
        return 1.0

    return 0.5


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_question(
    item: dict[str, Any],
) -> dict[str, Any]:

    question_id = item["id"]
    question = item["question"]
    expected_answer = item["expected_answer"]
    expected_document = item["expected_document"]
    category = item.get("category", "")

    session_id = f"{SESSION_PREFIX}-{question_id}"

    payload = {
        "message": question,
        "session_id": session_id,
    }

    start_time = time.perf_counter()

    try:
        response = post_json(
            CHAT_ENDPOINT,
            payload,
        )

        latency_seconds = time.perf_counter() - start_time

        answer = extract_answer(response)
        citations = extract_citations(response)

        similarity = lexical_scores(
            answer,
            expected_answer,
        )

        source_hit = calculate_source_hit(
            citations,
            expected_document,
        )

        faithfulness_proxy = calculate_citation_score(
            answer,
            citations,
        )

        citation_sources = [
            citation_source_text(citation)
            for citation in citations
        ]

        return {
            "id": question_id,
            "category": category,
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "expected_document": expected_document,
            "source_hit": source_hit,
            "citation_count": len(citations),
            "citation_sources": citation_sources,
            "answer_precision": similarity["answer_precision"],
            "answer_recall": similarity["answer_recall"],
            "answer_f1": similarity["answer_f1"],
            "faithfulness_proxy": faithfulness_proxy,
            "latency_seconds": round(latency_seconds, 3),
            "success": True,
            "error": None,
        }

    except Exception as exc:
        latency_seconds = time.perf_counter() - start_time

        return {
            "id": question_id,
            "category": category,
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": "",
            "expected_document": expected_document,
            "source_hit": False,
            "citation_count": 0,
            "citation_sources": [],
            "answer_precision": 0.0,
            "answer_recall": 0.0,
            "answer_f1": 0.0,
            "faithfulness_proxy": 0.0,
            "latency_seconds": round(latency_seconds, 3),
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def average(values: list[float]) -> float:
    if not values:
        return 0.0

    return round(sum(values) / len(values), 4)


def build_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    successful = [
        result
        for result in results
        if result["success"]
    ]

    source_hits = sum(
        1
        for result in successful
        if result["source_hit"]
    )

    total_successful = len(successful)

    retrieval_hit_rate = (
        source_hits / total_successful
        if total_successful
        else 0.0
    )

    return {
        "total_questions": len(results),
        "successful_requests": total_successful,
        "failed_requests": len(results) - total_successful,
        "retrieval_source_hits": source_hits,
        "retrieval_hit_rate": round(
            retrieval_hit_rate,
            4,
        ),
        "average_answer_precision": average(
            [
                result["answer_precision"]
                for result in successful
            ]
        ),
        "average_answer_recall": average(
            [
                result["answer_recall"]
                for result in successful
            ]
        ),
        "average_answer_f1": average(
            [
                result["answer_f1"]
                for result in successful
            ]
        ),
        "average_faithfulness_proxy": average(
            [
                result["faithfulness_proxy"]
                for result in successful
            ]
        ),
        "average_latency_seconds": average(
            [
                result["latency_seconds"]
                for result in successful
            ]
        ),
        "minimum_latency_seconds": round(
            min(
                (
                    result["latency_seconds"]
                    for result in successful
                ),
                default=0.0,
            ),
            3,
        ),
        "maximum_latency_seconds": round(
            max(
                (
                    result["latency_seconds"]
                    for result in successful
                ),
                default=0.0,
            ),
            3,
        ),
    }


# ---------------------------------------------------------
# File output
# ---------------------------------------------------------

def save_json(
    path: Path,
    data: Any,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_csv(
    results: list[dict[str, Any]],
) -> None:

    fields = [
        "id",
        "category",
        "question",
        "expected_document",
        "source_hit",
        "citation_count",
        "answer_precision",
        "answer_recall",
        "answer_f1",
        "faithfulness_proxy",
        "latency_seconds",
        "success",
        "error",
    ]

    with RESULTS_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in results:
            row = {
                field: result.get(field)
                for field in fields
            }

            writer.writerow(row)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    print("=" * 70)
    print("Advanced RAG & Agentic QA - Evaluation")
    print("=" * 70)
    print(f"Questions: {len(dataset)}")
    print(f"API: {CHAT_ENDPOINT}")
    print()

    results = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"[{index}/{len(dataset)}] "
            f"Q{item['id']}: {item['question']}"
        )

        result = evaluate_question(item)
        results.append(result)

        if result["success"]:
            print(
                "    SUCCESS | "
                f"Source Hit: {result['source_hit']} | "
                f"F1: {result['answer_f1']:.4f} | "
                f"Citations: {result['citation_count']} | "
                f"Latency: {result['latency_seconds']:.2f}s"
            )
        else:
            print(
                f"    FAILED | {result['error']}"
            )

        print()

    summary = build_summary(results)

    save_json(
        RESULTS_JSON_PATH,
        results,
    )

    save_json(
        SUMMARY_PATH,
        summary,
    )

    save_csv(results)

    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Successful Requests : "
        f"{summary['successful_requests']}/"
        f"{summary['total_questions']}"
    )

    print(
        f"Retrieval Hit Rate  : "
        f"{summary['retrieval_hit_rate'] * 100:.2f}%"
    )

    print(
        f"Average Answer F1   : "
        f"{summary['average_answer_f1']:.4f}"
    )

    print(
        f"Faithfulness Proxy  : "
        f"{summary['average_faithfulness_proxy']:.4f}"
    )

    print(
        f"Average Latency     : "
        f"{summary['average_latency_seconds']:.2f}s"
    )

    print(
        f"Min / Max Latency   : "
        f"{summary['minimum_latency_seconds']:.2f}s / "
        f"{summary['maximum_latency_seconds']:.2f}s"
    )

    print()
    print("Results saved to:")
    print(f"  {RESULTS_JSON_PATH}")
    print(f"  {RESULTS_CSV_PATH}")
    print(f"  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()