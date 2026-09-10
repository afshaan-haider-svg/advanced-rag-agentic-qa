"""Experimental comparison of chunk size and overlap configurations.

This script evaluates multiple RecursiveCharacterTextSplitter configurations
using the same local embedding model as the production RAG pipeline.

No Gemini / LLM API calls are made.

Configurations:
- 300 / 50
- 500 / 100
- 800 / 150

Metrics:
- MRR
- Recall@4
- Average relevant chunk rank
- Average query latency
- Total generated chunks
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from backend.app.core.config import settings
    from backend.app.rag.loaders import load_document
    from backend.app.rag.cleaning import clean_documents
    from backend.app.rag.chunking import chunk_documents
    from backend.app.rag.embeddings import get_embeddings_client
except ImportError as exc:
    raise RuntimeError(
        "Could not import backend modules. "
        "Run this script from the project root."
    ) from exc


# ---------------------------------------------------------------------
# Paths / Configuration
# ---------------------------------------------------------------------

EVALUATION_DIR = Path(__file__).resolve().parent

DATASET_PATH = EVALUATION_DIR / "eval_dataset.json"

RESULTS_PATH = (
    EVALUATION_DIR / "chunking_experiment_results.json"
)

SUMMARY_PATH = (
    EVALUATION_DIR / "chunking_experiment_summary.json"
)


CONFIGURATIONS = [
    {
        "name": "small",
        "chunk_size": 300,
        "chunk_overlap": 50,
    },
    {
        "name": "current",
        "chunk_size": 500,
        "chunk_overlap": 100,
    },
    {
        "name": "large",
        "chunk_size": 800,
        "chunk_overlap": 150,
    },
]


TOP_K = 8
RECALL_K = 4


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on",
    "for", "is", "are", "was", "were", "be", "been", "being",
    "with", "that", "this", "these", "those", "it", "its",
    "as", "at", "by", "from", "can", "could", "may", "might",
    "will", "would", "should", "has", "have", "had", "how",
    "what", "why", "when", "where", "which", "who", "about",
    "into", "than", "their", "them", "they", "also",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    words = re.findall(
        r"[a-zA-Z0-9]+",
        str(text).lower(),
    )

    return [
        word
        for word in words
        if word not in STOPWORDS and len(word) > 1
    ]


def lexical_f1(
    reference: str,
    candidate: str,
) -> float:
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)

    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_set = set(reference_tokens)
    candidate_set = set(candidate_tokens)

    common = reference_set.intersection(
        candidate_set
    )

    if not common:
        return 0.0

    precision = (
        len(common) / len(candidate_set)
    )

    recall = (
        len(common) / len(reference_set)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def reciprocal_rank(
    rank: int | None,
) -> float:
    if rank is None or rank <= 0:
        return 0.0

    return 1.0 / rank


def average(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def normalize_filename(
    value: str,
) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("\\", "/")
        .split("/")[-1]
    )


def filename_matches(
    actual: str,
    expected: str,
) -> bool:
    actual_name = normalize_filename(actual)
    expected_name = normalize_filename(expected)

    return (
        actual_name == expected_name
        or expected_name in actual_name
        or actual_name in expected_name
    )


# ---------------------------------------------------------------------
# Registry / document loading
# ---------------------------------------------------------------------

def load_registry_records() -> list[dict[str, Any]]:
    registry_path = Path(
        settings.REGISTRY_FILE_PATH
    )

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Document registry not found: "
            f"{registry_path}"
        )

    with registry_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    records = []

    for value in data.values():
        if value.get("status") == "indexed":
            records.append(value)

    if not records:
        raise RuntimeError(
            "No indexed documents were found "
            "in document registry."
        )

    return records


def find_uploaded_file(
    document_id: str,
    filename: str,
) -> Path:
    upload_dir = Path(
        settings.UPLOAD_DIR
    )

    expected_path = (
        upload_dir
        / f"{document_id}_{filename}"
    )

    if expected_path.exists():
        return expected_path

    matches = list(
        upload_dir.glob(
            f"{document_id}_*"
        )
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Uploaded source file missing for "
        f"{filename} ({document_id})"
    )


def load_clean_source_documents():
    registry_records = (
        load_registry_records()
    )

    loaded_sources = []

    print("Loading source documents...")

    for record in registry_records:
        document_id = str(
            record["document_id"]
        )

        filename = str(
            record["filename"]
        )

        file_path = find_uploaded_file(
            document_id,
            filename,
        )

        raw_documents = load_document(
            file_path,
            document_id,
            filename,
        )

        cleaned_documents = (
            clean_documents(
                raw_documents
            )
        )

        loaded_sources.extend(
            cleaned_documents
        )

        print(
            f"  Loaded: {filename} "
            f"({len(cleaned_documents)} pages/sections)"
        )

    print()

    return loaded_sources


# ---------------------------------------------------------------------
# Ground-truth relevance
# ---------------------------------------------------------------------

def get_best_relevant_chunk(
    chunks,
    expected_document: str,
    expected_answer: str,
):
    expected_source_chunks = [
        chunk
        for chunk in chunks
        if filename_matches(
            chunk.metadata.get(
                "filename",
                "",
            ),
            expected_document,
        )
    ]

    if not expected_source_chunks:
        return None, 0.0

    scored = []

    for chunk in expected_source_chunks:
        score = lexical_f1(
            expected_answer,
            chunk.page_content,
        )

        scored.append(
            (
                score,
                chunk,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return scored[0]


# ---------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------

def semantic_rank(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> list[int]:
    """Return chunk indexes sorted by cosine similarity.

    Embeddings are already normalized, so dot product equals cosine.
    """

    scores = np.dot(
        chunk_embeddings,
        query_embedding,
    )

    ranked_indexes = np.argsort(
        scores
    )[::-1]

    return ranked_indexes.tolist()


# ---------------------------------------------------------------------
# Evaluate one chunk configuration
# ---------------------------------------------------------------------

def evaluate_configuration(
    config: dict[str, Any],
    source_documents,
    dataset: list[dict[str, Any]],
    embeddings_client,
) -> dict[str, Any]:

    chunk_size = config[
        "chunk_size"
    ]

    chunk_overlap = config[
        "chunk_overlap"
    ]

    print("=" * 76)

    print(
        f"Testing chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}"
    )

    print("=" * 76)

    chunk_start = time.perf_counter()

    chunks = chunk_documents(
        source_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunking_seconds = (
        time.perf_counter()
        - chunk_start
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    print(
        f"Generated chunks : {len(chunks)}"
    )

    print(
        f"Chunking time    : "
        f"{chunking_seconds:.3f}s"
    )

    print(
        "Generating chunk embeddings..."
    )

    embedding_start = (
        time.perf_counter()
    )

    document_embeddings = (
        embeddings_client
        .embed_documents(texts)
    )

    embedding_seconds = (
        time.perf_counter()
        - embedding_start
    )

    chunk_embedding_matrix = np.asarray(
        document_embeddings,
        dtype=np.float32,
    )

    print(
        f"Embedding time   : "
        f"{embedding_seconds:.3f}s"
    )

    print()

    reciprocal_ranks = []
    relevant_ranks = []
    query_latencies = []

    recall_at_4_hits = 0
    source_hits = 0

    query_results = []

    for item in dataset:
        question_id = item["id"]

        question = str(
            item["question"]
        )

        expected_answer = str(
            item["expected_answer"]
        )

        expected_document = str(
            item["expected_document"]
        )

        overlap_score, best_chunk = (
    get_best_relevant_chunk(
        chunks,
        expected_document,
        expected_answer,
    )
)

        if best_chunk is None:
            print(
                f"Q{question_id}: "
                f"expected document missing"
            )

            continue

        relevant_chunk_index = (
            chunks.index(best_chunk)
        )

        query_start = (
            time.perf_counter()
        )

        query_embedding = np.asarray(
            embeddings_client
            .embed_query(question),
            dtype=np.float32,
        )

        ranking = semantic_rank(
            query_embedding,
            chunk_embedding_matrix,
        )

        query_latency = (
            time.perf_counter()
            - query_start
        )

        query_latencies.append(
            query_latency
        )

        try:
            relevant_rank = (
                ranking.index(
                    relevant_chunk_index
                )
                + 1
            )
        except ValueError:
            relevant_rank = None

        if relevant_rank is not None:
            relevant_ranks.append(
                float(relevant_rank)
            )

        rr = reciprocal_rank(
            relevant_rank
        )

        reciprocal_ranks.append(rr)

        top_4_indexes = ranking[
            :RECALL_K
        ]

        if (
            relevant_chunk_index
            in top_4_indexes
        ):
            recall_at_4_hits += 1

        top_k_indexes = ranking[
            :TOP_K
        ]

        source_hit = any(
            filename_matches(
                chunks[index]
                .metadata
                .get(
                    "filename",
                    "",
                ),
                expected_document,
            )
            for index in top_k_indexes
        )

        if source_hit:
            source_hits += 1

        query_results.append(
            {
                "id": question_id,
                "question": question,
                "expected_document": (
                    expected_document
                ),
                "relevant_chunk_index": (
                    best_chunk.metadata.get(
                        "chunk_index"
                    )
                ),
                "answer_overlap_f1": round(
                    overlap_score,
                    4,
                ),
                "semantic_rank": (
                    relevant_rank
                ),
                "reciprocal_rank": round(
                    rr,
                    4,
                ),
                "recall_at_4": (
                    relevant_rank
                    is not None
                    and relevant_rank
                    <= RECALL_K
                ),
                "source_hit_at_8": (
                    source_hit
                ),
                "query_latency_ms": round(
                    query_latency
                    * 1000,
                    2,
                ),
            }
        )

        print(
            f"Q{question_id:<2} | "
            f"Relevant Rank: "
            f"{str(relevant_rank):<3} | "
            f"RR: {rr:.4f} | "
            f"Recall@4: "
            f"{'YES' if relevant_rank and relevant_rank <= 4 else 'NO'}"
        )

    successful_queries = len(
        query_results
    )

    mrr = average(
        reciprocal_ranks
    )

    avg_relevant_rank = average(
        relevant_ranks
    )

    recall_at_4 = (
        recall_at_4_hits
        / successful_queries
        if successful_queries
        else 0.0
    )

    source_hit_at_8 = (
        source_hits
        / successful_queries
        if successful_queries
        else 0.0
    )

    avg_query_latency_ms = (
        average(
            query_latencies
        )
        * 1000
    )

    average_chunk_length = (
        average(
            [
                float(
                    len(chunk.page_content)
                )
                for chunk in chunks
            ]
        )
    )

    print()

    print(
        f"MRR                    : "
        f"{mrr:.4f}"
    )

    print(
        f"Recall@4               : "
        f"{recall_at_4:.2%}"
    )

    print(
        f"Source Hit@8           : "
        f"{source_hit_at_8:.2%}"
    )

    print(
        f"Avg Relevant Rank      : "
        f"{avg_relevant_rank:.2f}"
    )

    print(
        f"Avg Query Latency      : "
        f"{avg_query_latency_ms:.2f} ms"
    )

    print(
        f"Avg Actual Chunk Length: "
        f"{average_chunk_length:.2f} chars"
    )

    print()

    return {
        "name": config["name"],
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": len(chunks),
        "average_actual_chunk_length": round(
            average_chunk_length,
            2,
        ),
        "chunking_seconds": round(
            chunking_seconds,
            4,
        ),
        "embedding_seconds": round(
            embedding_seconds,
            4,
        ),
        "successful_queries": (
            successful_queries
        ),
        "mrr": round(
            mrr,
            4,
        ),
        "recall_at_4": round(
            recall_at_4,
            4,
        ),
        "source_hit_at_8": round(
            source_hit_at_8,
            4,
        ),
        "average_relevant_rank": round(
            avg_relevant_rank,
            4,
        ),
        "average_query_latency_ms": round(
            avg_query_latency_ms,
            2,
        ),
        "queries": query_results,
    }


# ---------------------------------------------------------------------
# Determine best configuration
# ---------------------------------------------------------------------

def score_configuration(
    result: dict[str, Any],
) -> float:
    """Balanced retrieval-quality score.

    Primary importance:
    1. MRR
    2. Recall@4
    3. Moderate penalty for excessive chunk count

    This score is only used to rank tested configurations.
    """

    quality_score = (
        result["mrr"] * 0.60
        + result["recall_at_4"] * 0.40
    )

    return quality_score


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset missing: "
            f"{DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    print()
    print("=" * 76)

    print(
        "Advanced RAG - Chunk Size / Overlap Experiment"
    )

    print("=" * 76)

    print(
        f"Evaluation questions : "
        f"{len(dataset)}"
    )

    print(
        "Embedding model      : "
        f"{settings.EMBEDDING_MODEL_NAME}"
    )

    print(
        "Configurations       : "
        "300/50, 500/100, 800/150"
    )

    print("=" * 76)

    print()

    source_documents = (
        load_clean_source_documents()
    )

    embeddings_client = (
        get_embeddings_client()
    )

    all_results = []

    for config in CONFIGURATIONS:
        result = evaluate_configuration(
            config,
            source_documents,
            dataset,
            embeddings_client,
        )

        result[
            "balanced_quality_score"
        ] = round(
            score_configuration(
                result
            ),
            4,
        )

        all_results.append(
            result
        )

    ranked_results = sorted(
        all_results,
        key=lambda item: (
            item[
                "balanced_quality_score"
            ],
            item["mrr"],
            item["recall_at_4"],
        ),
        reverse=True,
    )

    winner = ranked_results[0]

    summary = {
        "experiment": (
            "RecursiveCharacterTextSplitter "
            "chunk-size and overlap comparison"
        ),
        "embedding_model": (
            settings.EMBEDDING_MODEL_NAME
        ),
        "evaluation_questions": (
            len(dataset)
        ),
        "top_k": TOP_K,
        "recall_k": RECALL_K,
        "configurations": [
            {
                "name": result["name"],
                "chunk_size": (
                    result["chunk_size"]
                ),
                "chunk_overlap": (
                    result["chunk_overlap"]
                ),
                "total_chunks": (
                    result["total_chunks"]
                ),
                "mrr": result["mrr"],
                "recall_at_4": (
                    result["recall_at_4"]
                ),
                "source_hit_at_8": (
                    result[
                        "source_hit_at_8"
                    ]
                ),
                "average_relevant_rank": (
                    result[
                        "average_relevant_rank"
                    ]
                ),
                "average_query_latency_ms": (
                    result[
                        "average_query_latency_ms"
                    ]
                ),
                "balanced_quality_score": (
                    result[
                        "balanced_quality_score"
                    ]
                ),
            }
            for result in all_results
        ],
        "recommended_configuration": {
            "name": winner["name"],
            "chunk_size": (
                winner["chunk_size"]
            ),
            "chunk_overlap": (
                winner["chunk_overlap"]
            ),
            "reason": (
                "Highest balanced score across "
                "MRR and Recall@4 among the "
                "tested configurations."
            ),
        },
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

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

    print(
        "FINAL CHUNKING EXPERIMENT COMPARISON"
    )

    print("=" * 76)

    print()

    print(
        f"{'Config':<12}"
        f"{'Chunks':>10}"
        f"{'MRR':>10}"
        f"{'Recall@4':>12}"
        f"{'Avg Rank':>12}"
        f"{'Latency':>12}"
    )

    print("-" * 68)

    for result in all_results:
        label = (
            f"{result['chunk_size']}/"
            f"{result['chunk_overlap']}"
        )

        print(
            f"{label:<12}"
            f"{result['total_chunks']:>10}"
            f"{result['mrr']:>10.4f}"
            f"{result['recall_at_4']:>11.2%}"
            f"{result['average_relevant_rank']:>12.2f}"
            f"{result['average_query_latency_ms']:>10.2f}ms"
        )

    print()

    print(
        "Recommended configuration: "
        f"{winner['chunk_size']}/"
        f"{winner['chunk_overlap']}"
    )

    print(
        "Balanced quality score    : "
        f"{winner['balanced_quality_score']:.4f}"
    )

    print()

    print(
        "IMPORTANT: Recommendation is based "
        "on measured experimental results, "
        "not a hard-coded preference."
    )

    print()

    print("Saved:")

    print(
        f"  {RESULTS_PATH}"
    )

    print(
        f"  {SUMMARY_PATH}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()