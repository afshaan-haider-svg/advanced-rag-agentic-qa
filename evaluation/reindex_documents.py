from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from backend.app.rag.loaders import load_document
from backend.app.rag.cleaning import clean_documents
from backend.app.rag.chunking import chunk_documents
from backend.app.rag.vector_store import (
    delete_document_chunks,
    index_chunks,
)


def main():
    registry_path = Path(settings.REGISTRY_FILE_PATH)

    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    print("Starting re-index with:")
    print(f"  CHUNK_SIZE    = {settings.CHUNK_SIZE}")
    print(f"  CHUNK_OVERLAP = {settings.CHUNK_OVERLAP}")
    print()

    for document_id, record in registry.items():
        filename = record["filename"]

        source_path = (
            Path(settings.UPLOAD_DIR)
            / f"{document_id}_{filename}"
        )

        if not source_path.exists():
            print(f"SKIPPED: source file missing -> {filename}")
            continue

        print("=" * 70)
        print(f"Document: {filename}")
        print(f"ID      : {document_id}")

        deleted = delete_document_chunks(document_id)
        print(f"Deleted old chunks: {deleted}")

        raw_documents = load_document(
            source_path,
            document_id,
            filename,
        )

        cleaned_documents = clean_documents(raw_documents)

        chunks = chunk_documents(
            cleaned_documents,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        indexed = index_chunks(chunks)

        record["num_chunks"] = indexed
        record["status"] = "indexed"
        record["error_message"] = None

        print(f"Indexed new chunks: {indexed}")

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(
            registry,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("Re-index complete.")
    print("Registry updated successfully.")


if __name__ == "__main__":
    main()