"""Local Hugging Face sentence-transformers embedding model for semantic search."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer

try:
    from backend.app.core.config import settings
    from backend.app.core.logging import get_logger
except ImportError:
    from app.core.config import settings
    from app.core.logging import get_logger

logger = get_logger(__name__)

_embedding_model_instance: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Retrieve or initialize the cached SentenceTransformer model instance on CPU."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        logger.info(
            "Loading local embedding model '%s' on %s...",
            settings.EMBEDDING_MODEL_NAME,
            settings.EMBEDDING_DEVICE,
        )
        _embedding_model_instance = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
        )
        logger.info("Local embedding model loaded successfully.")
    return _embedding_model_instance


class LocalHuggingFaceEmbeddings:
    """Wrapper adhering to standard embedding protocols for Chroma and LangChain."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute normalized vector embeddings for document chunks."""
        if not texts:
            return []
        model = get_embedding_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute normalized vector embedding for a query string."""
        model = get_embedding_model()
        embedding = model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()


def get_embeddings_client() -> LocalHuggingFaceEmbeddings:
    """Get standard embedding client wrapper."""
    return LocalHuggingFaceEmbeddings()
