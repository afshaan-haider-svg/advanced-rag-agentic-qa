"""Application configuration using Pydantic Settings."""

import json
from pathlib import Path
from typing import List, Set, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core Application Info
    PROJECT_NAME: str = "Advanced RAG & Agentic Document QA System"
    SERVICE_NAME: str = "advanced-rag-api"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # CORS Configuration
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Storage and Persistence Paths
    DATA_DIR: Path = Path("data")
    UPLOAD_DIR: Path = Path("data/uploads")
    CHROMA_PERSIST_DIR: Path = Path("data/chroma")
    REGISTRY_FILE_PATH: Path = Path("data/document_registry.json")

    # Vector Database Settings
    CHROMA_COLLECTION_NAME: str = "document_chunks"

    # Embedding Model Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # Document Chunking Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Phase 3: Retrieval, MMR & Re-ranking Settings
    RETRIEVAL_TOP_K: int = 8
    SIMILARITY_THRESHOLD: float = 0.30
    RERANK_TOP_N: int = 4
    CONTEXT_MAX_CHARS: int = 12000
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_DEVICE: str = "cpu"
    MMR_LAMBDA: float = 0.7
    MMR_CANDIDATES: int = 16

    # Phase 4: LangGraph / LLM / Chat
    LLM_PROVIDER: str = "gemini"
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-3.6-flash"
    LLM_TEMPERATURE: float = 0.0
    MAX_RETRIEVAL_RETRIES: int = 2
    CHAT_HISTORY_FILE_PATH: Path = Path("data/chat_history.json")
    
    
        # Phase 6: PostgreSQL & Redis
    DATABASE_URL: str = "postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    
    # Supported File Formats
    SUPPORTED_EXTENSIONS: Set[str] = {".pdf", ".txt", ".docx"}
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from comma-separated string, JSON string, or list."""
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                try:
                    return json.loads(v_stripped)
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in v_stripped.split(",") if origin.strip()]
        if isinstance(v, list):
            return [str(origin).strip() for origin in v]
        return ["*"]


settings = Settings()
