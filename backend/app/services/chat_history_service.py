"""PostgreSQL-backed chat history with Redis caching and JSON fallback."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

try:
    from backend.app.core.config import settings
    from backend.app.core.database import SessionLocal
    from backend.app.core.redis_client import redis_client
    from backend.app.models.chat_history_db import ChatHistory
except ImportError:
    from app.core.config import settings
    from app.core.database import SessionLocal
    from app.core.redis_client import redis_client
    from app.models.chat_history_db import ChatHistory


class ChatHistoryService:
    def __init__(self):
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return Path(settings.CHAT_HISTORY_FILE_PATH)

    def _cache_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"

    def _read_json(self):
        if not self.path.exists():
            return {}

        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _append_json(self, session_id, user_message, answer, citations):
        with self._lock:
            data = self._read_json()

            data.setdefault(session_id, []).append(
                {
                    "user_message": user_message,
                    "assistant_answer": answer,
                    "citations": citations,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            self.path.parent.mkdir(parents=True, exist_ok=True)

            self.path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def append(self, session_id, user_message, answer, citations):
        try:
            with SessionLocal() as db:
                record = ChatHistory(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_answer=answer,
                    citations=citations,
                )

                db.add(record)
                db.commit()

            redis_client.delete(self._cache_key(session_id))

        except Exception:
            self._append_json(
                session_id=session_id,
                user_message=user_message,
                answer=answer,
                citations=citations,
            )

    def get(self, session_id):
        cache_key = self._cache_key(session_id)

        cached_history = redis_client.get_json(cache_key)

        if cached_history is not None:
            return cached_history

        try:
            with SessionLocal() as db:
                stmt = (
                    select(ChatHistory)
                    .where(ChatHistory.session_id == session_id)
                    .order_by(ChatHistory.created_at.asc())
                )

                records = db.execute(stmt).scalars().all()

                history = [
                    {
                        "user_message": record.user_message,
                        "assistant_answer": record.assistant_answer,
                        "citations": record.citations or [],
                        "timestamp": (
                            record.created_at.isoformat()
                            if record.created_at
                            else None
                        ),
                    }
                    for record in records
                ]

            redis_client.set_json(
                cache_key,
                history,
                ttl_seconds=3600,
            )

            return history

        except Exception:
            return self._read_json().get(session_id, [])


chat_history_service = ChatHistoryService()