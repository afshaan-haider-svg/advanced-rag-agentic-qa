from sqlalchemy import Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.sql import func

try:
    from backend.app.core.database import Base
except ImportError:
    from app.core.database import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    assistant_answer = Column(Text, nullable=False)
    citations = Column(JSON, nullable=False, default=list)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )