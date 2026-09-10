try:
    from backend.app.core.database import Base, engine
    from backend.app.models.chat_history_db import ChatHistory
except ImportError:
    from app.core.database import Base, engine
    from app.models.chat_history_db import ChatHistory


def init_db():
    Base.metadata.create_all(bind=engine)