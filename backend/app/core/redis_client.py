import json
from typing import Any, Optional

import redis

try:
    from backend.app.core.config import settings
except ImportError:
    from app.core.config import settings


class RedisClient:
    def __init__(self):
        self.client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def set_json(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
    ) -> bool:
        try:
            payload = json.dumps(value, ensure_ascii=False)
            return bool(
                self.client.set(
                    key,
                    payload,
                    ex=ttl_seconds,
                )
            )
        except Exception:
            return False

    def get_json(self, key: str) -> Optional[Any]:
        try:
            value = self.client.get(key)

            if value is None:
                return None

            return json.loads(value)

        except Exception:
            return None

    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(key))
        except Exception:
            return False


redis_client = RedisClient()