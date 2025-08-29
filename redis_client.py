import os
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import redis


REDIS_HOST = os.getenv("REDIS_HOST", "redis-17316.c330.asia-south1-1.gce.redns.redis-cloud.com")
REDIS_PORT = int(os.getenv("REDIS_PORT", "17316"))
REDIS_USER = os.getenv("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "QVoPG0oMdB7i7L9TU7qfNB08vRRRrxKm")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(30 * 60)))


def get_redis() -> Optional[redis.Redis]:
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            username=REDIS_USER,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        client.ping()
        return client
    except Exception:
        return None


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def _last_activity_key(session_id: str) -> str:
    return f"session:{session_id}:last_activity"


def save_session_snapshot(client: redis.Redis, session_id: str, snapshot: Dict[str, Any], ttl: Optional[int] = None) -> None:
    payload = json.dumps(snapshot, default=str)
    client.set(_session_key(session_id), payload, ex=ttl or SESSION_TTL_SECONDS)


def load_session_snapshot(client: redis.Redis, session_id: str) -> Optional[Dict[str, Any]]:
    raw = client.get(_session_key(session_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def touch_session(client: redis.Redis, session_id: str, ttl: Optional[int] = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    client.set(_last_activity_key(session_id), now, ex=ttl or SESSION_TTL_SECONDS)
    client.expire(_session_key(session_id), ttl or SESSION_TTL_SECONDS)


def delete_session(client: redis.Redis, session_id: str) -> None:
    client.delete(_session_key(session_id))
    client.delete(_last_activity_key(session_id))


