# session_store.py
import json
import os
import redis
from typing import List, Dict, Any

# Reads REDIS_URL from env (defaults to localhost)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# One global client is fine; redis-py manages pooling under the hood
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Key namespace and TTL (seconds)
SESSION_NS = "call_session:"
SESSION_TTL = 60 * 60  # 1 hour

def _key(call_sid: str) -> str:
    return f"{SESSION_NS}{call_sid}"

def get_history(call_sid: str) -> List[Dict[str, Any]]:
    """
    Returns the prior chat history list for this call_sid or [] if not present.
    """
    raw = r.get(_key(call_sid))
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []

def set_history(call_sid: str, history: List[Dict[str, Any]]) -> None:
    """
    Persists the chat history list (JSON) with a TTL.
    """
    r.set(_key(call_sid), json.dumps(history), ex=SESSION_TTL)

def clear_history(call_sid: str) -> None:
    """
    Deletes any stored history for this call.
    """
    r.delete(_key(call_sid))
