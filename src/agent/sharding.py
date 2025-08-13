# src/agent/sharding.py
from __future__ import annotations

import hashlib


def parse_csv(v: str | None) -> list[str]:
    """Парсит CSV-список из env-переменной в список, отбрасывая пустые элементы."""
    if not v:
        return []
    return [item.strip() for item in v.split(",") if item.strip()]

def _stable_hash(key: str) -> int:
    """Детерминированный 64-битный хэш (sha256 → первые 8 байт)."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big", signed=False)

def select_shard(thread_id: str, shard_count: int) -> int:
    """Определяет индекс шарда по thread_id (равномерное распределение)."""
    if shard_count <= 1:
        return 0
    return _stable_hash(thread_id) % shard_count
