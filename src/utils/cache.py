"""LRU cache with a single API: get, put, clear, stats."""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from typing import Any, Optional


class LRUCache:
    def __init__(self, max_size: int = 100, enabled: bool = True) -> None:
        self.max_size = max(1, int(max_size))
        self.enabled = bool(enabled)
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        self._store[key] = value
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enabled,
            "size": len(self._store),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
        }

    @staticmethod
    def make_key(*parts: str) -> str:
        joined = "\n||\n".join(parts)
        return sha256(joined.encode("utf-8")).hexdigest()
