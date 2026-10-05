from __future__ import annotations

from src.utils.cache import LRUCache


def test_cache_get_put_clear_stats() -> None:
    cache = LRUCache(max_size=2, enabled=True)
    assert cache.get("a") is None
    cache.put("a", {"score": 1})
    cache.put("b", {"score": 2})
    assert cache.get("a") == {"score": 1}
    cache.put("c", {"score": 3})
    assert cache.get("b") is None
    assert cache.get("c") == {"score": 3}
    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
    cache.clear()
    assert cache.get("c") is None
    assert cache.stats()["size"] == 0


def test_disabled_cache_never_stores() -> None:
    cache = LRUCache(max_size=10, enabled=False)
    cache.put("k", 123)
    assert cache.get("k") is None
    assert cache.stats()["size"] == 0
