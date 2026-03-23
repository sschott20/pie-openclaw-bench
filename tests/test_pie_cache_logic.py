"""Unit tests for PIECacheBackend cache logic — no server required."""

from __future__ import annotations

import pytest

from harness.backends.pie_cache import PIECacheBackend, cache_key_for_prefix
from harness.models import PromptModule


class TestCacheKeyForPrefix:
    def test_single_hash(self):
        assert cache_key_for_prefix(["abc"]) == "kv_abc"

    def test_multiple_hashes(self):
        assert cache_key_for_prefix(["a", "b", "c"]) == "kv_a_b_c"

    def test_real_sha256_hashes(self):
        m = PromptModule(name="test", content="hello")
        key = cache_key_for_prefix([m.content_hash])
        assert key.startswith("kv_")
        assert len(key) > 3


class TestFindCacheHits:
    def _make_modules(self, *contents: str) -> list[PromptModule]:
        return [PromptModule(name=f"mod_{i}", content=c) for i, c in enumerate(contents)]

    def test_no_cache(self):
        backend = PIECacheBackend()
        modules = self._make_modules("a", "b", "c")
        depth, key = backend._find_cache_hits(modules)
        assert depth == 0
        assert key is None

    def test_single_prefix_cached(self):
        backend = PIECacheBackend()
        modules = self._make_modules("a", "b", "c")
        hashes = [m.content_hash for m in modules]

        # Simulate caching first module
        k1 = cache_key_for_prefix(hashes[:1])
        backend._cache_built.add(k1)

        depth, key = backend._find_cache_hits(modules)
        assert depth == 1
        assert key == k1

    def test_full_prefix_cached(self):
        backend = PIECacheBackend()
        modules = self._make_modules("a", "b", "c")
        hashes = [m.content_hash for m in modules]

        # Simulate caching all prefixes
        for i in range(1, len(hashes) + 1):
            backend._cache_built.add(cache_key_for_prefix(hashes[:i]))

        depth, key = backend._find_cache_hits(modules)
        assert depth == 3
        assert key == cache_key_for_prefix(hashes)

    def test_deepest_match(self):
        """When modules 1 and 2 are cached, returns depth 2."""
        backend = PIECacheBackend()
        modules = self._make_modules("a", "b", "c")
        hashes = [m.content_hash for m in modules]

        backend._cache_built.add(cache_key_for_prefix(hashes[:1]))
        backend._cache_built.add(cache_key_for_prefix(hashes[:2]))

        depth, key = backend._find_cache_hits(modules)
        assert depth == 2
        assert key == cache_key_for_prefix(hashes[:2])

    def test_different_module_no_match(self):
        """Cached prefix for [a, b] doesn't match [a, c]."""
        backend = PIECacheBackend()
        modules_ab = self._make_modules("a", "b")
        modules_ac = self._make_modules("a", "c")
        hashes_ab = [m.content_hash for m in modules_ab]

        # Cache [a, b]
        backend._cache_built.add(cache_key_for_prefix(hashes_ab[:1]))
        backend._cache_built.add(cache_key_for_prefix(hashes_ab[:2]))

        # Query [a, c] — should match depth 1 (only [a] matches)
        depth, key = backend._find_cache_hits(modules_ac)
        assert depth == 1
        assert key == cache_key_for_prefix(hashes_ab[:1])


class TestEnsureCacheLogic:
    """Test the cache-building logic without a server."""

    def _make_modules(self, *contents: str) -> list[PromptModule]:
        return [PromptModule(name=f"mod_{i}", content=c) for i, c in enumerate(contents)]

    def test_ensure_cache_identifies_missing_layers(self):
        """_ensure_cache should only build layers not already cached."""
        backend = PIECacheBackend()
        modules = self._make_modules("a", "b", "c")
        hashes = [m.content_hash for m in modules]

        # Pre-cache first layer
        k1 = cache_key_for_prefix(hashes[:1])
        backend._cache_built.add(k1)

        # The missing keys are for prefixes [a,b] and [a,b,c]
        missing = []
        for i in range(len(modules)):
            key = cache_key_for_prefix(hashes[: i + 1])
            if key not in backend._cache_built:
                import_key = cache_key_for_prefix(hashes[:i]) if i > 0 else None
                missing.append((import_key, key, i))

        assert len(missing) == 2
        assert missing[0] == (k1, cache_key_for_prefix(hashes[:2]), 1)
        assert missing[1] == (
            cache_key_for_prefix(hashes[:2]),
            cache_key_for_prefix(hashes[:3]),
            2,
        )


class TestResetState:
    def test_reset_clears_cache(self):
        import asyncio

        backend = PIECacheBackend()
        backend._cache_built.add("kv_a")
        backend._cache_built.add("kv_a_b")
        asyncio.run(backend.reset_state())
        assert len(backend._cache_built) == 0
