"""Tests for pool discovery and phase map utilities."""

import pytest
from spaces_game.callbacks.pool_utils import (
    discover_pools,
    build_phase_map,
    LEGACY_POOL_ORDER,
    DIFFICULTY_CHECKPOINTS,
)


class TestBuildPhaseMap:
    """Tests for build_phase_map()."""

    def test_zero_pools(self):
        result = build_phase_map(0)
        assert result == {0: [0]}

    def test_single_pool(self):
        result = build_phase_map(1)
        assert result == {0: [0]}

    def test_two_pools(self):
        result = build_phase_map(2)
        assert result == {
            0: [0],        # pool 0 solo
            1: [1],        # pool 1 solo
            2: [0, 1],     # mix
        }

    def test_three_pools(self):
        result = build_phase_map(3)
        assert result == {
            0: [0],           # pool 0 solo
            1: [1],           # pool 1 solo
            2: [0, 1],        # mix 0+1
            3: [2],           # pool 2 solo
            4: [0, 1, 2],     # mix all
        }

    def test_four_pools(self):
        result = build_phase_map(4)
        assert len(result) == 7  # 1 + 2*3
        assert result[0] == [0]
        assert result[6] == [0, 1, 2, 3]

    def test_phase_count_formula(self):
        """Phase count = 1 + 2*(num_pools-1) for num_pools >= 2."""
        for n in range(2, 8):
            result = build_phase_map(n)
            expected = 1 + 2 * (n - 1)
            assert len(result) == expected, f"num_pools={n}"

    def test_max_phase_includes_all_pools(self):
        """The final phase should always include all pools."""
        for n in range(1, 6):
            result = build_phase_map(n)
            max_phase = max(result.keys())
            assert result[max_phase] == list(range(n))


class TestDiscoverPools:
    """Tests for discover_pools() — uses actual boards/ directory."""

    def test_size2_has_pools(self):
        pools = discover_pools(2)
        assert len(pools) > 0
        assert all(p.endswith(".json") for p in pools)

    def test_size3_has_pools(self):
        pools = discover_pools(3)
        assert len(pools) > 0

    def test_nonexistent_size_returns_empty(self):
        pools = discover_pools(99)
        assert pools == []

    def test_legacy_ordering(self):
        """If size2 has legacy-named pools, they should be in LEGACY_POOL_ORDER."""
        pools = discover_pools(2)
        if not pools:
            pytest.skip("No size-2 pools")
        from pathlib import Path
        stems = [Path(p).stem for p in pools]
        legacy_found = [s for s in stems if s in LEGACY_POOL_ORDER]
        if len(legacy_found) >= 2:
            indices = [LEGACY_POOL_ORDER.index(s) for s in legacy_found]
            assert indices == sorted(indices), "Legacy pools should be in LEGACY_POOL_ORDER"


class TestConstants:
    def test_legacy_pool_order(self):
        assert "simple" in LEGACY_POOL_ORDER
        assert "one_trap" in LEGACY_POOL_ORDER

    def test_difficulty_checkpoints(self):
        assert DIFFICULTY_CHECKPOINTS[0] == "beginner"
        assert DIFFICULTY_CHECKPOINTS[2] == "intermediate"
