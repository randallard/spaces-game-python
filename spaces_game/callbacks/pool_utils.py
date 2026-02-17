"""Pool discovery and phase map utilities for opponent curriculum."""

import re
from pathlib import Path
from typing import List, Dict


# Legacy ordering for pool files without numeric prefixes.
LEGACY_POOL_ORDER = ["simple", "one_trap", "super_move", "super_move_counter"]

# Map opponent phase completions to difficulty checkpoint names.
# When the agent finishes training on phase N, save as the corresponding difficulty.
DIFFICULTY_CHECKPOINTS = {
    0: "beginner",       # Can build valid boards, beats simple opponents
    2: "intermediate",   # Handles traps and mixed opponents
}
# "expert" is saved at phase 5 completion OR at training end (whichever comes last)


def discover_pools(board_size: int) -> List[str]:
    """
    Discover all .json board pool files in boards/sizeN/.

    Sorting rules:
    - Files starting with a digit (e.g. 00_simple.json) sort by their numeric
      prefix.
    - Files without a numeric prefix sort by LEGACY_POOL_ORDER, with unknown
      names appended alphabetically after.

    Returns list of paths sorted in curriculum order.
    """
    pool_dir = Path(f"boards/size{board_size}")
    if not pool_dir.is_dir():
        return []

    numbered = []   # (number, path)
    legacy = []     # (order_index, path)
    unknown = []    # (stem, path)

    for p in pool_dir.glob("*.json"):
        match = re.match(r'^(\d+)', p.stem)
        if match:
            numbered.append((int(match.group(1)), str(p)))
        elif p.stem in LEGACY_POOL_ORDER:
            legacy.append((LEGACY_POOL_ORDER.index(p.stem), str(p)))
        else:
            unknown.append((p.stem, str(p)))

    # If any files have numeric prefixes, use that ordering for all numbered
    # files, then append any non-numbered legacy/unknown files after.
    numbered.sort(key=lambda x: x[0])
    legacy.sort(key=lambda x: x[0])
    unknown.sort(key=lambda x: x[0])

    pools = [path for _, path in numbered]
    pools += [path for _, path in legacy]
    pools += [path for _, path in unknown]
    return pools


def build_phase_map(num_pools: int) -> Dict[int, List[int]]:
    """
    Build a progressive opponent phase map for the given number of pools.

    Pattern:
    - Phase 0: pool 0 solo
    - Phase 1: pool 1 solo
    - Phase 2: pools 0+1 mixed
    - Phase 3: pool 2 solo
    - Phase 4: pools 0+1+2 mixed
    - ...
    - Final phase: all pools mixed

    For each pool after the first, we add two phases: solo then cumulative mix.
    Single pool gets just one phase.
    """
    if num_pools == 0:
        return {0: [0]}

    if num_pools == 1:
        return {0: [0]}

    phase_map: Dict[int, List[int]] = {}
    phase = 0

    # Phase 0: first pool solo
    phase_map[phase] = [0]
    phase += 1

    # For each subsequent pool: solo phase, then cumulative mix
    for i in range(1, num_pools):
        # Solo phase for this pool
        phase_map[phase] = [i]
        phase += 1
        # Cumulative mix of all pools seen so far
        phase_map[phase] = list(range(i + 1))
        phase += 1

    return phase_map
