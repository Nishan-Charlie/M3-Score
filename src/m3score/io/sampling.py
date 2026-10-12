"""Choosing which images of a source are scored.

A subset is always a seeded random draw. Taking the first N files of a sorted folder
selects a handful of subjects, which inflates every distance; see
docs/REFERENCE_COHORT_FINDING.md in the repository.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import protocol


def sample_indices(
    n_total: int, n: int | None, groups: Sequence[str] | None, seed: int
) -> list[int]:
    """Indices of the items to score, in ascending order.

    ``n=None`` means everything up to ``protocol.MAX_IMAGES``. With known ``groups``
    the draw takes one item per subject in rotation, subjects and items both shuffled,
    so the sample is balanced across subjects and spans as many as possible.
    """
    if n is None:
        n = protocol.MAX_IMAGES
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if groups is not None and len(groups) != n_total:
        raise ValueError(f"got {len(groups)} group labels for {n_total} items")
    if n >= n_total:
        return list(range(n_total))

    rng = np.random.default_rng(seed)
    if groups is None:
        return sorted(int(i) for i in rng.choice(n_total, size=n, replace=False))

    by_subject: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_subject.setdefault(group, []).append(index)
    subjects = sorted(by_subject)
    pools = [
        [int(i) for i in rng.permutation(by_subject[subjects[s]])]
        for s in rng.permutation(len(subjects))
    ]
    chosen: list[int] = []
    depth = 0
    while len(chosen) < n:
        for pool in pools:
            if depth < len(pool):
                chosen.append(pool[depth])
                if len(chosen) == n:
                    break
        depth += 1
    return sorted(chosen)
