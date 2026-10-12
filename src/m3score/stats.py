"""Statistics of the m3-v1 protocol, as pure functions on (N, D) feature tensors.

Nothing here knows about images or models. The fidelity, memorization and coverage
arithmetic mirrors the research implementation in ``evaluation/m3_score_v2.py``
operation for operation; ``tests/test_parity.py`` holds the two together.
"""

from __future__ import annotations

import torch

from . import protocol

_EPS_L2 = 1e-8


def _l2_normalise(x: torch.Tensor) -> torch.Tensor:
    """Project each row onto the unit sphere."""
    return x / x.norm(dim=1, keepdim=True).clamp(min=_EPS_L2)


def _sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances, (Na, Nb)."""
    return ((a * a).sum(1, keepdim=True) + (b * b).sum(1) - 2.0 * (a @ b.t())).clamp(min=0.0)


def _median_bandwidth(d: torch.Tensor) -> float:
    """Median of the upper triangle of a pooled squared-distance matrix, floored at 1e-10."""
    n = d.shape[0]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=d.device), diagonal=1)
    return float(d[upper].median().clamp(min=1e-10))


def mmd2_rbf(real: torch.Tensor, gen: torch.Tensor, *, clamp: bool = True) -> float:
    """Unbiased MMD^2 under a mixture of Gaussian RBF kernels, on L2-normalised features.

    Bandwidths are the pooled median squared distance times each of
    ``protocol.RBF_BANDWIDTH_SCALES``. With ``clamp`` the estimate is floored at 0.
    """
    x = _l2_normalise(real.float())
    y = _l2_normalise(gen.float())
    m, n = x.shape[0], y.shape[0]
    if m < 2 or n < 2:
        raise ValueError(f"MMD^2 needs at least 2 samples per set; got {m} and {n}")
    dxx, dyy, dxy = _sq_dist(x, x), _sq_dist(y, y), _sq_dist(x, y)
    joint = torch.cat([x, y], dim=0)
    sigma2 = _median_bandwidth(_sq_dist(joint, joint))

    total = torch.zeros(1, device=x.device)
    for scale in protocol.RBF_BANDWIDTH_SCALES:
        gamma = 1.0 / (2.0 * (sigma2 * scale))
        kxx = torch.exp(-gamma * dxx)
        kyy = torch.exp(-gamma * dyy)
        kxy = torch.exp(-gamma * dxy)
        t_rr = (kxx.sum() - kxx.trace()) / (m * (m - 1))
        t_gg = (kyy.sum() - kyy.trace()) / (n * (n - 1))
        total = total + (t_rr + t_gg - 2.0 * kxy.mean())
    value = total / len(protocol.RBF_BANDWIDTH_SCALES)
    return float(value.clamp(min=0.0) if clamp else value)


def _kth_nn_dist(query: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each query row to its k-th nearest row of ``ref``."""
    d2 = torch.cdist(query, ref, p=2).pow(2)
    return d2.topk(k, largest=False, dim=1).values[:, -1].sqrt()


def _kth_nn_dist_within(x: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each row to its k-th nearest other row of the same set."""
    d2 = torch.cdist(x, x, p=2).pow(2)
    d2.fill_diagonal_(float("inf"))
    return d2.topk(k, largest=False, dim=1).values[:, -1].sqrt()


def memorization(real: torch.Tensor, gen: torch.Tensor) -> tuple[float, float]:
    """Near-duplicate rate and mean nearest-real distance.

    A generated point is flagged when it is closer to a real point than the median
    real point is to its own nearest real neighbour.
    """
    real, gen = real.float(), gen.float()
    gen_to_real = _kth_nn_dist(gen, real, 1)
    threshold = _kth_nn_dist_within(real, 1).median().item()
    return float((gen_to_real < threshold).float().mean()), float(gen_to_real.mean())


def coverage(real: torch.Tensor, gen: torch.Tensor, k: int = protocol.COVERAGE_K) -> float:
    """Fraction of real points with a generated point inside their own k-NN radius.

    The radius is estimated within the real set (Naeem et al. 2020), so removing
    generated samples can only lower the value.
    """
    real, gen = real.float(), gen.float()
    radii = _kth_nn_dist_within(real, k)
    nearest_gen = torch.cdist(real, gen, p=2).min(dim=1).values
    return float((nearest_gen <= radii).float().mean())


def precision_recall(
    real: torch.Tensor, gen: torch.Tensor, k: int = protocol.COVERAGE_K
) -> tuple[float, float]:
    """k-NN precision and recall (Kynkaanniemi et al. 2019).

    A point counts when it lies inside the k-NN ball of *any* point of the other set.
    Not part of the M3 axes: recall rises as generated diversity is removed, which is
    why M3 reports ``coverage`` instead. The research code's ``_manifold_fraction``
    tests only the nearest reference point's ball and returns lower values.
    """
    real, gen = real.float(), gen.float()
    if real.shape[0] <= k or gen.shape[0] <= k:
        raise ValueError(
            f"precision/recall needs more than k={k} points per set; "
            f"got {real.shape[0]} and {gen.shape[0]}"
        )

    def inside_any(query: torch.Tensor, ref: torch.Tensor, radii: torch.Tensor) -> float:
        return float((torch.cdist(query, ref, p=2) <= radii[None, :]).any(dim=1).float().mean())

    precision = inside_any(gen, real, _kth_nn_dist_within(real, k))
    recall = inside_any(real, gen, _kth_nn_dist_within(gen, k))
    return precision, recall
