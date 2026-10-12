"""Statistics of the m3-v1 protocol, as pure functions on (N, D) feature tensors.

Nothing here knows about images or models. The fidelity, memorization and coverage
arithmetic mirrors the research implementation in ``evaluation/m3_score_v2.py``
operation for operation; ``tests/test_parity.py`` holds the two together.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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


def _pooled_kernel(real: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
    """Bandwidth-averaged RBF kernel on the pooled sample, real rows first.

    Held in float64 so that sums over millions of entries keep their precision.
    Averaging the kernels equals averaging the per-bandwidth MMD^2 estimates.
    """
    pooled = _l2_normalise(torch.cat([real.float(), gen.float()], dim=0))
    d = _sq_dist(pooled, pooled)
    sigma2 = _median_bandwidth(d)
    dtype = torch.float32 if d.device.type == "mps" else torch.float64
    kernel = torch.zeros(d.shape, dtype=dtype, device=d.device)
    for scale in protocol.RBF_BANDWIDTH_SCALES:
        kernel += torch.exp(-d / (2.0 * (sigma2 * scale))).to(dtype)
    return kernel / len(protocol.RBF_BANDWIDTH_SCALES)


def bootstrap_ci(
    real: torch.Tensor,
    gen: torch.Tensor,
    *,
    n_bootstrap: int = protocol.N_BOOTSTRAP,
    alpha: float = protocol.CI_ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the fidelity MMD^2.

    Real and generated sets are resampled independently with replacement. The kernel
    is the one used for the point estimate. A resample contains repeated images;
    within-set pairs that are two copies of the same original are left out of the
    within-set averages, which keeps the bootstrap distribution centred on the point
    estimate. (Counting them, as the research code does, shifts the interval upward.)
    """
    nr, ng = real.shape[0], gen.shape[0]
    kernel = _pooled_kernel(real, gen)
    k_rr, k_gg, k_rg = kernel[:nr, :nr], kernel[nr:, nr:], kernel[:nr, nr:]
    diag_r, diag_g = k_rr.diagonal(), k_gg.diagonal()

    rng = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(n_bootstrap):
        # multiplicity of each original image in this resample
        c_r = torch.bincount(torch.randint(0, nr, (nr,), generator=rng), minlength=nr).to(kernel)
        c_g = torch.bincount(torch.randint(0, ng, (ng,), generator=rng), minlength=ng).to(kernel)
        pairs_r = nr * nr - float((c_r * c_r).sum())
        pairs_g = ng * ng - float((c_g * c_g).sum())
        if pairs_r <= 0 or pairs_g <= 0:  # a resample made of one image repeated
            continue
        t_rr = (torch.dot(c_r, k_rr @ c_r) - torch.dot(c_r * c_r, diag_r)) / pairs_r
        t_gg = (torch.dot(c_g, k_gg @ c_g) - torch.dot(c_g * c_g, diag_g)) / pairs_g
        t_rg = torch.dot(c_r, k_rg @ c_g) / (nr * ng)
        values.append(float(t_rr + t_gg - 2.0 * t_rg))
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    low = max(float(np.percentile(arr, 100 * alpha / 2)), 0.0)
    high = max(float(np.percentile(arr, 100 * (1 - alpha / 2))), 0.0)
    return low, high


def permutation_test(
    real: torch.Tensor,
    gen: torch.Tensor,
    *,
    n_permutations: int = protocol.N_PERMUTATIONS,
    seed: int = 0,
) -> tuple[float, float, float]:
    """One-sided permutation test of H0: MMD^2(real, gen) = 0.

    Returns ``(p_value, z_score, null_mean)``. The pooled median, and so every
    bandwidth, is unchanged by relabelling, so the null is evaluated on one pooled
    kernel matrix: each permutation costs a single matrix-vector product.
    """
    nr, ng = real.shape[0], gen.shape[0]
    n = nr + ng
    observed = mmd2_rbf(real, gen)
    kernel = _pooled_kernel(real, gen)
    diag = kernel.diagonal().clone()
    total, diag_total = kernel.sum(), diag.sum()

    rng = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_permutations):
        perm = torch.randperm(n, generator=rng)
        is_real = torch.zeros(n, dtype=kernel.dtype, device=kernel.device)
        is_real[perm[:nr].to(kernel.device)] = 1.0
        k_real = kernel @ is_real
        sum_rr = torch.dot(is_real, k_real)
        sum_rg = k_real.sum() - sum_rr
        sum_gg = total - sum_rr - 2.0 * sum_rg
        trace_r = torch.dot(is_real, diag)
        value = (
            (sum_rr - trace_r) / (nr * (nr - 1))
            + (sum_gg - (diag_total - trace_r)) / (ng * (ng - 1))
            - 2.0 * sum_rg / (nr * ng)
        )
        null.append(max(float(value), 0.0))
    arr = np.array(null)
    exceed = int((arr >= observed).sum())
    p_value = (1 + exceed) / (1 + n_permutations)
    z_score = float((observed - arr.mean()) / max(float(arr.std()), 1e-10))
    return float(p_value), z_score, float(arr.mean())


def real_floor(
    real: torch.Tensor,
    groups: Sequence[str] | None = None,
    *,
    n_splits: int = protocol.FLOOR_SPLITS,
    seed: int = 0,
) -> tuple[float, float, bool] | None:
    """Real-vs-real MMD^2 over random halvings of the real set.

    Gives the fidelity value a scale: this is what the metric reads when both sides
    are real. With two or more known subjects the halves are subject-disjoint.
    Returns ``(mean clamped at 0, sample std, subject_disjoint)``, or ``None`` when no
    split leaves two images on each side.
    """
    real = real.float()
    n = real.shape[0]
    subjects = sorted(set(groups)) if groups is not None else []
    by_subject = len(subjects) >= 2
    rng = np.random.default_rng(seed)

    values = []
    for _ in range(n_splits):
        if by_subject:
            order = rng.permutation(len(subjects))
            half = {subjects[i] for i in order[: len(subjects) // 2]}
            in_a = np.fromiter((g in half for g in groups), dtype=bool, count=n)
        else:
            in_a = np.zeros(n, dtype=bool)
            in_a[rng.permutation(n)[: n // 2]] = True
        if in_a.sum() < 2 or (~in_a).sum() < 2:
            continue
        mask = torch.from_numpy(in_a).to(real.device)
        values.append(mmd2_rbf(real[mask], real[~mask], clamp=False))
    if not values:
        return None
    arr = np.array(values)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return max(float(arr.mean()), 0.0), std, by_subject
