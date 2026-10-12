import numpy as np
import pytest
import torch

from m3score import stats


def _features(seed, n, d=16, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) + shift


def _naive_mmd2(x, y):
    """Float64 double-loop reference for the unbiased multi-bandwidth RBF MMD^2."""
    x = x.double().numpy()
    y = y.double().numpy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    joint = np.concatenate([x, y])
    d = ((joint[:, None, :] - joint[None, :, :]) ** 2).sum(-1)
    upper = np.sort(d[np.triu_indices(len(joint), k=1)])
    median = upper[(len(upper) - 1) // 2]  # lower median, as torch.median returns

    def k(a, b, sigma2):
        return np.exp(-((a - b) ** 2).sum() / (2.0 * sigma2))

    m, n = len(x), len(y)
    total = 0.0
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        s2 = median * scale
        rr = sum(k(x[i], x[j], s2) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
        gg = sum(k(y[i], y[j], s2) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
        rg = sum(k(x[i], y[j], s2) for i in range(m) for j in range(n)) / (m * n)
        total += rr + gg - 2.0 * rg
    return total / 5.0


@pytest.mark.parametrize("shift", [0.4, 2.0])
def test_mmd_matches_a_naive_reference(shift):
    # at shift 0.4 with this few points the unbiased estimate is negative, so both
    # the raw value and the clamped value are checked
    real, gen = _features(0, 12), _features(1, 9, shift=shift)
    expected = _naive_mmd2(real, gen)
    assert stats.mmd2_rbf(real, gen, clamp=False) == pytest.approx(expected, rel=1e-3, abs=1e-5)
    assert stats.mmd2_rbf(real, gen) == pytest.approx(max(expected, 0.0), rel=1e-3, abs=1e-5)


def test_mmd_of_a_set_with_itself_is_zero_when_clamped_and_negative_when_not():
    x = _features(2, 30)
    assert stats.mmd2_rbf(x, x) == 0.0
    assert stats.mmd2_rbf(x, x, clamp=False) < 0.0


def test_mmd_is_symmetric():
    real, gen = _features(3, 20), _features(4, 25, shift=0.3)
    assert stats.mmd2_rbf(real, gen) == pytest.approx(stats.mmd2_rbf(gen, real), abs=1e-6)


def test_mmd_grows_with_distribution_shift():
    real = _features(5, 80)
    near, far = _features(6, 80, shift=0.1), _features(6, 80, shift=1.0)
    assert stats.mmd2_rbf(real, far) > stats.mmd2_rbf(real, near)


def test_mmd_needs_two_samples_per_set():
    with pytest.raises(ValueError, match="at least 2"):
        stats.mmd2_rbf(_features(0, 1), _features(1, 5))


def _line(*values):
    return torch.tensor([[float(v), 0.0] for v in values])


def test_coverage_on_a_hand_built_line():
    real = _line(*range(10))  # every real point has its nearest neighbour at distance 1
    gen = _line(0.5)
    # reals at 0 and 1 are within 0.5 <= 1 of the generated point; real at 2 is 1.5 away
    assert stats.coverage(real, gen, k=1) == pytest.approx(0.2)
    assert stats.coverage(real, real, k=1) == 1.0
    assert stats.coverage(real, _line(1000.0), k=1) == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_coverage_never_rises_when_generated_samples_are_removed(seed):
    real, gen = _features(seed, 60, d=8), _features(seed + 100, 60, d=8, shift=0.2)
    values = [stats.coverage(real, gen[:n]) for n in (60, 40, 20, 5)]
    assert values == sorted(values, reverse=True)


def test_memorization_on_a_hand_built_line():
    real = _line(*range(10))  # median real-to-real nearest distance is 1
    gen = _line(0.2, 5.5, 100.0)  # nearest-real distances 0.2, 0.5, 91
    rate, nn_dist = stats.memorization(real, gen)
    assert rate == pytest.approx(2 / 3)
    assert nn_dist == pytest.approx((0.2 + 0.5 + 91.0) / 3, rel=1e-5)


def test_exact_copies_are_fully_memorized():
    real = _features(7, 40)
    rate, nn_dist = stats.memorization(real, real[:10].clone())
    assert rate == 1.0
    assert nn_dist == pytest.approx(0.0, abs=1e-3)


def test_precision_recall_uses_any_reference_ball():
    # k=1 radii: real 0 -> 0.1, real 0.1 -> 0.1, real 3 -> 2.9, real 8 -> 5.
    # The generated point at 1.0 is outside its nearest real ball (0.1, distance 0.9)
    # but inside the ball of the real point at 3 (distance 2.0 <= 2.9).
    real = _line(0.0, 0.1, 3.0, 8.0)
    gen = _line(1.0, 1.5)
    precision, recall = stats.precision_recall(real, gen, k=1)
    assert precision == 1.0
    assert recall == 0.0


def test_precision_recall_needs_more_than_k_points():
    with pytest.raises(ValueError, match="more than k"):
        stats.precision_recall(_features(0, 5), _features(1, 5), k=5)


def _slow_bootstrap_values(real, gen, n_bootstrap, seed):
    """Reference: build each resample explicitly and skip pairs sharing an original index."""
    nr, ng = real.shape[0], gen.shape[0]
    kernel = stats._pooled_kernel(real, gen).numpy()
    rng = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_bootstrap):
        ri = torch.randint(0, nr, (nr,), generator=rng).numpy()
        gi = torch.randint(0, ng, (ng,), generator=rng).numpy()
        rr = [kernel[a, b] for a in ri for b in ri if a != b]
        gg = [kernel[nr + a, nr + b] for a in gi for b in gi if a != b]
        rg = [kernel[a, nr + b] for a in ri for b in gi]
        out.append(sum(rr) / len(rr) + sum(gg) / len(gg) - 2.0 * sum(rg) / len(rg))
    return np.array(out)


def test_bootstrap_matches_a_slow_reference():
    real, gen = _features(1, 25), _features(2, 20, shift=0.3)
    values = _slow_bootstrap_values(real, gen, 8, seed=4)
    expected = (max(np.percentile(values, 2.5), 0.0), max(np.percentile(values, 97.5), 0.0))
    assert stats.bootstrap_ci(real, gen, n_bootstrap=8, seed=4) == pytest.approx(expected, abs=1e-9)


def test_bootstrap_interval_contains_the_point_estimate():
    g = torch.Generator().manual_seed(2)
    real = torch.randn(300, 64, generator=g)
    gen = torch.randn(300, 64, generator=g) + 0.15
    low, high = stats.bootstrap_ci(real, gen, seed=0)
    assert low <= stats.mmd2_rbf(real, gen) <= high


def test_bootstrap_interval_reaches_zero_for_one_distribution():
    g = torch.Generator().manual_seed(2)
    real = torch.randn(300, 64, generator=g)
    gen = torch.randn(300, 64, generator=g)
    low, _ = stats.bootstrap_ci(real, gen, seed=0)
    assert low == 0.0


def test_bootstrap_is_deterministic_per_seed():
    real, gen = _features(0, 50), _features(1, 50, shift=0.2)
    assert stats.bootstrap_ci(real, gen, seed=3) == stats.bootstrap_ci(real, gen, seed=3)
    assert stats.bootstrap_ci(real, gen, seed=3) != stats.bootstrap_ci(real, gen, seed=4)


def test_permutation_test_separates_shifted_from_identical_distributions():
    real = _features(0, 80, d=32)
    same = _features(1, 80, d=32)
    shifted = _features(1, 80, d=32, shift=0.5)
    p_same, _, _ = stats.permutation_test(real, same, n_permutations=100, seed=1)
    p_shift, z_shift, _ = stats.permutation_test(real, shifted, n_permutations=100, seed=1)
    assert p_same > 0.05
    assert p_shift == pytest.approx(1 / 101)
    assert z_shift > 5


def test_permutation_null_equals_recomputation():
    real, gen = _features(3, 30), _features(4, 25, shift=0.2)
    _, _, null_mean = stats.permutation_test(real, gen, n_permutations=20, seed=7)
    pooled = torch.cat([real, gen])
    rng = torch.Generator().manual_seed(7)
    slow = []
    for _ in range(20):
        perm = torch.randperm(55, generator=rng)
        slow.append(stats.mmd2_rbf(pooled[perm[:30]], pooled[perm[30:]]))
    assert null_mean == pytest.approx(float(np.mean(slow)), abs=1e-5)


def test_floor_is_small_for_one_distribution_and_deterministic():
    real = _features(0, 200, d=32)
    mean, std, disjoint = stats.real_floor(real, seed=0)
    assert not disjoint
    assert 0.0 <= mean < 0.01
    assert std >= 0.0
    assert stats.real_floor(real, seed=0) == (mean, std, disjoint)


def test_floor_splits_are_subject_disjoint(monkeypatch):
    # Column 0 carries the subject index so the two halves can be traced back.
    subjects = [f"s{i:02d}" for i in range(10) for _ in range(6)]
    real = _features(0, 60)
    real[:, 0] = torch.tensor([float(s[1:]) for s in subjects])
    seen = []
    original = stats.mmd2_rbf

    def spy(a, b, **kwargs):
        seen.append((set(a[:, 0].tolist()), set(b[:, 0].tolist())))
        return original(a, b, **kwargs)

    monkeypatch.setattr(stats, "mmd2_rbf", spy)
    _, _, disjoint = stats.real_floor(real, subjects, n_splits=5, seed=0)
    assert disjoint
    assert len(seen) == 5
    assert all(not (a & b) for a, b in seen)


def test_floor_falls_back_to_images_with_one_subject():
    result = stats.real_floor(_features(0, 40), ["only"] * 40, seed=0)
    assert result is not None and result[2] is False


def test_floor_is_none_without_enough_images():
    assert stats.real_floor(_features(0, 3)) is None


def test_statistics_stay_finite_on_degenerate_features():
    normal = _features(0, 40)
    constant = torch.ones(12, 16)
    for real, gen in ((normal, normal.clone()), (normal, constant), (constant, constant)):
        values = [
            stats.mmd2_rbf(real, gen),
            *stats.bootstrap_ci(real, gen, n_bootstrap=10),
            *stats.permutation_test(real, gen, n_permutations=10),
            *stats.memorization(real, gen),
            stats.coverage(real, gen),
        ]
        assert all(np.isfinite(v) for v in values), values
    assert stats.mmd2_rbf(normal, normal.clone()) == 0.0
    assert stats.coverage(normal, normal.clone()) == 1.0
