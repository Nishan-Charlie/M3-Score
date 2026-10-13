import json

import numpy as np
import pytest
import torch

import m3score
from _images import write_png, write_pngs
from m3score import M3Result, M3Score, M3ScoreWarning

pytestmark = pytest.mark.filterwarnings("ignore::m3score.M3ScoreWarning")


def _images(seed, n, size=32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, size, size, generator=g)


def _numbers(result):
    return (
        result.fidelity,
        result.memorization,
        result.coverage,
        result.memorization_nn_dist,
        *result.fidelity_ci,
        result.fidelity_p,
        result.fidelity_z,
        result.fidelity_floor,
        result.fidelity_floor_std,
    )


@pytest.fixture()
def metric(offline_backbone):
    return M3Score(device="cpu")


def test_streaming_equals_one_shot(offline_backbone):
    real, gen = _images(0, 40), _images(1, 30)
    streamed = M3Score(device="cpu", batch_size=7)
    for start in range(0, 40, 9):
        streamed.update(real[start : start + 9], real=True)
    streamed.update(gen, real=False)
    one_shot = M3Score(device="cpu").score(real, gen)
    assert _numbers(streamed.compute()) == pytest.approx(_numbers(one_shot), rel=1e-3, abs=1e-4)


def test_result_does_not_depend_on_batch_size(offline_backbone):
    real, gen = _images(0, 40), _images(1, 30)
    small = M3Score(device="cpu", batch_size=4).score(real, gen)
    large = M3Score(device="cpu", batch_size=64).score(real, gen)
    assert _numbers(small) == pytest.approx(_numbers(large), rel=1e-3, abs=1e-4)


def test_same_seed_same_result_and_the_seed_moves_only_the_statistics(offline_backbone):
    real, gen = _images(0, 40), _images(1, 30)
    a = M3Score(device="cpu", seed=3).score(real, gen)
    b = M3Score(device="cpu", seed=3).score(real, gen)
    c = M3Score(device="cpu", seed=4).score(real, gen)
    assert a == b
    assert a.fidelity == c.fidelity
    assert a.fidelity_ci != c.fidelity_ci


def test_reset_can_keep_the_real_features(metric):
    real, gen = _images(0, 40), _images(1, 30)
    metric.update(real, real=True)
    metric.update(gen, real=False)
    first = metric.compute()
    metric.reset(keep_real=True)
    metric.update(gen, real=False)
    assert metric.compute() == first
    metric.reset()
    with pytest.raises(ValueError, match="at least 6 real"):
        metric.compute()


def test_too_few_images_is_an_error(metric):
    metric.update(_images(0, 5), real=True)
    metric.update(_images(1, 5), real=False)
    with pytest.raises(ValueError, match="got 5 real and 5 generated"):
        metric.compute()


def test_statistics_can_be_skipped(metric):
    result = metric.score(_images(0, 20), _images(1, 20), stats=False)
    assert result.fidelity_ci is None and result.fidelity_p is None
    assert result.fidelity_z is None and result.fidelity_floor is None
    assert 0.0 <= result.coverage <= 1.0
    assert 0.0 <= result.memorization <= 1.0


def test_default_configuration_is_labelled_m3_v1(metric):
    result = metric.score(_images(0, 20), _images(1, 20), stats=False)
    assert result.protocol == "m3-v1"
    assert result.layers == (12, 9, 4)
    assert result.backbone == "tiny-test-vit"


def test_custom_backbone_requires_layers(tiny_model):
    with pytest.raises(ValueError, match="layers="):
        M3Score(device="cpu", backbone=tiny_model)


def test_custom_configuration_is_labelled_and_warns(tiny_model):
    metric = M3Score(device="cpu", backbone=tiny_model, layers=(12, 8, 3))
    with pytest.warns(M3ScoreWarning, match="not comparable"):
        result = metric.score(_images(0, 20), _images(1, 20), stats=False)
    assert result.protocol == "custom"
    assert result.layers == (12, 8, 3)
    assert any("not comparable" in w for w in result.warnings)


def test_layers_beyond_the_backbone_depth_are_rejected(tiny_model):
    metric = M3Score(device="cpu", backbone=tiny_model, layers=(13, 9, 4))
    with pytest.raises(ValueError, match="12 blocks"):
        metric.update(_images(0, 8), real=True)


def test_small_sets_are_flagged(metric):
    result = metric.score(_images(0, 20), _images(1, 20), stats=False)
    assert any("fewer than 500" in w for w in result.warnings)


def test_subjects_are_counted_and_low_diversity_is_flagged(metric):
    metric.update(_images(0, 20), real=True, groups=[f"s{i % 4}" for i in range(20)])
    metric.update(_images(1, 20), real=False)
    result = metric.compute()
    assert result.n_real_subjects == 4
    assert result.n_gen_subjects is None
    assert any("only 4 subjects" in w for w in result.warnings)


def test_partly_labelled_updates_leave_subjects_unknown(metric):
    metric.update(_images(0, 10), real=True, groups=["a"] * 10)
    metric.update(_images(2, 10), real=True)
    metric.update(_images(1, 20), real=False)
    assert metric.compute(stats=False).n_real_subjects is None


def test_group_labels_must_match_the_batch(metric):
    with pytest.raises(ValueError, match="group labels"):
        metric.update(_images(0, 10), real=True, groups=["a"] * 9)


def test_out_of_range_input_is_rejected(metric):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        metric.update(_images(0, 8) * 2 - 1, real=True)


def test_folders_are_scored_and_sampled(metric, tmp_path):
    write_pngs(tmp_path / "real", 30, seed=0)
    write_pngs(tmp_path / "gen", 24, seed=1)
    result = metric.score(tmp_path / "real", tmp_path / "gen", n=12, stats=False)
    assert (result.n_real, result.n_gen) == (12, 12)
    assert any("subjects unknown" in w for w in result.warnings)


def test_group_pattern_need_not_match_the_generated_set(metric, tmp_path):
    for subject in range(8):
        name = f"BraTS2021_{subject:05d}_slice{{:03d}}.png"
        write_pngs(tmp_path / "real", 3, seed=subject, name=name)
    write_pngs(tmp_path / "gen", 20, seed=99, name="generated_{:04d}.png")
    result = metric.score(
        tmp_path / "real", tmp_path / "gen", group_pattern=r"(BraTS2021_\d+)", stats=False
    )
    assert result.n_real_subjects == 8
    assert result.n_gen_subjects is None


def test_group_pattern_must_match_the_real_set(metric, tmp_path):
    write_pngs(tmp_path / "real", 10, name="scan_{:04d}.png")
    write_pngs(tmp_path / "gen", 10)
    with pytest.raises(ValueError, match="does not match"):
        metric.score(tmp_path / "real", tmp_path / "gen", group_pattern=r"(BraTS2021_\d+)")


def test_mixed_image_sizes_are_scored(metric, tmp_path):
    rng = np.random.default_rng(0)
    for i, size in enumerate([(240, 240), (256, 256), (64, 48), (17, 90)] * 3):
        pixels = rng.integers(0, 256, size=size, dtype=np.uint8)
        write_png(tmp_path / "real" / f"r{i:02d}.png", pixels)
    write_pngs(tmp_path / "gen", 10, size=(256, 256), seed=1)
    result = metric.score(tmp_path / "real", tmp_path / "gen", stats=False)
    assert (result.n_real, result.n_gen) == (12, 10)


def test_same_images_as_real_and_generated(metric):
    images = _images(0, 30)
    result = metric.score(images, images.clone())
    assert result.fidelity == 0.0
    assert result.coverage == 1.0
    assert result.memorization == 1.0
    assert all(np.isfinite(v) for v in _numbers(result))


def test_blank_generated_images_give_finite_values(metric):
    result = metric.score(_images(0, 30), torch.zeros(12, 1, 32, 32))
    assert all(np.isfinite(v) for v in _numbers(result))
    assert result.fidelity > 0.0


def test_unequal_set_sizes_and_oversized_n(metric):
    result = metric.score(_images(0, 8), _images(1, 60), n=1000)
    assert (result.n_real, result.n_gen) == (8, 60)
    swapped = metric.score(_images(1, 60), _images(0, 3), n=1000, stats=False)
    assert (swapped.n_real, swapped.n_gen) == (60, 3)


def test_cuda_request_without_cuda_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        M3Score(device="cuda")


def test_unreadable_files_are_reported_and_left_out(metric, tmp_path):
    write_pngs(tmp_path / "real", 10)
    (tmp_path / "real" / "broken.png").write_bytes(b"not a png")
    write_pngs(tmp_path / "gen", 10, seed=1)
    result = metric.score(tmp_path / "real", tmp_path / "gen", stats=False)
    assert result.n_real == 10
    assert any("broken.png" in w for w in result.warnings)


def test_result_serialises_and_prints(metric):
    result = metric.score(_images(0, 20), _images(1, 20))
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["protocol"] == "m3-v1"
    assert payload["layers"] == [12, 9, 4]
    assert isinstance(payload["fidelity_ci"], list)
    text = str(result)
    for word in ("fidelity", "memorization", "coverage", "floor", "permutation test", "warnings"):
        assert word in text
    assert text.isascii()


def test_module_level_score(offline_backbone):
    result = m3score.score(
        _images(0, 20), _images(1, 20), device="cpu", stats=False, progress=False
    )
    assert isinstance(result, M3Result)


def test_include_need_not_match_the_generated_set(metric, tmp_path):
    write_pngs(tmp_path / "real", 12, seed=0, name="scan_flair_{:04d}.png")
    write_pngs(tmp_path / "real", 5, seed=1, name="scan_seg_{:04d}.png")
    write_pngs(tmp_path / "gen", 10, seed=2, name="generated_{:04d}.png")
    result = metric.score(tmp_path / "real", tmp_path / "gen", include="*_flair_*", stats=False)
    assert (result.n_real, result.n_gen) == (12, 10)
    assert any("not applied" in note for note in result.warnings)


def test_include_must_match_the_real_set(metric, tmp_path):
    write_pngs(tmp_path / "real", 10)
    write_pngs(tmp_path / "gen", 10, seed=1)
    with pytest.raises(FileNotFoundError, match="matching"):
        metric.score(tmp_path / "real", tmp_path / "gen", include="*_flair_*")
