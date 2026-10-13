import numpy as np
import pytest

from m3score import protocol
from m3score.io.volumes import band_indices, check_band, check_window, scale_intensity


def test_default_band_is_slices_55_to_85_of_155():
    assert band_indices(155, protocol.SLICE_BAND) == range(55, 86)


def test_band_scales_with_depth_and_is_never_empty():
    assert band_indices(10, (0.0, 1.0)) == range(0, 10)
    assert band_indices(64, protocol.SLICE_BAND) == range(23, 36)
    assert len(band_indices(1, protocol.SLICE_BAND)) == 1
    assert len(band_indices(3, protocol.SLICE_BAND)) == 1


@pytest.mark.parametrize("band", [(0.5, 0.5), (0.6, 0.4), (-0.1, 0.5), (0.2, 1.1)])
def test_invalid_bands_are_rejected(band):
    with pytest.raises(ValueError, match="slice band"):
        check_band(band)


def test_invalid_window_is_rejected():
    with pytest.raises(ValueError, match="intensity window"):
        check_window((400, -1000))


def test_percentile_scaling_ignores_background_zeros():
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[2:8, 2:8, 2:8] = np.linspace(100, 200, 216).reshape(6, 6, 6)
    scaled = scale_intensity(volume, None)
    assert scaled.dtype == np.float32
    assert scaled.min() == 0.0 and scaled.max() == 1.0
    assert (scaled[volume == 0] == 0.0).all()
    assert 0.45 < scaled[volume > 0].mean() < 0.55


def test_an_explicit_window_is_applied_exactly():
    volume = np.array([[[-1000.0, -300.0, 400.0, 2000.0]]], dtype=np.float32)
    scaled = scale_intensity(volume, (-1000.0, 400.0))
    assert scaled.flatten().tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])


def test_empty_or_constant_volumes_cannot_be_scaled():
    assert scale_intensity(np.zeros((4, 4, 4), dtype=np.float32), None) is None
    assert scale_intensity(np.full((4, 4, 4), 7.0, dtype=np.float32), None) is None


def test_a_constant_volume_cannot_be_scaled_even_with_a_window():
    assert scale_intensity(np.zeros((4, 4, 4), dtype=np.float32), (-1000.0, 400.0)) is None
    assert scale_intensity(np.full((4, 4, 4), 7.0, dtype=np.float32), (0.0, 10.0)) is None
