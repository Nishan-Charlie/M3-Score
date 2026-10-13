import numpy as np
import pytest
import torch

from m3score.io.arrays import ArraySource, check_images, to_float


def test_uint8_greyscale_stack_becomes_unit_range_with_a_channel():
    images = torch.tensor([[[0, 255], [51, 102]]], dtype=torch.uint8)
    tensor, divisor = check_images(images)
    out = to_float(tensor, divisor)
    assert out.shape == (1, 1, 2, 2)
    assert out.dtype == torch.float32
    assert out.flatten().tolist() == pytest.approx([0.0, 1.0, 0.2, 0.4])


def test_float_rgb_passes_through_unchanged():
    images = torch.rand(3, 3, 8, 8)
    tensor, divisor = check_images(images)
    assert divisor == 1.0
    assert torch.equal(to_float(tensor, divisor), images)


def test_numpy_arrays_are_accepted():
    tensor, divisor = check_images(np.zeros((2, 8, 8), dtype=np.uint8))
    assert tensor.shape == (2, 1, 8, 8)
    assert divisor == 255.0


def test_values_outside_unit_range_are_rejected_with_the_fix():
    with pytest.raises(ValueError, match=r"\(x \+ 1\) / 2"):
        check_images(torch.rand(2, 1, 8, 8) * 2 - 1)


def test_non_finite_values_are_rejected():
    images = torch.rand(2, 1, 8, 8)
    images[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        check_images(images)


@pytest.mark.parametrize("shape", [(8, 8), (2, 2, 8, 8), (2, 8, 8, 3), (1, 2, 3, 8, 8)])
def test_wrong_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="shaped"):
        check_images(torch.zeros(shape))


def test_unsupported_dtypes_are_rejected():
    with pytest.raises(ValueError, match="dtype"):
        check_images(torch.zeros(2, 1, 8, 8, dtype=torch.int64))


def test_non_arrays_are_rejected():
    with pytest.raises(TypeError, match="torch.Tensor or numpy.ndarray"):
        check_images([[0.0]])


def test_empty_batches_are_rejected():
    with pytest.raises(ValueError, match="no images"):
        check_images(torch.zeros(0, 1, 8, 8))


def test_array_source_serves_single_images():
    source = ArraySource(np.full((4, 6, 5), 255, dtype=np.uint8))
    assert len(source) == 4
    assert source.groups is None
    assert source.warnings == []
    image = source.load(2)
    assert image.shape == (1, 6, 5)
    assert float(image.max()) == 1.0
