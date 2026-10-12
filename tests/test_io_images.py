import numpy as np
import pytest
import torch

from _images import write_png, write_pngs
from m3score.io import ArraySource, ImageSource, SkippedItem, open_source


def test_8bit_greyscale_png_loads_as_three_equal_channels(tmp_path):
    write_png(tmp_path / "a.png", np.array([[0, 255], [51, 102]], dtype=np.uint8))
    image = open_source(tmp_path).load(0)
    assert image.shape == (3, 2, 2)
    assert image.dtype == torch.float32
    assert image[0].flatten().tolist() == pytest.approx([0.0, 1.0, 0.2, 0.4])
    assert torch.equal(image[0], image[2])


def test_16bit_png_is_scaled_by_its_full_range(tmp_path):
    write_png(tmp_path / "a.png", np.array([[0, 65535], [13107, 32768]], dtype=np.uint16))
    image = open_source(tmp_path).load(0)
    assert image.shape == (1, 2, 2)
    assert image.flatten().tolist() == pytest.approx([0.0, 1.0, 0.2, 32768 / 65535], abs=1e-4)


def test_rgb_png_keeps_its_channels(tmp_path):
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    write_png(tmp_path / "a.png", pixels)
    image = open_source(tmp_path).load(0)
    assert float(image[0].min()) == 1.0
    assert float(image[1].max()) == 0.0


def test_folders_are_scanned_recursively_and_other_files_ignored(tmp_path):
    write_pngs(tmp_path / "b", 2)
    write_pngs(tmp_path / "a", 1)
    write_png(tmp_path / "UPPER.PNG", np.zeros((4, 4), dtype=np.uint8))
    (tmp_path / "notes.txt").write_text("not an image")
    source = open_source(tmp_path)
    assert len(source) == 4
    assert source.groups is None
    assert source.warnings == []


def test_include_restricts_the_scan(tmp_path):
    write_pngs(tmp_path, 3, name="flair_{:04d}.png")
    write_pngs(tmp_path, 2, name="seg_{:04d}.png")
    assert len(open_source(tmp_path, include="flair_*")) == 3


def test_group_pattern_assigns_subjects(tmp_path):
    for subject in ("BraTS2021_00000", "BraTS2021_00002"):
        write_pngs(tmp_path, 3, name=subject + "_slice{:03d}.png")
    source = open_source(tmp_path, group_pattern=r"(BraTS2021_\d+)")
    assert source.groups == ["BraTS2021_00000"] * 3 + ["BraTS2021_00002"] * 3


def test_group_pattern_that_misses_a_file_is_an_error(tmp_path):
    write_pngs(tmp_path, 2, name="BraTS2021_00000_slice{:03d}.png")
    write_pngs(tmp_path, 1, name="other_{:04d}.png")
    with pytest.raises(ValueError, match="does not match"):
        open_source(tmp_path, group_pattern=r"(BraTS2021_\d+)")


def test_group_pattern_needs_a_capture_group(tmp_path):
    write_pngs(tmp_path, 1)
    with pytest.raises(ValueError, match="capture group"):
        open_source(tmp_path, group_pattern=r"img_\d+")


def test_invalid_group_pattern_is_a_value_error(tmp_path):
    write_pngs(tmp_path, 1)
    with pytest.raises(ValueError, match="not a valid regular expression"):
        open_source(tmp_path, group_pattern="(")


def test_optional_group_pattern_that_matches_nothing_leaves_groups_unknown(tmp_path):
    write_pngs(tmp_path, 3, name="generated_{:04d}.png")
    source = open_source(tmp_path, group_pattern=r"(BraTS2021_\d+)", group_pattern_optional=True)
    assert source.groups is None


def test_optional_group_pattern_still_rejects_a_partial_match(tmp_path):
    write_pngs(tmp_path, 2, name="BraTS2021_00000_slice{:03d}.png")
    write_pngs(tmp_path, 1, name="other_{:04d}.png")
    with pytest.raises(ValueError, match="does not match"):
        open_source(tmp_path, group_pattern=r"(BraTS2021_\d+)", group_pattern_optional=True)


def test_a_folder_without_images_lists_the_supported_types(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(FileNotFoundError, match=r"\.png"):
        open_source(tmp_path)


def test_a_missing_path_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file or directory"):
        open_source(tmp_path / "absent")


def test_an_unreadable_image_is_skipped_with_its_name(tmp_path):
    (tmp_path / "broken.png").write_bytes(b"this is not a png")
    with pytest.raises(SkippedItem, match="broken.png"):
        open_source(tmp_path).load(0)


def test_a_single_file_is_a_source(tmp_path):
    path = write_pngs(tmp_path, 1)[0]
    assert len(open_source(path)) == 1


def test_tensors_arrays_and_sources_pass_through():
    assert isinstance(open_source(torch.zeros(2, 1, 4, 4)), ArraySource)
    assert isinstance(open_source(np.zeros((2, 4, 4), dtype=np.uint8)), ArraySource)
    source = ArraySource(torch.zeros(2, 1, 4, 4))
    assert open_source(source) is source
    assert isinstance(source, ImageSource)


def test_other_types_are_rejected():
    with pytest.raises(TypeError, match="path"):
        open_source(42)
