import sys

import numpy as np
import pytest
import torch

from m3score import M3ScoreError
from m3score.io import open_source
from m3score.io.dicom import _require_pydicom, is_dicom

pydicom = pytest.importorskip("pydicom")

from _dicom import write_dicom  # noqa: E402  (imports pydicom, so it must follow importorskip)


def _ramp(shape=(8, 8)):
    return np.linspace(0, 4095, shape[0] * shape[1]).reshape(shape).astype(np.uint16)


def _write_series(folder, n_slices, *, series_uid, patient_id="P1"):
    """Slice k is filled with 100 * (k + 1). Files are written in shuffled order."""
    order = np.random.default_rng(0).permutation(n_slices)
    for file_number, k in enumerate(order):
        write_dicom(
            folder / f"file{file_number:03d}.dcm",
            np.full((8, 8), 100 * (int(k) + 1), dtype=np.uint16),
            series_uid=series_uid,
            study_uid="1.2.826.0.1.1",
            patient_id=patient_id,
            position=2.5 * int(k),
            instance=file_number + 1,  # deliberately unrelated to position
        )


def test_a_single_image_is_scored_on_its_own(tmp_path):
    write_dicom(tmp_path / "cxr.dcm", _ramp())
    source = open_source(tmp_path)
    assert len(source) == 1
    assert source.groups == ["P1"]
    image = source.load(0)
    assert image.shape == (1, 8, 8)
    assert image.dtype == torch.float32
    assert float(image.min()) == 0.0 and float(image.max()) == 1.0


def test_rescale_and_the_stored_window_are_applied(tmp_path):
    pixels = np.array([[824, 1024], [1224, 3000]], dtype=np.uint16)  # HU -200, 0, 200, 1976
    write_dicom(tmp_path / "ct.dcm", pixels, intercept=-1024.0, window=(0.0, 400.0))
    image = open_source(tmp_path).load(0)
    assert image.flatten().tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])


def test_an_explicit_window_overrides_the_stored_one(tmp_path):
    pixels = np.array([[24, 1024], [1424, 3000]], dtype=np.uint16)  # HU -1000, 0, 400, 1976
    write_dicom(tmp_path / "ct.dcm", pixels, intercept=-1024.0, window=(0.0, 400.0))
    image = open_source(tmp_path, window=(-1000.0, 400.0)).load(0)
    assert image.flatten().tolist() == pytest.approx([0.0, 1000 / 1400, 1.0, 1.0])


def test_monochrome1_is_inverted(tmp_path):
    window = (2048.0, 4096.0)
    write_dicom(tmp_path / "a" / "x.dcm", _ramp(), window=window)
    write_dicom(tmp_path / "b" / "x.dcm", _ramp(), window=window, photometric="MONOCHROME1")
    normal = open_source(tmp_path / "a").load(0)
    inverted = open_source(tmp_path / "b").load(0)
    assert torch.allclose(normal + inverted, torch.ones_like(normal))


def test_a_series_is_a_volume_ordered_by_position_with_the_band_applied(tmp_path):
    _write_series(tmp_path, 20, series_uid="1.2.3")
    source = open_source(tmp_path, window=(0.0, 4000.0))
    assert len(source) == 4  # the default band of a 20-slice volume is slices 7 to 10
    values = [round(float(source.load(i).mean()) * 4000.0) for i in range(4)]
    assert values == [800, 900, 1000, 1100]
    assert source.groups == ["P1"] * 4


def test_series_are_grouped_by_patient(tmp_path):
    _write_series(tmp_path / "p1", 20, series_uid="1.2.3", patient_id="P1")
    _write_series(tmp_path / "p2", 20, series_uid="1.2.4", patient_id="P2")
    source = open_source(tmp_path)
    assert len(source) == 8
    assert sorted(set(source.groups)) == ["P1", "P2"]


def test_subject_falls_back_to_the_study_without_a_patient_id(tmp_path):
    write_dicom(tmp_path / "a.dcm", _ramp(), patient_id=None, study_uid="1.2.826.0.1")
    assert open_source(tmp_path).groups == ["1.2.826.0.1"]


def test_files_named_by_uid_are_found_by_their_magic_bytes(tmp_path):
    path = write_dicom(tmp_path / "1.2.840.113619.2.55", _ramp())
    (tmp_path / "README").write_text("not dicom")
    assert is_dicom(path)
    assert not is_dicom(tmp_path / "README")
    assert len(open_source(tmp_path)) == 1


def test_8bit_rgb_is_read_as_colour(tmp_path):
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    write_dicom(tmp_path / "us.dcm", pixels, photometric="RGB")
    image = open_source(tmp_path).load(0)
    assert image.shape == (3, 4, 4)
    assert float(image[0].min()) == 1.0
    assert float(image[1].max()) == 0.0


def test_other_colour_encodings_are_skipped_with_a_warning(tmp_path):
    write_dicom(tmp_path / "ybr.dcm", np.zeros((4, 4, 3), dtype=np.uint8), photometric="YBR_FULL")
    write_dicom(tmp_path / "ok.dcm", _ramp())
    source = open_source(tmp_path)
    assert len(source) == 1
    assert any("ybr.dcm" in note and "YBR_FULL" in note for note in source.warnings)


def test_a_multi_frame_file_is_a_volume(tmp_path):
    frames = np.stack([np.full((8, 8), 100 * (k + 1), dtype=np.uint16) for k in range(20)])
    write_dicom(tmp_path / "enhanced.dcm", frames, frames=20)
    source = open_source(tmp_path, window=(0.0, 4000.0))
    assert len(source) == 4
    assert round(float(source.load(0).mean()) * 4000.0) == 800


def test_undecodable_pixel_data_names_the_decoder_packages(tmp_path, monkeypatch):
    write_dicom(tmp_path / "a.dcm", _ramp())

    def no_handler(self):
        raise RuntimeError("no available image handler")

    monkeypatch.setattr(pydicom.dataset.Dataset, "pixel_array", property(no_handler))
    with pytest.raises(M3ScoreError, match="pylibjpeg"):
        open_source(tmp_path).load(0)


def test_missing_pydicom_explains_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pydicom", None)
    with pytest.raises(ImportError, match=r"m3-score\[dicom\]"):
        _require_pydicom()


def test_dicom_over_the_voxel_limit_is_skipped_with_a_warning(tmp_path, monkeypatch):
    from m3score.io import volumes

    _write_series(tmp_path / "series", 20, series_uid="1.2.3")  # 20 x 8 x 8 = 1280 voxels
    write_dicom(tmp_path / "wide.dcm", _ramp((16, 16)))  # 256 pixels
    write_dicom(tmp_path / "small.dcm", _ramp((4, 4)))  # 16 pixels
    monkeypatch.setattr(volumes, "MAX_VOXELS", 100)
    source = open_source(tmp_path)
    assert len(source) == 1
    notes = " | ".join(source.warnings)
    assert "too large" in notes
    assert "wide.dcm" in notes and "series/" in notes
