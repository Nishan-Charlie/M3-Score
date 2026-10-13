import sys

import numpy as np
import pytest
import torch

from m3score import M3Score
from m3score.io import SkippedItem, open_source
from m3score.io.nifti import _require_nibabel, is_nifti

nib = pytest.importorskip("nibabel")

SHAPE = (6, 7, 20)  # left-right, posterior-anterior, inferior-superior
BAND = range(7, 11)  # the default band for a depth of 20
WINDOW = (0.0, 30000.0)  # wider than any test volume, so scaling is a plain division


def _ras_volume(seed=0):
    """Values rise towards the patient's right, the anterior and the top, over a random base."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*(np.arange(n) for n in SHAPE), indexing="ij")
    return (100.0 + 10.0 * x + 100.0 * y + 1000.0 * z + rng.random(SHAPE)).astype(np.float32)


def _save(path, data, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), str(path))
    return path


def test_nifti_suffixes_are_recognised_in_any_case(tmp_path):
    assert is_nifti(tmp_path / "a.nii")
    assert is_nifti(tmp_path / "a.nii.gz")
    assert is_nifti(tmp_path / "A.NII.GZ")
    assert not is_nifti(tmp_path / "a.png")


def test_a_volume_yields_its_central_band(tmp_path):
    _save(tmp_path / "subject1.nii.gz", _ras_volume())
    source = open_source(tmp_path)
    assert len(source) == len(BAND)
    assert source.groups == ["subject1.nii.gz"] * len(BAND)
    image = source.load(0)
    assert image.shape == (1, SHAPE[1], SHAPE[0])
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


def test_the_first_item_is_the_first_slice_of_the_band(tmp_path):
    data = _ras_volume()
    _save(tmp_path / "v.nii.gz", data)
    image = open_source(tmp_path, window=WINDOW).load(0)[0]
    expected = data[:, :, BAND.start][::-1, ::-1].T / WINDOW[1]
    assert np.allclose(image.numpy(), expected, atol=1e-6)


def test_slices_are_shown_anterior_up_with_the_patient_right_on_the_left(tmp_path):
    _save(tmp_path / "v.nii.gz", _ras_volume())
    image = open_source(tmp_path, window=WINDOW).load(0)[0]
    assert image[0, 0] > image[-1, 0]  # the top row is more anterior than the bottom row
    assert image[0, 0] > image[0, -1]  # the left column is further to the patient's right


def test_stored_orientation_does_not_change_the_slices(tmp_path):
    data = _ras_volume()
    _save(tmp_path / "ras" / "v.nii.gz", data)
    # the same anatomy stored left-posterior-superior, as BraTS files are
    lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    _save(tmp_path / "lps" / "v.nii.gz", data[::-1, ::-1, :].copy(), lps)
    # and with the array axes stored in the order superior, right, anterior
    sra = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    _save(tmp_path / "sra" / "v.nii.gz", np.transpose(data, (2, 0, 1)).copy(), sra)

    reference = open_source(tmp_path / "ras", window=WINDOW)
    for name in ("lps", "sra"):
        other = open_source(tmp_path / name, window=WINDOW)
        assert len(other) == len(reference) == len(BAND)
        for index in range(len(reference)):
            assert torch.allclose(other.load(index), reference.load(index), atol=1e-6)


def test_each_file_is_a_subject_and_include_filters_them(tmp_path):
    for name in ("a_flair.nii.gz", "b_flair.nii.gz", "a_seg.nii.gz"):
        _save(tmp_path / name, _ras_volume())
    assert len(set(open_source(tmp_path).groups)) == 3
    flair = open_source(tmp_path, include="*_flair.nii.gz")
    assert sorted(set(flair.groups)) == ["a_flair.nii.gz", "b_flair.nii.gz"]


def test_group_pattern_overrides_the_file_as_subject(tmp_path):
    for name in ("P01_t1.nii.gz", "P01_t2.nii.gz", "P02_t1.nii.gz"):
        _save(tmp_path / name, _ras_volume())
    source = open_source(tmp_path, group_pattern=r"(P\d+)")
    assert sorted(set(source.groups)) == ["P01", "P02"]


def test_the_slice_band_can_be_overridden(tmp_path):
    _save(tmp_path / "v.nii.gz", _ras_volume())
    assert len(open_source(tmp_path, slice_band=(0.0, 1.0))) == SHAPE[2]


def test_a_4d_file_is_an_error_naming_the_file(tmp_path):
    _save(tmp_path / "bold.nii.gz", np.zeros((4, 4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match=r"bold\.nii\.gz.*3D"):
        open_source(tmp_path)


def test_a_trailing_singleton_dimension_is_accepted(tmp_path):
    _save(tmp_path / "v.nii.gz", _ras_volume()[..., None])
    assert open_source(tmp_path).load(0).shape == (1, SHAPE[1], SHAPE[0])


def test_a_constant_volume_is_skipped_with_its_name(tmp_path):
    _save(tmp_path / "blank.nii.gz", np.zeros(SHAPE, dtype=np.float32))
    with pytest.raises(SkippedItem, match="blank.nii.gz"):
        open_source(tmp_path).load(0)


def test_missing_nibabel_explains_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "nibabel", None)
    with pytest.raises(ImportError, match=r"m3-score\[nifti\]"):
        _require_nibabel()


@pytest.mark.filterwarnings("ignore::m3score.M3ScoreWarning")
def test_volumes_are_scored_end_to_end(offline_backbone, tmp_path):
    for i in range(3):
        _save(tmp_path / "real" / f"r{i}.nii.gz", _ras_volume(seed=i))
        _save(tmp_path / "gen" / f"g{i}.nii.gz", _ras_volume(seed=10 + i))
    _save(tmp_path / "real" / "blank.nii.gz", np.zeros(SHAPE, dtype=np.float32))
    result = M3Score(device="cpu").score(tmp_path / "real", tmp_path / "gen", stats=False)
    assert result.n_real == 3 * len(BAND)
    assert result.n_real_subjects == 3
    assert result.n_gen_subjects == 3
    assert any("blank.nii.gz" in note for note in result.warnings)


def test_a_volume_over_the_voxel_limit_is_left_out_with_a_note(tmp_path, monkeypatch):
    from m3score.io import volumes

    _save(tmp_path / "giant.nii.gz", _ras_volume())  # 840 voxels
    _save(tmp_path / "small.nii.gz", _ras_volume()[:2, :2, :].copy())  # 80 voxels
    monkeypatch.setattr(volumes, "MAX_VOXELS", 100)
    source = open_source(tmp_path)
    assert set(source.groups) == {"small.nii.gz"}
    assert any("giant.nii.gz" in note and "too large" in note for note in source.warnings)


def test_a_truncated_volume_is_skipped_with_its_name(tmp_path):
    noise = np.random.default_rng(0).random((40, 40, 40)).astype(np.float32)
    path = _save(tmp_path / "cut.nii.gz", noise)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    source = open_source(tmp_path)  # the header is intact, so the file is listed
    with pytest.raises(SkippedItem, match="cut.nii.gz"):
        source.load(0)


def test_a_file_that_is_not_nifti_is_left_out_with_a_note(tmp_path):
    _save(tmp_path / "good.nii.gz", _ras_volume())
    (tmp_path / "bad.nii.gz").write_bytes(b"this is not a nifti file")
    source = open_source(tmp_path)
    assert len(source) == len(BAND)
    assert any("bad.nii.gz" in note for note in source.warnings)


def test_several_volumes_in_one_subject_folder_are_flagged(tmp_path):
    for subject in ("S1", "S2"):
        for kind in ("t1", "flair", "seg"):
            _save(tmp_path / subject / f"{subject}_{kind}.nii.gz", _ras_volume())
    assert any("several volumes" in note for note in open_source(tmp_path).warnings)
    filtered = open_source(tmp_path, include="*_flair.nii.gz")
    assert not any("several volumes" in note for note in filtered.warnings)
    named = open_source(tmp_path, group_pattern=r"(S\d)")
    assert not any("several volumes" in note for note in named.warnings)


def test_one_volume_per_subject_is_not_flagged(tmp_path):
    for name in ("a", "b", "c"):
        _save(tmp_path / f"{name}.nii.gz", _ras_volume())
    assert open_source(tmp_path).warnings == []


def test_a_file_overwritten_in_place_is_read_again(tmp_path):
    _save(tmp_path / "v.nii.gz", _ras_volume(seed=0))
    first = open_source(tmp_path, window=WINDOW).load(0)
    _save(tmp_path / "v.nii.gz", _ras_volume(seed=0) * 0.5)
    second = open_source(tmp_path, window=WINDOW).load(0)
    assert not torch.allclose(first, second)
