"""Helper that writes small uncompressed DICOM files for tests."""

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def _rescale_group(slope, intercept):
    """A functional group holding a rescale, as enhanced multi-frame objects store it."""
    transform = Dataset()
    transform.RescaleSlope = slope
    transform.RescaleIntercept = intercept
    group = Dataset()
    group.PixelValueTransformationSequence = Sequence([transform])
    return group


def write_dicom(
    path,
    pixels,
    *,
    orientation=None,
    shared_rescale=None,
    per_frame_rescale=None,
    series_uid=None,
    study_uid=None,
    patient_id="P1",
    position=None,
    instance=1,
    slope=1.0,
    intercept=0.0,
    window=None,
    photometric="MONOCHROME2",
    frames=None,
):
    """Write one DICOM file.

    ``pixels`` is ``(H, W)`` 16-bit, ``(F, H, W)`` 16-bit together with ``frames=F``,
    or ``(H, W, 3)`` uint8 for a colour image. ``shared_rescale`` and
    ``per_frame_rescale`` store the rescale in functional groups and leave the
    top-level rescale tags out, as enhanced multi-frame objects do.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    if patient_id is not None:
        ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.InstanceNumber = instance
    ds.PhotometricInterpretation = photometric

    colour = pixels.ndim == 3 and frames is None
    if colour:
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 0
        ds.Rows, ds.Columns = pixels.shape[:2]
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
        ds.PixelRepresentation = 0
    else:
        ds.SamplesPerPixel = 1
        ds.Rows, ds.Columns = pixels.shape[-2:]
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 1 if pixels.dtype == np.int16 else 0
        if shared_rescale is None and per_frame_rescale is None:
            ds.RescaleSlope = slope
            ds.RescaleIntercept = intercept
    if shared_rescale is not None:
        ds.SharedFunctionalGroupsSequence = Sequence([_rescale_group(*shared_rescale)])
    if per_frame_rescale is not None:
        ds.PerFrameFunctionalGroupsSequence = Sequence(
            [_rescale_group(slope_k, intercept_k) for slope_k, intercept_k in per_frame_rescale]
        )
    if frames is not None:
        ds.NumberOfFrames = frames
    if position is not None:
        ds.ImagePositionPatient = [0.0, 0.0, float(position)]
        ds.ImageOrientationPatient = list(orientation or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = np.ascontiguousarray(pixels).tobytes()

    try:
        ds.save_as(str(path), enforce_file_format=True)  # pydicom 3
    except TypeError:  # pydicom 2
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(str(path), write_like_original=False)
    return path
