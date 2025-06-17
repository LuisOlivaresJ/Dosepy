import pytest
from Dosepy import bed
from pathlib import Path
import SimpleITK as sitk

# Test the bed module

# Test: TypeError if path_to_file is not str or Path
def test_load_type_error():
    with pytest.raises(TypeError):
        bed.load(123)

# Test: FileNotFoundError if file does not exist
def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        bed.load("/non/existent/file.dcm")

# Test: ValueError if file is not a valid DICOM
def test_load_invalid_dicom():
    with pytest.raises(ValueError):
        bed.load(Path(__file__).parent / "fixtures" / "calibracion.png")

# Test: ValueError if DICOM does not contain required metadata tag
def test_load_missing_metadata():
    with pytest.raises(ValueError):
        bed.load(Path(__file__).parent / "fixtures" / "CT_image.dcm")

# Test: Load a valid DICOM file
def test_load_valid_dicom():
    dose = bed.load(Path(__file__).parent / "fixtures" / "RD_20x20cm2_256x256pix.dcm")
    assert isinstance(dose, sitk.Image)