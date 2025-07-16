""" Test the bed module. """

import pytest
from Dosepy import bed
from pathlib import Path
import SimpleITK as sitk
import numpy as np


## Test load_dose 
#--------------------

# Test: TypeError if path_to_file is not str or Path
def test_load_type_error():
    with pytest.raises(TypeError):
        bed.load_dose(123)

# Test: FileNotFoundError if file does not exist
def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        bed.load_dose("/non/existent/file.dcm")

# Test: ValueError if file is not a valid DICOM
def test_load_invalid_dicom():
    with pytest.raises(ValueError):
        bed.load_dose(Path(__file__).parent / "fixtures" / "calibracion.png")

# Test: ValueError if DICOM does not contain required metadata tag
def test_load_missing_metadata():
    with pytest.raises(ValueError):
        bed.load_dose(Path(__file__).parent / "fixtures" / "CT_image.dcm")

# Test: Load a valid DICOM file
def test_load_valid_dicom():
    dose = bed.load_dose(Path(__file__).parent / "fixtures" / "RD_20x20cm2_256x256pix.dcm")
    assert isinstance(dose, sitk.Image)


## Test eqd2 function
#--------------------

# Test: 3D Dose of 8 Gy/1 fx is equivalent to 17.6 EQD2Gy, using alpha/beta = 3
def test_3D_8Gy():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    # Compute EQD2
    dose_eqd2 = bed.eqd2(dose8Gy, 3, 1)
    
    dose_eqd2_array = sitk.GetArrayFromImage(dose_eqd2)

    assert pytest.approx(np.mean(dose_eqd2_array), 0.1) == 17.6

# Test: Load an invalid dose object
def test_load_invalid_dose():
    with pytest.raises(ValueError):
        bed.eqd2("invalid_input", 3, 10)

# Test: Invalid number for alpha_beta parameter: 0
def test_invalid_number_alpha_beta_parameter():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    with pytest.raises(ValueError):
        bed.eqd2(dose8Gy, 0, 10)

# Test: Invalid type for alpha_beta parameter: string
def test_invalid_type_alpha_beta_parameter_string():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    with pytest.raises(ValueError):
        bed.eqd2(dose8Gy, "bad parameter", 10)

# Test: Invalid type for number_fractions parameter: string
def test_invalid_type_number_fractions_parameter_string():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    with pytest.raises(ValueError):
        bed.eqd2(dose8Gy, 3, "bad parameter")

# Test: Invalid type for number_fractions parameter: float
def test_invalid_type_number_fractions_parameter_float():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    with pytest.raises(ValueError):
        bed.eqd2(dose8Gy, 3, 10.5)


## Test get_structures
#---------------------

# Test: Get structures from a DICOM file
def test_get_structures_valid_file():
    structures = bed.get_structures_names(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
    structures == {"BODY": 1, "PTV_High": 2, "CouchSurface": 3, "CouchInterior": 4}

# Test: Check that the result has a correct shape
def test_get_structures_shape_ptv():
    structures = bed.get_structure_coordinates(
         "PTV_High",
        Path(__file__).parent / "fixtures" / "RS_anonymized.dcm"
    )
    assert structures[0].shape == (742, 3)

def test_get_structures_shape_body():
    structures = bed.get_structure_coordinates(
        "BODY",
        Path(__file__).parent / "fixtures" / "RS_anonymized.dcm",
    )
    assert len(structures) == 119

# Test: TypeError if path_to_file is not str or Path
def test_load_structures_type_error():
    with pytest.raises(TypeError):
        bed.get_structures_names(123)

# Test: FileNotFoundError if file does not exist
def test_get_structures_file_not_found():
    with pytest.raises(FileNotFoundError):
        bed.get_structures_names("/non/existent/file.dcm")

# Test: ValueError if DICOM is not a RT structure set
def test_get_structures_invalid_DICOM():
    with pytest.raises(ValueError):
        bed.get_structures_names(Path(__file__).parent / "fixtures" / "CT_image.dcm")

# Test: ValueError if file is not a valid DICOM
def test_load_invalid_dicom():
    with pytest.raises(ValueError):
        bed.get_structures_names(Path(__file__).parent / "fixtures" / "calibracion.png")