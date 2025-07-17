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

# Test: Check raise ValueError if structure not found
def test_get_structure_coordinates_not_found():
    with pytest.raises(ValueError):
        bed.get_structure_coordinates(
            "NonExistentStructure",
            Path(__file__).parent / "fixtures" / "RS_anonymized.dcm"
        )

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


## Test _get_dose_plane_by_coordinate
#------------------------------------
# Test: Get a dose plane by z coordinate that is not in the dose distribution (interpolation)
def test_get_dose_plane_by_coordinate():
    dose_distribution = bed.load_dose(Path(__file__).parent / "fixtures" / "/home/luis/GH/Dosepy/tests/unit/fixtures/RTDose_3D.dcm")
    z_coordinate = 0.0  # Example z coordinate
    dose_plane = bed._get_dose_plane_by_coordinate(dose_distribution, z_coordinate)
    
    assert pytest.approx(dose_plane[50, 50, 0], abs = 0.1) == 13.4


# Test _get_dose_plane_by_coordinate with invalid z coordinate
def test_get_dose_plane_by_coordinate_invalid_z():
    dose_distribution = bed.load_dose(Path(__file__).parent / "fixtures" / "/home/luis/GH/Dosepy/tests/unit/fixtures/RTDose_3D.dcm")
    z_coordinate = 100.0  # Example z coordinate outside the range of the dose distribution
    
    with pytest.raises(ValueError):
        bed._get_dose_plane_by_coordinate(dose_distribution, z_coordinate)

# Test: Get a dose plane by z coordinate that is in the dose distribution (not interpolation needed)
def test_get_dose_plane_by_coordinate_valid_z():
    dose_distribution = bed.load_dose(Path(__file__).parent / "fixtures" / "/home/luis/GH/Dosepy/tests/unit/fixtures/RTDose_3D.dcm")
    z_coordinate = 1  # Example z coordinate that is in the dose distribution
    dose_plane = bed._get_dose_plane_by_coordinate(dose_distribution, z_coordinate)
    
    assert pytest.approx(dose_plane[50, 50, 0], abs = 0.1) == 13.2

# Test: Give a z_coordinate that is not a number
def test_get_dose_plane_by_coordinate_invalid_z_type():
    dose_distribution = bed.load_dose(Path(__file__).parent / "fixtures" / "/home/luis/GH/Dosepy/tests/unit/fixtures/RTDose_3D.dcm")
    
    with pytest.raises(ValueError):
        bed._get_dose_plane_by_coordinate(dose_distribution, "invalid_z")


## Test get_2D_mask_by_coordinates_and_image_shape()
#-------------------------------------------------
# Test: shape and mean of the mask generated by coordinates and image shape
def test_get_2D_mask_by_coordinates_and_image_shape():
    # Load a dose distribution
    dose = bed.load_dose(Path(__file__).parent / "fixtures" / "RTDose_3D.dcm")
    # Get the coordinates of a structure
    coordinates = bed.get_structure_coordinates(
        "PTV_High",
        Path(__file__).parent / "fixtures" / "RS_anonymized.dcm"
    )[0]
    # Get the mask by coordinates and image shape
    mask = bed.get_2D_mask_by_coordinates_and_image_shape(coordinates, dose)

    assert mask.shape == (90, 102)
    assert pytest.approx(np.mean(mask), abs=0.1) == 0.7
    
# Test: TypeError if coordinates is not a numpy array
def test_get_2D_mask_by_coordinates_and_image_shape_type_error_coordinates():
    dose = bed.load_dose(Path(__file__).parent / "fixtures" / "RTDose_3D.dcm")
    
    with pytest.raises(TypeError):
        bed.get_2D_mask_by_coordinates_and_image_shape("invalid_coordinates", dose)

# Test: TypeError if image is not a SimpleITK image
def test_get_2D_mask_by_coordinates_and_image_shape_type_error_image():
    coordinates = np.array([[0, 0, 0], [1, 1, 1]])
    
    with pytest.raises(TypeError):
        bed.get_2D_mask_by_coordinates_and_image_shape(coordinates, "invalid_image")