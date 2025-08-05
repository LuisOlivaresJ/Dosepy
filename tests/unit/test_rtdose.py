""" Test the bed module. """

from turtle import rt
import pytest
from Dosepy import rtdose
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pydicom


## Test StructureSet class
#-------------------------

## Test load_structures
# Test: Correct type ourput
def test_load_structures_ourput_type():
    structures = rtdose.load_structures(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
    assert isinstance(structures, rtdose.StructureSet)

# Test: TypeError if path_to_file is not str or Path
def test_load_structures_type_error():
    with pytest.raises(TypeError):
        rtdose.load_structures(123)

# Test: FileNotFoundError if file does not exist
def test_get_structures_file_not_found():
    with pytest.raises(FileNotFoundError):
        rtdose.load_structures("/non/existent/file.dcm")

# Test: ValueError if DICOM is not a RT structure set
def test_get_structures_invalid_DICOM():
    with pytest.raises(ValueError):
        rtdose.load_structures(Path(__file__).parent / "fixtures" / "CT_image.dcm")

# Test: ValueError if file is not a valid DICOM
def test_load_invalid_dicom():
    with pytest.raises(ValueError):
        rtdose.load_structures(Path(__file__).parent / "fixtures" / "calibracion.png")

## Test get_structures
#---------------------

# Test: Get structures from a DICOM file
def test_get_structures_valid_file():
    structures = rtdose.load_structures(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
    #structures = rtdose.get_structures_names(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
    structures.get_names == {"BODY": 1, "PTV_High": 2, "CouchSurface": 3, "CouchInterior": 4}

# Test: Check that the result has a correct shape
def test_get_structure_shape_ptv():
    ds = rtdose.load_structures(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
    structures = ds.get_coordinates("PTV_High")
    assert structures[0].shape == (742, 3)

def test_get_structures_shape_body():
    ds = rtdose.load_structures(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
    structures = ds.get_coordinates("BODY")
    assert len(structures) == 119

# Test: Attempt to get coordinates for a non-existent structure
def test_get_structure_coordinates_not_found():
    ds = rtdose.load_structures(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
   
    with pytest.raises(ValueError):
        ds.get_coordinates("NonExistentStructure")




## Test load_dose 
#--------------------

# Test: TypeError if path_to_file is not str or Path
def test_load_type_error():
    with pytest.raises(TypeError):
        rtdose.load_dose(123)

# Test: FileNotFoundError if file does not exist
def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        rtdose.load_dose("/non/existent/file.dcm")

# Test: ValueError if file is not a valid DICOM
def test_load_invalid_dicom():
    with pytest.raises(ValueError):
        rtdose.load_dose(Path(__file__).parent / "fixtures" / "calibracion.png")

# Test: ValueError if DICOM does not contain required metadata tag
def test_load_missing_metadata():
    with pytest.raises(ValueError):
        rtdose.load_dose(Path(__file__).parent / "fixtures" / "CT_image.dcm")

# Test: Load a valid DICOM file
def test_load_valid_dicom():
    dose = rtdose.load_dose(Path(__file__).parent / "fixtures" / "RD_20x20cm2_256x256pix.dcm")
    assert isinstance(dose, sitk.Image)


## Test eqd2 function
#--------------------

# Test: 3D Dose of 8 Gy/1 fx is equivalent to 17.6 EQD2Gy, using alpha/beta = 3
""" def test_3D_8Gy():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    # Compute EQD2
    dose_eqd2 = rtdose.eqd2(dose8Gy, 3, 1)
    
    dose_eqd2_array = sitk.GetArrayFromImage(dose_eqd2)

    assert pytest.approx(np.mean(dose_eqd2_array), 0.1) == 17.6

# Test: Load an invalid dose object
def test_load_invalid_dose():
    with pytest.raises(ValueError):
        rtdose.eqd2("invalid_input", 3, 10)

# Test: Invalid number for alpha_beta parameter: 0
def test_invalid_number_alpha_beta_parameter():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    with pytest.raises(ValueError):
        rtdose.eqd2(dose8Gy, 0, 10)

# Test: Invalid type for alpha_beta parameter: string
def test_invalid_type_alpha_beta_parameter_string():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    with pytest.raises(ValueError):
        rtdose.eqd2(dose8Gy, "bad parameter", 10)

# Test: Invalid type for number_fractions parameter: string
def test_invalid_type_number_fractions_parameter_string():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    with pytest.raises(ValueError):
        rtdose.eqd2(dose8Gy, 3, "bad parameter")

# Test: Invalid type for number_fractions parameter: float
def test_invalid_type_number_fractions_parameter_float():
    # Create a 3D dose Image
    dose8Gy = sitk.Image(3, 3, 3, sitk.sitkFloat32) + 8
    with pytest.raises(ValueError):
        rtdose.eqd2(dose8Gy, 3, 10.5) """




## Test get_dose_plane_by_coordinate
#------------------------------------
# Test: Get a dose plane by z coordinate that is not in the dose distribution (interpolation)
def testget_dose_plane_by_coordinate():
    dose_distribution = rtdose.load_dose(Path(__file__).parent / "fixtures" / "/home/luis/GH/Dosepy/tests/unit/fixtures/RTDose_3D.dcm")
    z_coordinate = 0.0  # Example z coordinate
    dose_plane = rtdose.get_dose_plane_by_coordinate(dose_distribution, z_coordinate)
    
    assert pytest.approx(dose_plane[50, 50], abs = 0.1) == 13.4


# Test get_dose_plane_by_coordinate with invalid z coordinate
def testget_dose_plane_by_coordinate_invalid_z():
    dose_distribution = rtdose.load_dose(Path(__file__).parent / "fixtures" / "/home/luis/GH/Dosepy/tests/unit/fixtures/RTDose_3D.dcm")
    z_coordinate = 100.0  # Example z coordinate outside the range of the dose distribution
    
    with pytest.raises(ValueError):
        rtdose.get_dose_plane_by_coordinate(dose_distribution, z_coordinate)

# Test: Get a dose plane by z coordinate that is in the dose distribution (not interpolation needed)
def testget_dose_plane_by_coordinate_valid_z():
    dose_distribution = rtdose.load_dose(Path(__file__).parent / "fixtures" / "/home/luis/GH/Dosepy/tests/unit/fixtures/RTDose_3D.dcm")
    z_coordinate = 1  # Example z coordinate that is in the dose distribution
    dose_plane = rtdose.get_dose_plane_by_coordinate(dose_distribution, z_coordinate)
    
    assert pytest.approx(dose_plane[50, 50], abs = 0.1) == 13.2

# Test: Give a z_coordinate that is not a number
def testget_dose_plane_by_coordinate_invalid_z_type():
    dose_distribution = rtdose.load_dose(Path(__file__).parent / "fixtures" / "/home/luis/GH/Dosepy/tests/unit/fixtures/RTDose_3D.dcm")
    
    with pytest.raises(ValueError):
        rtdose.get_dose_plane_by_coordinate(dose_distribution, "invalid_z")


## Test get_2D_mask_by_coordinates_and_image_shape()
#-------------------------------------------------
# Test: shape and mean of the mask generated by coordinates and image shape
def test_get_2D_mask_by_coordinates_and_image_shape():
    # Load a dose distribution
    dose = rtdose.load_dose(Path(__file__).parent / "fixtures" / "RTDose_3D.dcm")
    # Load the structures
    ds = rtdose.load_structures(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
    # Get the coordinates of a structure
    coordinates = ds.get_coordinates("PTV_High")[0]
    # Get the mask by coordinates and image shape
    mask = rtdose.get_2D_mask_by_coordinates_and_image_shape(coordinates, dose)

    assert mask.shape == (90, 102)
    assert pytest.approx(np.mean(mask), abs=0.1) == 0.7
    
# Test: TypeError if coordinates is not a numpy array
def test_get_2D_mask_by_coordinates_and_image_shape_type_error_coordinates():
    dose = rtdose.load_dose(Path(__file__).parent / "fixtures" / "RTDose_3D.dcm")
    
    with pytest.raises(TypeError):
        rtdose.get_2D_mask_by_coordinates_and_image_shape("invalid_coordinates", dose)

# Test: TypeError if image is not a SimpleITK image
def test_get_2D_mask_by_coordinates_and_image_shape_type_error_image():
    coordinates = np.array([[0, 0, 0], [1, 1, 1]])
    
    with pytest.raises(TypeError):
        rtdose.get_2D_mask_by_coordinates_and_image_shape(coordinates, "invalid_image")


## Test get_dose_in_structure
# Test: for PTV_High and mean
def test_get_dose_ptv_mean():
    # Load the dose distribution
    dose_distribution = rtdose.load_dose(Path(__file__).parent / "fixtures" / "RTDose_3D.dcm")
    # Load the structures
    ds = rtdose.load_structures(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
    # Get the coordinates of the PTV_High structure
    ptv_coordinates = ds.get_coordinates("PTV_High")

    # Get the dose values for the PTV structure
    dose_ptv = rtdose.get_dose_in_structure(
        dose_distribution,
        ptv_coordinates)
    
    assert len(dose_ptv) == 6259  # Number of points
    assert pytest.approx(np.mean(dose_ptv), abs=0.1) == 2.80  # Mean dose in the PTV structure given by Eclipse V16

# Test: get dose for BODY structure
def test_get_dose_body():
    # Load the dose distribution
    dose_distribution = rtdose.load_dose(Path(__file__).parent / "fixtures" / "RTDose_3D.dcm")
    # Load the structures
    ds = rtdose.load_structures(Path(__file__).parent / "fixtures" / "RS_anonymized.dcm")
    # Get the coordinates of the BODY structure
    body_coordinates = ds.get_coordinates("BODY")
                                         
    # Get the dose values for the BODY structure
    dose_body = rtdose.get_dose_in_structure(
        dose_distribution,
        body_coordinates)
        
    assert pytest.approx(np.mean(dose_body), abs=0.1) == 0.67  # Mean dose in the BODY structure given by Eclipse V16