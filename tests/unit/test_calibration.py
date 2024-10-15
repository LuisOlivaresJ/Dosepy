# This file contains the unit tests for the calibration.py file.

import unittest
import pytest

import numpy as np

from Dosepy.calibration import CalibrationLUT
from Dosepy.image import load, TiffImage

from pathlib import Path

cwd = Path(__file__).parent


@pytest.fixture
def example_image():
    file_path = cwd / "fixtures" / "CAL" / "film20240620_002.tif"
    return load(file_path)

@pytest.fixture
def example_profile():
    profile_path = cwd / "fixtures" / "CAL" / "BeamProfile.txt"
    return np.genfromtxt(profile_path)

@pytest.fixture
def example_metadata():
    return {
        "author": "John Doe",
        "film_lote": "20240620",
        "scanner": "EPSON 12000XL",
        "date_exposed": "2024-06-20",
        "date_scanned": "2024-06-21",
        "wait_time": 24,
    }

# Test the instance of the CalibrationLUT class
def test_instance(example_image, example_profile, example_metadata):
    
        profile = example_profile
        cal = CalibrationLUT(example_image,
                            lateral_correction = True,
                            beam_profile = profile,
                            filter = 3,
                            metadata = example_metadata,
                            )
        assert isinstance(cal, CalibrationLUT)

# Test a correct orientation of the image 
# #TODO Implement the test_correct_orientation method
def test_correct_orientation(example_image):
    
    cal = CalibrationLUT(example_image)

    assert cal.tiff_image.orientation == "portrait"

# Test the initialization of the CalibrationLUT class
def test_initialization(example_image, example_profile, example_metadata):
    
    profile = example_profile

    cal = CalibrationLUT(example_image,
                        lateral_correction = True,
                        beam_profile = profile,
                        filter = 3,
                        metadata = example_metadata,
                        )
    
    assert isinstance(cal.tiff_image, TiffImage)
    assert cal.lut["author"] == "John Doe"
    assert cal.lut["film_lote"] == "20240620"
    assert cal.lut["scanner"] == "EPSON 12000XL"
    assert cal.lut["date_exposed"] == "2024-06-20"
    assert cal.lut["date_scanned"] == "2024-06-21"
    assert cal.lut["wait_time"] == 24
    assert cal.lut["resolution"] == 75


# Test CalibrationLUT initialization with default values
def test_initialization_default(example_image):
    
    cal = CalibrationLUT(example_image)

    assert cal.lut["author"] is None
    assert cal.lut["film_lote"] is None
    assert cal.lut["scanner"] is None
    assert cal.lut["date_exposed"] is None
    assert cal.lut["date_scanned"] is None
    assert cal.lut["wait_time"] is None


# Test the create_central_rois method, 6 rois should be created
def test_create_central_rois_with_known_number_of_rois(example_image):
    
    cal = CalibrationLUT(example_image)

    cal.create_central_rois(size = (10, 10))

    assert len(cal.lut["rois"]) == 6


# Test the create_central_rois method
def test_create_central_rois(example_image):
    
    cal = CalibrationLUT(example_image)

    # Create rectangular rois, 5 mm width and 10 mm height
    cal.create_central_rois(size = (30, 8))

    assert cal.lut["rois"] == [
        {
            "x": 81,
            "y": 390,
            "width": 88,
            "height": 23,
        },
        {
            "x": 174,
            "y": 399,
            "width": 88,
            "height": 23,
        },
        {
            "x": 271,
            "y": 392,
            "width": 88,
            "height": 23,
        },
        {
            "x": 365,
            "y": 402,
            "width": 88,
            "height": 23,
        },
        {
            "x": 457,
            "y": 399,
            "width": 88,
            "height": 23,
        },
        {
            "x": 559,
            "y": 401,
            "width": 88,
            "height": 23,
        },
    ]


# Test the compute_lateral_lut method
def test_compute_lateral_lut(example_image):
        
        cal = CalibrationLUT(example_image)
    
        cal.create_central_rois(size = (180, 8))
    
        cal.compute_lateral_lut()

        assert cal.lut["lateral_limits"]["left"] == -93
        assert cal.lut["lateral_limits"]["right"] == 82
    
        
        assert cal.lut[(2, 0)]["I_red"] == 42384
        assert cal.lut[(2, 0)]["S_red"] == 54

        assert cal.lut[(2, 0)]["I_green"] == 41719
        assert cal.lut[(2, 0)]["S_green"] == 80

        assert cal.lut[(2, 0)]["I_blue"] == 32419
        assert cal.lut[(2, 0)]["S_blue"] == 52

        assert cal.lut[(2, 0)]["I_mean"] == 38840
        assert cal.lut[(2, 0)]["S_mean"] == 36
        

# Test the set_doses method of the CalibrationLUT class, with unordered doses
def test_set_doses(example_image, doses = [2, 0, 4, 10, 8, 6]):
    
    cal = CalibrationLUT(example_image)

    cal.set_doses(doses)

    assert cal.lut["nominal_doses"] == [0, 2, 4, 6, 8, 10]


# Test the compute_lut method
## TODO: Implement the test_compute_lut method and delete the CalibImage class
