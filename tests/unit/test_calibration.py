# This file contains the unit tests for the calibration.py file.

import pytest

import numpy as np

from Dosepy.calibration import CalibrationLUT, _get_dose_from_fit
from Dosepy.image import load, TiffImage

from pathlib import Path

cwd = Path(__file__).parent


@pytest.fixture
def example_image():
    file_path = cwd / "fixtures" / "CAL" / "film20240620_002.tif"
    return load(file_path)

@pytest.fixture
def example_profile():
    profile_path = cwd / "fixtures" / "CAL" / "BeamProfile.csv"
    return np.genfromtxt(profile_path, delimiter=",")

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
def test_instance(example_image, example_metadata):
    
        #profile = example_profile
        cal = CalibrationLUT(example_image,
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
                        #lateral_correction = True,
                        #beam_profile = profile,
                        #filter = 3,
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


# Test the set_central_rois method, 6 rois should be created
def test_set_central_rois_with_known_number_of_rois(example_image):
    
    cal = CalibrationLUT(example_image)

    cal.set_central_rois(size = (10, 10))

    assert len(cal.lut["rois"]) == 6


# Test the set_central_rois method
def test_set_central_rois(example_image):
    
    cal = CalibrationLUT(example_image)

    # Create rectangular rois, 5 mm width and 10 mm height
    cal.set_central_rois(size = (30, 8))

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
    
        cal.set_central_rois(size = (180, 8))
    
        cal.compute_lateral_lut()

        assert cal.lut["lateral_limits"]["left"] == -93
        assert cal.lut["lateral_limits"]["right"] == 82
    
        
        assert cal.lut[(2, 0)]["I_red"] == 42353
        assert cal.lut[(2, 0)]["S_red"] == 110

        assert cal.lut[(2, 0)]["I_green"] == 41625
        assert cal.lut[(2, 0)]["S_green"] == 108

        assert cal.lut[(2, 0)]["I_blue"] == 32326
        assert cal.lut[(2, 0)]["S_blue"] == 95

        assert cal.lut[(2, 0)]["I_mean"] == 38768
        assert cal.lut[(2, 0)]["S_mean"] == 60
        
def test_get_lateral_response_below_10(example_image):
        
        cal = CalibrationLUT(example_image) 
        cal.set_central_rois(size = (180, 8)) 
        cal.compute_lateral_lut()
        
        intensity, _, _ = cal.get_lateral_respose(roi=5, channel="red")
        I_central = intensity[int(len(intensity)/2)]
        I_relative = intensity/I_central

        assert all(abs(I_relative) < 10) 

# Test the set_doses method of the CalibrationLUT class, with unordered doses
def test_set_doses(example_image, doses = [2, 0, 4, 10, 8, 6]):
    
    cal = CalibrationLUT(example_image)

    cal.set_doses(doses)

    assert cal.lut["nominal_doses"] == [0, 2, 4, 6, 8, 10]


def test_set_beam_profile_two_columns(
          example_image,
          ):
    
    cal = CalibrationLUT(example_image)

    cal.set_beam_profile(
          str(cwd / "fixtures" / "CAL" / "BeamProfile.csv"),
    )

    # Check the shape of the beam profile
    assert cal.lut["beam_profile"]
    

def test_get_lateral_doses(example_image):
    # Test the get_lateral_doses method of the CalibrationLUT class
    cal = CalibrationLUT(example_image)
    cal.set_beam_profile(
          str(cwd / "fixtures" / "CAL" / "BeamProfile.csv"),
    )
    cal.set_doses([0, 2, 4, 6, 8, 10])

    assert cal._get_lateral_doses(position = 0) == [0, 2, 4, 6, 8, 10]

    corrected_doses = list(np.array([0, 2, 4, 6, 8, 10]) * 1.02699)
    assert cal._get_lateral_doses(position = -105) == pytest.approx(corrected_doses, rel = 1e-1)

    # Test an interpolated position
    corrected_doses = list(np.array([0, 2, 4, 6, 8, 10]) * 1.00227)
    assert cal._get_lateral_doses(position = 12.0) == pytest.approx(corrected_doses, rel = 1e-1)
    

def test_get_dose_from_fit_polynomial(example_image):
    # Test the get_dose_from_fit method of the CalibrationLUT class
    cal = CalibrationLUT(example_image)
    cal.set_beam_profile(
          str(cwd / "fixtures" / "CAL" / "BeamProfile.csv"),
    )
    cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
    cal.set_central_rois(size = (180, 8))
    cal.compute_lateral_lut()

    position = 0

    dose = cal._get_lateral_doses(position = position)

    # Red channel
    i_red, _ = cal._get_intensities(
          lateral_position = position,
          channel = "red",
    )
    response_red = -np.log10(i_red / i_red[0])
    
    dose_from_fit_red_poly = _get_dose_from_fit(
          response_red,
          dose,
          response_red,
          "polynomial",
    )

    # Green channel
    i_green, _ = cal._get_intensities(
          lateral_position = position,
          channel = "green",
    )
    response_green = -np.log10(i_green / i_green[0])

    dose_from_fit_green_poly = _get_dose_from_fit(
            response_green,
            dose,
            response_green,
            "polynomial",
        )
    
    # Blue channel
    i_blue, _ = cal._get_intensities(
          lateral_position = position,
          channel = "blue",
    )

    response_blue = -np.log10(i_blue / i_blue[0])

    dose_from_fit_blue_poly = _get_dose_from_fit(
            response_blue,
            dose,
            response_blue,
            "polynomial",
        )

    assert dose == pytest.approx(dose_from_fit_red_poly, rel = 5e-1)
    assert dose == pytest.approx(dose_from_fit_green_poly, rel = 5e-1)
    assert dose == pytest.approx(dose_from_fit_blue_poly, rel = 5e-1)


def test_get_dose_from_fit_rational(example_image):
    # Test the get_dose_from_fit method of the CalibrationLUT class
    cal = CalibrationLUT(example_image)
    cal.set_beam_profile(
          str(cwd / "fixtures" / "CAL" / "BeamProfile.csv"),
    )
    cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
    cal.set_central_rois(size = (180, 8))
    cal.compute_lateral_lut()

    position = 0

    dose = cal._get_lateral_doses(position = position)

    # Red channel
    i_red, _ = cal._get_intensities(
          lateral_position = position,
          channel = "red",
    )
    response_red = i_red / i_red[0]
    
    dose_from_fit_red_rat = _get_dose_from_fit(
          response_red,
          dose,
          response_red,
          "rational",
    )

    # Green channel
    i_green, _ = cal._get_intensities(
          lateral_position = position,
          channel = "green",
    )
    response_green = i_green / i_green[0]

    dose_from_fit_green_rat = _get_dose_from_fit(
            response_green,
            dose,
            response_green,
            "rational",
        )
    
    # Blue channel
    i_blue, _ = cal._get_intensities(
          lateral_position = position,
          channel = "blue",
    )

    response_blue = i_blue / i_blue[0]

    dose_from_fit_blue_rat = _get_dose_from_fit(
            response_blue,
            dose,
            response_blue,
            "rational",
        )

    assert dose[1:] == pytest.approx(dose_from_fit_red_rat[1:], rel = 5e-1)
    assert dose[1:] == pytest.approx(dose_from_fit_green_rat[1:], rel = 5e-1)
    assert dose[1:] == pytest.approx(dose_from_fit_blue_rat[1:], rel = 5e-1)


def test_compute_central_lut(example_image):
    # Test the compute_central_lut method of the CalibrationLUT class
    cal = CalibrationLUT(example_image)
    cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
    cal.set_central_rois(size = (8, 8))

    cal.compute_central_lut()

    intensities, std = cal._get_intensities(channel = "red")
    
    # Assert first roi
    assert int(intensities[0]) == 42544
    assert int(std[0]) == 137
