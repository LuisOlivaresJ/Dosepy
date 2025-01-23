# This file contains the unit tests for the calibration.py file.

import pytest

import numpy as np

from Dosepy.calibration import LUT#, _get_dose_from_fit
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

@pytest.fixture
def cal_img_with_filters():
    file_path = cwd / "fixtures/CAL20241106_001.tif"
    cal_img = load(file_path)

    return cal_img

# Test the instance of the LUT class
def test_instance(example_image, example_metadata):
    
        #profile = example_profile
        cal = LUT(example_image,
                            metadata = example_metadata,
                            )
        assert isinstance(cal, LUT)

# Test a correct orientation of the image 
# #TODO Implement the test_correct_orientation method
def test_correct_orientation(example_image):
    
    cal = LUT(example_image)

    assert cal.tiff_image.orientation == "portrait"

# Test the initialization of the LUT class
def test_initialization(example_image, example_profile, example_metadata):
    
    profile = example_profile

    cal = LUT(
         example_image, 
         metadata = example_metadata
         )
    
    assert isinstance(cal.tiff_image, TiffImage)
    assert cal.lut["author"] == "John Doe"
    assert cal.lut["film_lote"] == "20240620"
    assert cal.lut["scanner"] == "EPSON 12000XL"
    assert cal.lut["date_exposed"] == "2024-06-20"
    assert cal.lut["date_scanned"] == "2024-06-21"
    assert cal.lut["wait_time"] == 24
    assert cal.lut["resolution"] == 75


# Test LUT initialization with default values
def test_initialization_default(example_image):
    
    cal = LUT(example_image)

    assert cal.lut["author"] is None
    assert cal.lut["film_lote"] is None
    assert cal.lut["scanner"] is None
    assert cal.lut["date_exposed"] is None
    assert cal.lut["date_scanned"] is None
    assert cal.lut["wait_time"] is None


# Test the set_central_rois method, 6 rois should be created
def test_set_central_rois_with_known_number_of_rois(example_image):
    
    cal = LUT(example_image)

    cal.set_central_rois(size = (10, 10))

    assert len(cal.lut["rois"]) == 6


# Test the set_central_rois method
def test_set_central_rois(example_image):
    
    cal = LUT(example_image)

    # Create rectangular rois, 5 mm width and 10 mm height
    cal.set_central_rois(size = (30, 8))

    assert cal.lut["rois"] == [
        {
            "x": 81,
            "y": 391,
            "width": 88,
            "height": 23,
        },
        {
            "x": 174,
            "y": 398,
            "width": 88,
            "height": 23,
        },
        {
            "x": 270,
            "y": 392,
            "width": 88,
            "height": 23,
        },
        {
            "x": 365,
            "y": 401,
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
            "y": 400,
            "width": 88,
            "height": 23,
        },
    ]


# Test the compute_lateral_lut method
def test_compute_lateral_lut(example_image):
        
        # Relative tolerance of 1% for mean intensity.
        rel = 1/100
        
        cal = LUT(example_image)
    
        cal.set_central_rois(size = (180, 8))
    
        cal.compute_lateral_lut()

        assert cal.lut["lateral_limits"]["left"] == -93
        assert cal.lut["lateral_limits"]["right"] == 82
    
        assert cal.lut[(2, 0)]["I_red"] == pytest.approx(42353, rel=rel)

        assert cal.lut[(2, 0)]["I_green"] == pytest.approx(41625, rel=rel)

        assert cal.lut[(2, 0)]["I_blue"] == pytest.approx(32326, rel=rel)

        assert cal.lut[(2, 0)]["I_mean"] == pytest.approx(38768, rel=rel)


# Test lateral response below to 10 % of the central response        
def test_get_lateral_response_below_10(example_image):
        
        cal = LUT(example_image) 
        cal.set_central_rois(size = (180, 8)) 
        cal.compute_lateral_lut()
        
        intensity, _, _ = cal.get_lateral_intensity(roi=5, channel="red")
        I_central = intensity[int(len(intensity)/2)]
        I_relative = intensity/I_central

        assert all(abs(I_relative) < 10) 

# Test the set_doses method of the LUT class, with unordered doses
def test_set_doses(example_image, doses = [2, 0, 4, 10, 8, 6]):
    
    cal = LUT(example_image)

    cal.set_doses(doses)

    assert cal.lut["nominal_doses"] == [0, 2, 4, 6, 8, 10]


def test_set_beam_profile_two_columns(
          example_image,
          ):
    
    cal = LUT(example_image)

    cal.set_beam_profile(
          str(cwd / "fixtures" / "CAL" / "BeamProfile.csv"),
    )

    # Check the shape of the beam profile
    assert cal.lut["beam_profile"]
    

def test_get_lateral_doses(example_image):
    # Test the get_lateral_doses method of the LUT class
    cal = LUT(example_image)
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
    # Test the get_dose_from_fit method of the LUT class
    cal = LUT(example_image)
    cal.set_beam_profile(
          str(cwd / "fixtures" / "CAL" / "BeamProfile.csv"),
    )
    cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
    cal.set_central_rois(size = (180, 8))
    cal.compute_lateral_lut()

    position = 0

    dose = cal._get_lateral_doses(position = position)

    # Red channel
    i_red, u_r = cal.get_intensities(
          lateral_position = position,
          channel = "red",
    )
    
    dose_from_fit_red_poly, p, u_p = cal._get_dose_from_fit(
          calib_film_intensities = i_red,
          calib_dose = dose,
          fit_function = "polynomial",
          intensities = i_red,
    )

    # Green channel
    i_green, u_g = cal.get_intensities(
          lateral_position = position,
          channel = "green",
    )
    #response_green = -np.log10(i_green / i_green[0])

    dose_from_fit_green_poly, p, u_p = cal._get_dose_from_fit(
            calib_film_intensities = i_green,
            calib_dose = dose,
            fit_function = "polynomial",
            intensities = i_green,
        )
    
    # Blue channel
    i_blue, u_b = cal.get_intensities(
          lateral_position = position,
          channel = "blue",
    )

    #response_blue = -np.log10(i_blue / i_blue[0])

    dose_from_fit_blue_poly, p, u_p = cal._get_dose_from_fit(
            calib_film_intensities = i_blue,
            calib_dose = dose,
            fit_function = "polynomial",
            intensities = i_blue,
        )

    assert dose == pytest.approx(dose_from_fit_red_poly, rel = 5e-1)
    assert dose == pytest.approx(dose_from_fit_green_poly, rel = 5e-1)
    assert dose == pytest.approx(dose_from_fit_blue_poly, rel = 5e-1)


def test_get_dose_from_fit_rational(example_image):
    # Test the get_dose_from_fit method of the LUT class
    cal = LUT(example_image)
    cal.set_beam_profile(
          str(cwd / "fixtures" / "CAL" / "BeamProfile.csv"),
    )
    cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
    cal.set_central_rois(size = (180, 8))
    cal.compute_lateral_lut()

    position = 0

    dose = cal._get_lateral_doses(position = position)

    # Red channel
    i_red, u_r = cal.get_intensities(
          lateral_position = position,
          channel = "red",
    )
    
    dose_from_fit_red_rat, p, up = cal._get_dose_from_fit(
        calib_film_intensities = i_red,
        calib_dose = dose,
        intensities = i_red,
        fit_function = "rational",
    )

    # Green channel
    i_green, u_g = cal.get_intensities(
          lateral_position = position,
          channel = "green",
    )

    dose_from_fit_green_rat, p, up = cal._get_dose_from_fit(
            calib_film_intensities = i_green,
            calib_dose = dose,
            intensities = i_green,
            fit_function = "rational",
        )
    
    # Blue channel
    i_blue, u_b = cal.get_intensities(
          lateral_position = position,
          channel = "blue",
    )

    dose_from_fit_blue_rat, p, up = cal._get_dose_from_fit(
            calib_film_intensities = i_blue,
            calib_dose = dose,
            intensities = i_blue,
            fit_function = "rational",
        )

    assert dose[1:] == pytest.approx(dose_from_fit_red_rat[1:], rel = 5e-1)
    assert dose[1:] == pytest.approx(dose_from_fit_green_rat[1:], rel = 5e-1)
    assert dose[1:] == pytest.approx(dose_from_fit_blue_rat[1:], rel = 5e-1)


def test_compute_central_lut(example_image):
    # Test the compute_central_lut method of the LUT class
    cal = LUT(example_image)
    cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
    cal.set_central_rois(size = (8, 8))

    cal.compute_central_lut()

    intensities, std = cal.get_intensities(channel = "red")
    
    # Assert first roi
    assert int(intensities[0]) == 42533
    assert int(std[0]) == 133


# Test if LUT has mean intensities of filters
def test_optical_filters_in_lut(cal_img_with_filters):

    cal = LUT(cal_img_with_filters)
    cal_img_with_filters.set_labeled_films_and_filters()
    cal._set_roi_and_intensity_of_optical_filters()
    intensity_of_filters = cal.get_intensities_of_optical_filters()

    assert intensity_of_filters[0] == pytest.approx(9329, 0.01)
    assert intensity_of_filters[1] == pytest.approx(13264, 0.01)
    assert intensity_of_filters[2] == pytest.approx(20746, 0.01)


# Test coordinate of optical filters
def test_coordinate_optical_filters(cal_img_with_filters):
    
    cal = LUT(cal_img_with_filters)
    cal_img_with_filters.set_labeled_films_and_filters()
    cal._set_roi_and_intensity_of_optical_filters()

    rois = cal.get_rois_of_optical_filters()
    
    assert rois[0].get("x") == pytest.approx(960, abs=20)
    assert rois[0].get("y") == pytest.approx(600, abs=20)
    assert rois[0].get("radius") == pytest.approx(29, abs=10)

    assert rois[1].get("x") == pytest.approx(960, abs=20)
    assert rois[1].get("y") == pytest.approx(310, abs=20)
    assert rois[1].get("radius") == pytest.approx(29, abs=10)

    assert rois[2].get("x") == pytest.approx(960, abs=20)
    assert rois[2].get("y") == pytest.approx(455, abs=20)
    assert rois[2].get("radius") == pytest.approx(29, abs=10)