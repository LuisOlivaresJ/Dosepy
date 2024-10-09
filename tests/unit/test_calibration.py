# This file contains the unit tests for the calibration.py file.

import unittest
import pytest

import numpy as np

from Dosepy.calibration import CalibrationLUT
from Dosepy.image import load

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


# Test the instance of the CalibrationLUT class
def test_instance(example_image, example_profile):
    
        profile = example_profile
        information = {
            "author": "John Doe",
            "film_lote": "20240620",
            "scanner": "EPSON Perfection 12000XL",
            "date_exposed": "2024-06-20",
            "date_scanned": "2024-06-21",
            "wait_time": 24,
            
        }
        cal = CalibrationLUT(example_image,
                            doses = [0, 2, 4, 6, 8, 10],
                            lateral_correction = True,
                            beam_profile = profile,
                            filter = 3,
                            metadata = information
                            )
        assert isinstance(cal, CalibrationLUT)



# Test the initialization of the CalibrationLUT class



# TODO: Test the create_central_rois method

# Test the compute_lut method
## TODO: Implement the test_compute_lut method and delete the CalibImage class
