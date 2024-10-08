# This file contains the unit tests for the calibration.py file.

import unittest

import numpy as np

from Dosepy.calibration import CalibrationLUT
from Dosepy.image import load

from pathlib import Path

cwd = Path(__file__).parent

# Test the instance of the CalibrationLUT class
class TestCalibration(unittest.TestCase):

    # Test the instance of the CalibrationLUT class
    def test_instance(self):
        file_path = cwd / "fixtures" / "CAL" / "film20240620_002.tif"
        img = load(file_path)
        profile_path = cwd / "fixtures" / "CAL" / "BeamProfile.txt"
        profile = np.genfromtxt(profile_path)
        cal = CalibrationLUT(img,
                             doses = [0, 2, 4, 6, 8, 10],
                             lateral_correction = True,
                             beam_profile = profile,
                             filter = 3,
                             metadata = {}  # TODO
                            )
        self.assertIsInstance(cal, CalibrationLUT)


    # TODO: Test the create_central_rois method

    # Test the compute_lut method
    ## TODO: Implement the test_compute_lut method and delete the CalibImage class
