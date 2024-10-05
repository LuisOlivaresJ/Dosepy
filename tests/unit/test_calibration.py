# This file contains the unit tests for the calibration.py file.

import unittest
from Dosepy.calibration import CalibrationLUT

# Test the instance of the CalibrationLUT class
class TestCalibration(unittest.TestCase):

    # Test the instance of the CalibrationLUT class
    def test_instance(self):
        self.assertIsInstance(CalibrationLUT(), CalibrationLUT)


    # TODO: Test the create_central_rois method

    # Test the compute_lut method
    ## TODO: Implement the test_compute_lut method and delete the CalibImage class
