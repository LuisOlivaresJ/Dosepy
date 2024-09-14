import unittest
from Dosepy.config.io_settings import Settings
from Dosepy.config.io_settings import load_settings

# Test for the io_settings.py file
class TestSettings(unittest.TestCase):

    # Test a correct creation of an instance of the Settings class
    def test_settings_instance(self):
        settings = Settings(roi_size_h = 8.0, roi_size_v = 8.0)
        self.assertIsInstance(settings, Settings)

    # Test the get_calib_roi_size method
    def test_get_calib_roi_size(self):
        settings = Settings(roi_size_h = 8.0, roi_size_v = 8.0)
        self.assertEqual(settings.get_calib_roi_size(), (8.0, 8.0))

    # Test the load_settings function
    def test_load_settings(self):
        settings = load_settings()
        self.assertIsInstance(settings, Settings)

    # Test the get_calib_roi_size method from the load_settings function
    def test_get_calib_roi_size_load_settings(self):
        settings = load_settings()
        self.assertEqual(settings.get_calib_roi_size(), (8.0, 8.0))
