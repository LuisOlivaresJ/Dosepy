import unittest
from Dosepy.config.io_settings import Settings
from Dosepy.config.io_settings import load_settings

# Test for the io_settings.py file
class TestSettings(unittest.TestCase):

    # Test a correct creation of an instance of the Settings class
    def test_settings_instance(self):
        settings = Settings(roi_size_h = 8.0, roi_size_v = 8.0)
        self.assertIsInstance(settings, Settings)

    # Test the get_roi_automatic method
    def test_get_roi_automatic(self):
        settings = Settings(roi_automatic = True)
        self.assertTrue(settings.get_roi_automatic())

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

    # Test the set_roi_automatic method
    def test_set_roi_automatic(self):
        settings = Settings(roi_automatic = True)
        settings.set_roi_automatic(False)
        self.assertFalse(settings.get_roi_automatic())

    # Test the set_calib_roi_size method
    def test_set_calib_roi_size(self):
        settings = Settings(roi_size_h = 10.0, roi_size_v = 10.0)
        settings.set_calib_roi_size((8.0, 8.0))
        self.assertEqual((settings.roi_size_h, settings.roi_size_v), (8.0, 8.0))

    # Test the set_calib_roi_size method with an exception
    def test_set_calib_roi_size_exception(self):
        settings = Settings(roi_size_h = 8.0, roi_size_v = 8.0)
        with self.assertRaises(ValueError):
            # The tuple must have two elements
            settings.set_calib_roi_size(10.0)

    # Test the set_calib_roi_size method with an exception
    def test_set_calib_roi_size_exception(self):
        settings = Settings(roi_size_h = 8.0, roi_size_v = 8.0)
        with self.assertRaises(ValueError):
            # The tuple must have two elements
            settings.set_calib_roi_size((10.0, 10.0, 10.0))

    # Test the set_calib_roi_size method checking the creation of a new settings.toml file
    def test_set_calib_roi_size_new_settings_file(self):
        settings = Settings(roi_size_h = 10, roi_size_v = 10)
        settings.set_calib_roi_size((8.0, 8.0))
        settings = load_settings()
        self.assertEqual(settings.get_calib_roi_size(), (8.0, 8.0))

    # Test the get_channel method
    def test_get_channel(self):
        settings = load_settings()
        self.assertEqual(settings.get_channel(), "Red")

    # Test the set_channel method
    def test_set_channel(self):
        settings = load_settings()
        settings.set_channel("Red")
        self.assertEqual(settings.channel, "Red")

    def test_get_fit_function(self):
        settings = load_settings()
        self.assertEqual(settings.get_fit_function(), "Rational")

    # Test the set_fit_function method
    def test_set_fit_function(self):
        settings = load_settings()
        settings.set_fit_function("Rational")
        self.assertEqual(settings.fit_function, "Rational")

    # Test the get_lateral_correction method
    #def test_get_lateral_correction(self):
    #    settings = load_settings()
    #    self.assertEqual(settings.get_lateral_correction(), False)

    # Test the set_lateral_correction method
    def test_set_lateral_correction(self):
        settings = load_settings()
        settings.set_lateral_correction(True)
        self.assertEqual(settings.lateral_correction, True)