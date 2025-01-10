import unittest

from pathlib import Path

from Dosepy import image
import numpy as np

cwd = Path(__file__).parent

class TestReadableImage(unittest.TestCase):

    def test_bad_file(self):
        #cwd = Path(__file__).parent
        file_path = cwd / "fixtures" / "an_incorrect_image_file.md"
        with self.assertRaises(TypeError):
            img = image.load(file_path)

    def test_bad_format(self):
        #cwd = Path(__file__).parent
        file_path = cwd / "fixtures" / "calibracion.png"
        self.assertFalse(image._is_tif_file(file_path))

    def test_bad_RGB(self):
        file_path = cwd / "fixtures" / "red_channel.tif"
        self.assertFalse(image._is_RGB(file_path))

    def test_readable_image(self):
        #cwd = Path(__file__).parent
        file_path = cwd / "fixtures" / "image.tif"
        img = image.load(file_path)
        self.assertIsInstance(img, image.TiffImage)

    #TO DO: A test function for TiffImage.dpi


class TestTiffImage(unittest.TestCase):

    # Test the get_labeled_objects method
    def test_get_labeled_objects_six_films(self):
        file_path = cwd / "fixtures" / "CAL" / "film20240620_002.tif"
        img = image.load(file_path)
        _, num_object = img.get_labeled_objects(
            return_num=True
        )
        self.assertEqual(num_object, 6)

    def test_get_labeled_objects_with_filters(self):
        file_path = cwd / "fixtures" / "CAL20241106_001.tif"
        img = image.load(file_path)
        _, num_object = img.get_labeled_objects(
            return_num=True,
        )
        self.assertEqual(num_object, 11)


class Test_ArrayImage(unittest.TestCase):
    # Test the creation of an ArrayImage instance
    def test_create_array_image_instance(self):
        img = image.ArrayImage(
            array = np.array([[1, 2], [3, 4]]),
            dpi = 100,
            )
        self.assertIsInstance(img, image.ArrayImage)

    # Test the reduce_resolution_as method
    def test_reduce_resolution_as(self):
        dose = image.ArrayImage(
            array = np.random.rand(100, 100),
            dpi = 100,
            )
        reference = image.ArrayImage(
            array = np.random.rand(10,10),
            dpi = 10,
            )
        dose.reduce_resolution_as(reference)
        self.assertEqual(dose.shape, (10, 10))

    # Test raiseError in reduce_resolution_as method with images of different physical dimension
    def test_reduce_resolution_as_different_physical_dimension(self):
        dose = image.ArrayImage(
            array = np.random.rand(106, 106),
            dpi = 100,
            )
        reference = image.ArrayImage(
            array = np.random.rand(10,10),
            dpi = 10,
            )
        with self.assertRaises(AttributeError):
            dose.reduce_resolution_as(reference)

    # Test reduce_resolution_as method with images of different 
    # physical dimension, image to crop is bigger than the reference
    # but withing the tolerance
    def test_reduce_resolution_with_different_physical_dimension_but_in_tolerance(self):
        dose = image.ArrayImage(
            array = np.random.rand(104, 104),
            dpi = 100,
            )
        reference = image.ArrayImage(
            array = np.random.rand(10,10),
            dpi = 10,
            )
        dose.reduce_resolution_as(reference)
        self.assertEqual(dose.shape, (10, 10))

    # Test reduce_resolution_as method with images of different
    # physical dimension, image to crop is smaller than the reference
    # but within the tolerance
    def test_reduce_resolution_with_different_physical_dimension_but_in_tolerance_2(self):
        dose = image.ArrayImage(
            array = np.random.rand(96, 96),
            dpi = 100,
            )
        reference = image.ArrayImage(
            array = np.random.rand(10,10),
            dpi = 10,
            )
        dose.reduce_resolution_as(reference)
        self.assertEqual(dose.shape, (10, 10))

    # Test reduce_resolution_as method with a cuasi real example
    def test_reduce_resolution_with_a_real_example(self):
        dose = image.ArrayImage(
            array = np.random.rand(2362, 2362),
            dpi = 300,
            )
        reference = image.ArrayImage(
            array = np.random.rand(256, 256),
            dpi = 32.512,
            )
        dose.reduce_resolution_as(reference)
        self.assertEqual(dose.shape, (256, 256))


class Test_DoseImage(unittest.TestCase):

    # Test the creation of a DoseImage instance
    def test_create_dose_image_instance(self):
        dose = image.DoseImage(
            array = np.array([[1, 2], [3, 4]]),
            dpi = 100,
            reference_point=[0, 0],
            orientation=(1, 0, 0, 0, 1, 0),
            dose_unit="Gy",
            )
        self.assertIsInstance(dose, image.DoseImage)
