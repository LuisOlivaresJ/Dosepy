import unittest

from pathlib import Path

from Dosepy import image
from Dosepy.image import (
    load,
    load_multiples,
    average_tiff_images,
    equate_array_size,
    stack_images
)
from skimage.measure import label
import numpy as np
import pytest
import os

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


    def test_set_labeled_films_and_filters(self):
        """Test set_labeled_films_and_filters, count the number of films"""
        file_path = cwd / "fixtures" / "CAL20241106_001.tif"
        img = image.load(file_path)
        img.set_labeled_films_and_filters()
        _, num_of_films = label(img.labeled_films, return_num=True)
        self.assertEqual(num_of_films, 8)


    def test_set_labeled_films_and_filters(self):
        """Test set_labeled_films_and_filters, count the number of films"""
        file_path = cwd / "fixtures" / "CAL20241106_001.tif"
        img = image.load(file_path)
        img.set_labeled_films_and_filters()
        _, num_of_filters = label(img.labeled_optical_filters, return_num=True)
        self.assertEqual(num_of_filters, 3)


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






#=============================================
# Test helper functions
#=============================================
# Helper function to get the paths of all files in a directory
# as str or PosixPath
def get_file_paths(directory: str, type: str) -> list[str]:
    """
    Use type = str to get a list of strings
    or  type = posix_path to get a list of PosixPath instances
    """
    # List to store file paths
    file_paths = []

    # Iterate over all files in the directory
    for file_name in os.listdir(directory):
        # Get the full path of the file
        if type == "str":
            full_path = os.path.join(directory, file_name)
        if type == "posix_path":
            full_path = Path(os.path.join(directory, file_name))
        
        # Check if it is a file (and not a directory)
        if os.path.isfile(full_path):
            file_paths.append(full_path)

    return file_paths


# Test average_tiff_images function using 18 files
def test_average_tiff_images_with_path_as_str():
    path_to_folder = "/media/luis/TOMO/Dosepy/BQT_INCAN/Cal_Der/"

    # Colocar en una lista el path a los archivos en el folder
    path_to_files = get_file_paths(path_to_folder, type="str")


    all_images = []
    for path in path_to_files:
        all_images.append(image.load(path))

    images = average_tiff_images(all_images)

    # The number of files with different 
    # (excluding las 7 character) name is 9
    assert len(images) == 9


# Test average_tiff_images using path as pathlib.PosixPath
def test_average_tiff_images_with_path_as_PosixPath():
    path_to_folder = "/media/luis/TOMO/Dosepy/BQT_INCAN/Cal_Der/"

    path_to_files = get_file_paths(path_to_folder, type="posix_path")

    all_images = []
    for path in path_to_files:
        all_images.append(load(path))

    images = average_tiff_images(all_images)

    assert len(images) == 9


# Test width reduction with equate_array_size function
def test_equate_array_size():

    img1 = load(np.ones((6, 6)), dpi=1)
    img2 = load(np.ones((5, 5)), dpi=1)

    (new1, new2) = equate_array_size(
        image_list=[img1, img2],
        axis=("width")
        )

    assert new1.shape == (6, 5)


# Test height reduction with equate_array_size function
def test_equate_array_size_height():

    img1 = load(np.ones((6, 6)), dpi=1)
    img2 = load(np.ones((5, 5)), dpi=1)

    (new1, new2) = equate_array_size(
        image_list=[img1, img2],
        axis=("height")
        )

    assert new1.shape == (5, 6)


# Test height and width reduction with equate_array_size function
def test_equate_array_size_height_and_width():

    img1 = load(np.ones((8, 8)), dpi=1)
    img2 = load(np.ones((5, 5)), dpi=1)

    (new1, new2) = equate_array_size(
        image_list=[img1, img2],
        axis=("height", "width")
        )

    assert new1.shape == (5, 5)


# Test stack_images

def test_stack_images():

    img1 = load(np.ones((5, 5, 3)), dpi=1)
    img2 = load(np.ones((5, 5, 3)), dpi=1)

    img = stack_images([img1, img2])

    assert img.shape == (10, 5, 3)



