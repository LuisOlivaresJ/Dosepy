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
import gdown

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

# One film per file, 2 films per dose
cal_files = {
    "tests/unit/fixtures/tiff_files/individual/Cal200_D001.tif": "1wfTgGyAaW_jtoeXH07meOeS9gMMcRzxH",
    "tests/unit/fixtures/tiff_files/individual/Cal200_D002.tif": "1bDKaWkrQMvVSHrJ-3lY0EujO0wpuBmjt",
    "tests/unit/fixtures/tiff_files/individual/Cal400_D001.tif": "1ZXqNXonmsyGS3VjieSdMGJhmAos-iTPY",
    "tests/unit/fixtures/tiff_files/individual/Cal400_D002.tif": "1tvIooa5-1WjKrUct2BOBIau05D3BRKVl",
    "tests/unit/fixtures/tiff_files/individual/Cal500_D001.tif": "1RAMTOoWo4Lput1foHw98v-N7_JhliM8p",
    "tests/unit/fixtures/tiff_files/individual/Cal500_D002.tif": "1H-i3Sn2TDQNlUzsRvsU7Ro7fo27V1x3-",
    "tests/unit/fixtures/tiff_files/individual/Cal600D001.tif": "1VtztK-FgRlPUPVKXwJOXWYAWRJGQ_eQq",
    "tests/unit/fixtures/tiff_files/individual/Cal600D002.tif": "12ljGw-Rk63ITiSRzZdai1QifCXH8sFcw",
    "tests/unit/fixtures/tiff_files/individual/Cal700D001.tif": "10M6gvlf3NRjem--mzinEDnR7B3SReYlC",
    "tests/unit/fixtures/tiff_files/individual/Cal700D002.tif": "1qdFrlp6yf_4gEu4C5kOM-RtzXo3X990o",
    "tests/unit/fixtures/tiff_files/individual/Cal800D001.tif": "1jBS_-UTNA90wfeb3MJFlUfkUb7AMboeO",
    "tests/unit/fixtures/tiff_files/individual/Cal800D002.tif": "1lm66v8Q0GqGRMaoHTgt0JWTvfe1zaElz",
    "tests/unit/fixtures/tiff_files/individual/Cal900D001.tif": "1acU_DuKxLbbDpAORcoWpBQy77rNCc1rj",
    "tests/unit/fixtures/tiff_files/individual/Cal900D002.tif": "1a3PUzHcbCCuNbxR7syxjAgW2IHQRbs-2",
    "tests/unit/fixtures/tiff_files/individual/Cal1000D001.tif": "1Qh50WtJNowXOhQHb4yejoqR18io8l9jr",
    "tests/unit/fixtures/tiff_files/individual/Cal1000D002.tif": "12iYrg61OvMHFePpzk6RXx3hdrtxLLAj3",
    "tests/unit/fixtures/tiff_files/individual/sumergidaD001.tif": "1rRf9jYf4a97XpMan6esq4a29WAbU_Qou",
    "tests/unit/fixtures/tiff_files/individual/sumergidaD002.tif": "14dfZsRcqKdpZ16QDpHnLgZupcOL_sJdl",
}
os.makedirs("tests/unit/fixtures/tiff_files/individual/", exist_ok=True)
for filename, file_id in cal_files.items():
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} aldready exists.")

ver_files = [file for file in cal_files.keys()]



# Test average_tiff_images function using 18 files
def test_average_tiff_images_with_path_as_str():

    path_to_folder = "tests/unit/fixtures/tiff_files/individual/"

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
    
    path_to_folder = "tests/unit/fixtures/tiff_files/individual/"

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


# Test get_optical_filters
def test_get_optical_filters():

    img = load(cwd / "fixtures" / "Ver_050dpi20241106_001.tif")

    optical_filters = img.get_optical_filters()
    intensities = optical_filters["intensities_of_optical_filters"]

    assert all(np.isclose(
        intensities,
        [9335, 13275, 20734],
        rtol=1e-2,
    ))
