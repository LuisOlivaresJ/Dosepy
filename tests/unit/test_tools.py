import pytest
from pathlib import Path

from Dosepy.tools.files_to_image import (
    average_tiff_images,
    equate_array_size,
)
from Dosepy.image import load

import numpy as np

import os


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
        all_images.append(load(path))

    images = average_tiff_images(path_to_files, all_images)

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

    images = average_tiff_images(path_to_files, all_images)

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