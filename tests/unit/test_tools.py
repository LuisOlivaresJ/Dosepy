import pytest
from pathlib import Path

from Dosepy.tools.files_to_image import average_tiff_images
from Dosepy.image import load

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