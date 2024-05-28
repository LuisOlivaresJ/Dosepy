"""
Script that crops and equates the array size of several TIFF files.
"""

import os
import numpy as np
import imageio.v3 as iio

from Dosepy.image import load
from Dosepy.image import equate_images


def _find_smallest_image(images):

    min_higth = images[0].shape[0]
    min_width = images[0].shape[1]

    index_min_height = 0
    index_min_width = 0

    for count, img in enumerate(images[1:], start=1):

        if img.shape[0] < min_higth:
            min_higth = img.shape[0]
            index_min_height = count

        if img.shape[1] < min_width:
            min_width = img.shape[1]
            index_min_width = count

    return index_min_height, index_min_width


def _save_as_tif(file_names, images):

    new_folder = "cropped_files"
    os.mkdir(new_folder)

    for count, img in enumerate(images):

        file_path = os.path.join(os.getcwd(), new_folder, file_names[count])

        img_array = img.array.astype(np.uint16)
        tif_encoded = iio.imwrite(
            "<bytes>",
            img_array,
            extension = ".tif",
            resolution = (img.dpi, img.dpi)
        )
        with open (file_path, "wb") as f:
            f.write(tif_encoded)
            f.close()


def equate(path: str):
    """
    Equate several TIFF files to have the same array size. The new images are stored in a folder named "cropped_files".

    Parameters
    ----------
    folder : string
        Folder name with the TIFF files.
    """

    folder_path = path
    images = []

    for file in os.listdir(folder_path):
        print(os.path.join(folder_path, file))
        images.append(load(os.path.join(folder_path, file)))

    idx_min_height, idx_min_width =_find_smallest_image(images)

    for count, img in enumerate(images):
        if count in [idx_min_width, idx_min_height]: continue
        _, images[count] = equate_images(images[idx_min_height], img)

    _save_as_tif(os.listdir(folder_path), images)
