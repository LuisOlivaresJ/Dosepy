"""
Script that crops and equates the array size of several TIFF files.
"""

import os
import numpy as np
import imageio.v3 as iio
import copy
import math

from Dosepy.image import load
from Dosepy.image import equate_images, load_multiples


def _find_smallest_images(images):

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


def _save_as_tif(file_names, images, folder_path):

    new_folder = "cropped_files"
    path = os.path.join(folder_path, new_folder)
    os.mkdir(path)

    for count, img in enumerate(images):

        file_path = os.path.join(folder_path, new_folder, file_names[count])

        img_array = load(img.array.astype(np.uint16), dpi=img.dpi)
        img_array.save_as_tif(file_path)


def _equate_height(small_image, image):

    height_diff = abs(int(image.shape[0] - small_image.shape[0]))

    if height_diff > 0:
                
        if height_diff == 1:
            image.crop(height_diff, edges="botton")

        elif not(height_diff%2):
            image.crop(int(height_diff/2), edges=('bottom', 'top'))

        else:
            image.crop(int(math.floor(height_diff/2)), edges="top")
            image.crop(int(math.floor(height_diff/2) + 1), edges="bottom")

    return image

def _equate_width(small_image, image):

    width_diff = abs(int(image.shape[1] - small_image.shape[1]))

    if width_diff > 0:

        if width_diff==1:
            image.crop(width_diff, edges="right")

        elif not(width_diff%2):
            image.crop(int(width_diff/2), edges=("left", "right"))

        else:
            image.crop(int(math.floor(width_diff/2)), edges="left")
            image.crop(int(math.floor(width_diff/2) + 1), edges="right")

    return image

def equate(file_list: list, axis: tuple[str, ...] = ("height", "width"), save=False):
    """
    Equate several TIFF files to have the same array size of the samllest array.

    Parameters
    ----------
    file_list : list
        List with the paths of the TIFF files.

    axis : str
        Axis to equate: height or width

    save : bool
        True if we need to save the cutted images as tif files in the Home directory.

    Return
    ------
        A list with the new images.
    """

    images = []

    for file in file_list:
        images.append(load(file))
    
    cropped_images = copy.deepcopy(images)
    idx_min_height, idx_min_width =_find_smallest_images(images)
    print("Inside equate index")
    print(idx_min_height, idx_min_width)

    if "height" in axis:
        for count, img in enumerate(images):
            if count == idx_min_height: continue
            cropped_images[count] = _equate_height(images[idx_min_height], img)
    
    if "width" in axis:

        print("Inside equate")
        for count, img in enumerate(images):
            if count == idx_min_width: continue
            cropped_images[count] = _equate_width(images[idx_min_width], img)
            print(f"img shape: {img.shape}")
            print(f"cropped: {cropped_images[count].shape}")

    if save:
        _save_as_tif(file_list, cropped_images, os.path.expandvars("HOME"))

    return cropped_images


def merge(file_list, images):
    """
    Merge images with the same file name. Last 7 characters are not accounted.
    """
    film_list = list(set([file[:-7] for file in file_list]))
    img_list = []

    for film in film_list:
        merge_list =[]
        first_img = copy.deepcopy(images[0])  # Placeholder
        for file, image in zip(file_list, images):
            if file[:-7] == film:
                merge_list.append(image)
        
        new_array = np.stack(tuple(img.array for img in merge_list), axis=-1)
        combined_arr = np.mean(new_array, axis=3)
        first_img.array = combined_arr

        img_list.append(first_img)
    
    return img_list