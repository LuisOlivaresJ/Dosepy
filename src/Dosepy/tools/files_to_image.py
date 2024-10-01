"""
Script that crops and equates the array size of several TIFF files.
"""

import os
import numpy as np
import imageio.v3 as iio
import copy
import math
import logging

#from Dosepy.image import load

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.WARNING,
    filename="files_to_image.log",
    )


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

    logging.debug(f"Smallest height: {min_higth} at index {index_min_height}")
    logging.debug(f"Smallest width: {min_width} at index {index_min_width}")
    return index_min_height, index_min_width

"""
def _save_as_tif(file_names, images, folder_path):

    new_folder = "cropped_files"
    path = os.path.join(folder_path, new_folder)
    os.mkdir(path)

    for count, img in enumerate(images):

        file_path = os.path.join(folder_path, new_folder, file_names[count])

        img_array = load(img.array.astype(np.uint16), dpi=img.dpi)
        img_array.save_as_tif(file_path)
"""


def _equate_height(small_image, image):
    """
    Crop the image to have the same height as the small_image. 
    If the difference is odd, the extra pixel is cropped from the top.
    Otherwise, the extra pixels are cropped equally from both sides.
    """
    logging.debug(f"Image height before cropping: {image.shape}")
    height_diff = abs(int(image.shape[0] - small_image.shape[0]))
    logging.debug(f"Height difference: {height_diff}")

    if height_diff > 0:
                
        if height_diff == 1:
            image.crop(height_diff, edges="bottom")

        elif not(height_diff%2):
            image.crop(int(height_diff/2), edges=('bottom', 'top'))

        else:
            image.crop(int(math.floor(height_diff/2)), edges="top")
            image.crop(int(math.floor(height_diff/2) + 1), edges="bottom")


    logging.debug(f"Image height after cropping: {image.shape}")
    return image


def _equate_width(small_image, image):
    """
    Crop the image to have the same width as the small_image.
    If the difference is odd, the extra pixel is cropped from the left.
    Otherwise, the extra pixels are cropped equally from both sides.
    """

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


def equate_array_size(
        image_list: list,
        axis: tuple[str, ...] = ("height", "width"),
        ) -> list:
    """
    Equate TIFF files to have the same array size with respect of the smallest one.

    Parameters
    ----------
    image_list : list
        List with images (TiffImage, ArrayImage instance).

    axis : str
        Axis to equate: height, width or both.

    Return
    ------
        A list with the new images.
    """
    
    cropped_images = copy.deepcopy(image_list)
    idx_min_height, idx_min_width = _find_smallest_image(image_list)

    if "height" in axis:
        for count, img in enumerate(image_list):
            if count == idx_min_height: continue
            cropped_images[count] = _equate_height(image_list[idx_min_height], img)
            
    image_list = cropped_images
    if "width" in axis:

        for count, img in enumerate(image_list):
            if count == idx_min_width: continue
            cropped_images[count] = _equate_width(image_list[idx_min_width], img)

    return cropped_images


def merge(file_list: list, images: list) -> list:
    """
    Merge images with the same file name. Last 7 characters are not accounted.

    Parameters
    ----------
    file_list : list
        list of strings with the tiff file path.

    images : list
        list of TiffImage

    Return
    ------
    img_list
        list of TiffImage
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


def stack_images(img_list, axis=0, padding=0):
    """
    Takes in a list of images and concatenate them side by side.
    Useful for film calibration, when more than one image is needed
    to scan all gafchromic bands.
    
    Adapted from OMG_Dosimetry (https://omg-dosimetry.readthedocs.io/en/latest/)

    Parameters
    ----------
    img_list : list
        The images to be stacked. List of TiffImage objects.

    axis : int, default: 0
        The axis along which the arrays will be joined. 0 if vertical or 1 if horizontal.

    padding : float, default: 0
        Add padding in milimeters to simulate an empty space betwen films.

    Returns
    -------
    ::class:`~Dosepy.image.TiffImage`
        Instance of a TiffImage class.

    """

    first_img = copy.deepcopy(img_list[0])

    # Check that all images are the same width
    for img in img_list:
        
        if axis == 0:
            if img.shape[1] != first_img.shape[1]:
                raise ValueError("Images were not the same width")
        if axis == 1:
            if img.shape[0] != first_img.shape[0]:
                raise ValueError("Images were not the same height")

    #height = first_img.shape[0]
    width = first_img.shape[1]

    padding_pixels = int(padding * img_list[0].dpmm)

    new_img_list = []
    
    for img in img_list:

        height = img.shape[0]

        background = np.zeros(
            (2*padding_pixels + height, 2*padding_pixels + width, 3)
            ) + int(2**16 - 1)

        background[
            padding_pixels: padding_pixels + height,
            padding_pixels: padding_pixels + width,
            :
            ] = img.array
        new_img = copy.deepcopy(img)
        new_img.array = background
        new_img_list.append(new_img)
    
    new_array = np.concatenate(tuple(img.array for img in new_img_list), axis)
    first_img.array = new_array

    return first_img