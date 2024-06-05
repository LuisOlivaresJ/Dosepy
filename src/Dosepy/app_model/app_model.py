"""Functions used as a model. VMC pattern."""

from Dosepy.image import _is_RGB, _is_image_file, load, ImageLike
from Dosepy.tools.files_to_image import equate, merge, load_images, stack_images
import imageio.v3 as iio
import numpy as np
from importlib import resources
import pickle

class Model:
    """
    This class is used to store main data for film dosimetry like tif images and 
    lut file for calibration.
    Also, there are methods to open tif files, to ask for correct tif files,
    save or load a lut.
    """
    def __init__(self):
        self.calibration_img = None  # The image used to produce a calibration curve
        self.tif_img = None  # The tif image to be analysed
        self.lut = None  # The calibration object used for tif to dose calculation
        self.ref_dose_img = None  # The reference dose distribution (usally calculated from a tif file)

    def are_valid_tif_files(self, files: list) -> bool:
        return all([_is_image_file(file) and _is_RGB(file) for file in files])
        

    def are_files_equal_shape(self, files: list) -> bool:
        first_img_shape = self.props = iio.improps(files[0]).shape
        for file in files:
            if iio.improps(file).shape != first_img_shape:
                return False
        return True
    
    def load_calib_files(self, files: list) -> ImageLike:

        if len(files) == 1:

            return load(files[0], for_calib=True)
        
        elif len(files) > 1:

            img = load(files[0], for_calib=True) # Placeholder
            images = load_images(files)
            equated_images = equate(images, axis=("width", "height"))
            merged_images = merge(files, equated_images)
            stacked = stack_images(merged_images, padding=6)
            img.array = stacked.array

            return img


    def load_files(self, files: list) -> ImageLike:
        if len(files) == 1:
            return load(files[0])
        
        else:
            img = load(files[0]) # Placeholder
            images = load_images(files)
            equated_images = equate(images, axis=("width", "height"))
            merged_images = merge(files, equated_images)
            stacked = stack_images(merged_images, padding=6)
            img.array = stacked.array
            return img


    def create_dosepy_lut(self, doses, roi):
        #channel = ["m", "r", "g", "b"]
        doses = np.array(doses)
        lut = np.zeros([6, len(doses)])
        lut[0,:] = doses
        lut[1,:] = doses * 1  # Correct doses for machine daily output
        lut[2,:], _ = np.array(
            self.calibration_img.get_stat(
                ch="m",
                roi=roi,
                show=False,
                threshold=None
                )
            )
        lut[3,:], _ = np.array(
            self.calibration_img.get_stat(
                ch="r",
                roi=roi,
                show=False,
                threshold=None
               )
            )
        lut[4,:], _ = np.array(
            self.calibration_img.get_stat(
                ch="g",
                roi=roi,
                show=False,
                threshold=None
                )
            )
        lut[5,:], _ = np.array(
            self.calibration_img.get_stat(
                ch="b",
                roi=roi,
                show=False,
                threshold=None
                )
            )

        return lut
    
    def save_lut(self, file_path: str):
        
        file = open(file_path + ".cal", 'wb')
        pickle.dump(self.lut, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def load_lut(self, lut_path: str):
        file = open(lut_path, 'rb')
        lut = pickle.load(file)
        file.close()
        return lut
    
    def save_dose_as_tif(self, file_name: str):
        data_array = self.ref_dose_img.array*100  # Gy to cGy
        data = data_array.astype(np.uint16)
        img = load(data, dpi = self.ref_dose_img.dpi)
        img.save_as_tif(file_name)

