"""Functions used as a model. VMC pattern."""

#from Dosepy.tools.image import _is_RGB
from image import _is_RGB, _is_image_file, load_multiples, load, ImageLike
import imageio.v3 as iio
import numpy as np
from importlib import resources
import os
from pathlib import Path

class Model:
    def __init__(self):
        self.calibration_img = None

    def are_valid_tif_files(self, files: list) -> bool:
        return all([_is_image_file(file) and _is_RGB(file) for file in files])
        

    def are_files_equal_shape(self, files: list) -> bool:
        first_img_shape = self.props = iio.improps(files[0]).shape
        for file in files:
            if iio.improps(file).shape != first_img_shape:
                return False
        return True
    
    def load_files(self, files: list, for_calib=False) -> ImageLike:
        if len(files) == 1:
            return load(files[0], for_calib=for_calib)
        
        elif len(files) > 1:
            return load_multiples(files, for_calib=for_calib)
    

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

        