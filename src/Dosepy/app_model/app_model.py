"""Functions used as a model. VMC pattern."""

from Dosepy.image import _is_tif_file, load, load_multiples, ImageLike, TiffImage, ArrayImage
import imageio.v3 as iio
import numpy as np
from numpy import ndarray
from importlib import resources
import pickle
from Dosepy.config.io_settings import Settings, load_settings
from Dosepy.calibration import LUT


class Model:
    """
    This class is used to store main data for film dosimetry like tif images and 
    lut file for calibration.
    Also, there are methods to open tif files, to ask for correct tif files,
    save or load a lut.
    """
    def __init__(self):
        self.calibration_img: TiffImage = None  # The image used to produce a calibration curve
        self.tif_img: TiffImage = None  # The tif image to be analysed
        self.lut: LUT = None  # The calibration object used for tif to dose calculation
        self.dose_img_from_film = ArrayImage  # The dose distribution calculated from a tiff file

        self.config: Settings = load_settings()  # The settings for the application.
        
        self.ct_array_img: ndarray = None  # The CT image to be used for user mark localization
        # Index of the user's mark in the CT image.
        # Row as +y, column as +x, slice as +z. The same as DICOM convention.
        self.ct_index: list[int, int, int] = None
        # Aspect ratio of the CT image.
        self.ct_aspect: dict[str, float] = None


    def are_valid_tif_files(self, files: list) -> bool:
        return all([_is_tif_file(file) for file in files])
        

    def are_files_equal_shape(self, files: list) -> bool:
        first_img_shape = self.props = iio.improps(files[0]).shape
        for file in files:
            if iio.improps(file).shape != first_img_shape:
                return False
        return True


    def load_files(self, files: list) -> ImageLike:
        # TODO Perform input validation
        return load_multiples(files)
    
    
    def save_lut(self, file_path: str):
        self.lut.to_yaml_file(file_path)


    def load_lut(self, lut_path: str):     
        return LUT.from_yaml_file(lut_path)
    

    def save_dose_as_tif(self, file_name: str):
        data_array = self.dose_img_from_film.array*100  # Gy to cGy
        data = data_array.astype(np.uint16)
        img = load(data, dpi = self.dose_img_from_film.dpi)
        img.save_as_tif(file_name)

