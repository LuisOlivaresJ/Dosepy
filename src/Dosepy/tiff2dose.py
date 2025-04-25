from abc import ABC, abstractmethod

from skimage.filters.rank import median
from skimage.morphology import square
from skimage.measure import label, regionprops

import numpy as np
from numpy import ndarray
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from Dosepy.calibration import LUT, MM_PER_INCH
from Dosepy.image import TiffImage, ArrayImage
from Dosepy.tools.functions import (
    optical_density,
    rational_function,
    uncertainty_optical_density,
    polynomial_n,
    ratio,
)

import math

import logging

logger = logging.getLogger(__name__)


# Define t2d method for tiff to dose conversion
T2D_METHOD_MAP = {
    ("red", "polynomial"): "RP",
    ("green", "polynomial"): "GP",
    ("blue", "polynomial"): "BP",
    ("red", "rational"): "RR",
    ("green", "rational"): "GR",
    ("blue", "rational"): "BR"
}

CHANNEL_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
}


class Tiff2DoseM:
    """
    Class used as manager for tiff to dose convertion.
    Methods
    -------
    get_dose(img: TiffImage, format: str, lut: LUT)
        Get the dose array from a tiff image.
    
    Notes
    -----
    This class implements the Factory Method Pattern. The get_dose() method
    uses the Tiff2DoseFactory to get the correct dose converter.
    """

    def __init__(self):
        self.dose_converter = None

    def _set_dose_converter(self, format: str):
        self.dose_converter = dose_converter_factory.get_dose_converter(format)

    def get_dose(
            self,
            img: TiffImage,
            format: str,
            lut: LUT
            ) -> ArrayImage:
        """
        Get the dose array from a tiff image.

        Parameters
        ----------
        img : TiffImage
            The tiff image to convert to dose.
        format : str
            The channel and fit function to use for the conversion.
            "RP" for red channel and polynomial fit function of the
            form y = a*x + b*x**n, where a, b and n are the fit coefficients.
        lut : LUT
            The look up table with the calibration data.

        Returns
        -------
        ArrayImage
            The dose distribution
        """
        self._set_dose_converter(format)
        return self.dose_converter.convert2dose(img, lut)
    

class Tiff2DoseFactory:
    """Used to manage tiff to dose converters."""
    def __init__(self):
        self._converters = {}
    
    def register_method(self, format, converter):
        self._converters[format] = converter

    def get_dose_converter(self, format):
        converter = self._converters.get(format)
        if not converter:
            print("Invalid format.")
            raise ValueError(format)
        return converter()


class DoseConverter(ABC):
    """
    An abstract class used to create DoseConverter classes.

    Note: The TiffImage should have a reference film with zero dose.
    """

    def __init__(self):
        self.pixel_positions_mm = None

    
    @abstractmethod
    def convert2dose(self, img: TiffImage, lut: LUT) -> ArrayImage:
        pass
    

    def check_optical_filters(self, img: TiffImage, lut: LUT):
        # TODO Obtener mean and std de la intensidad de los filtros al momento
        # de la calibración. Deberán almacenarse en LUT
        return True
    

    def _set_lateral_positions(self, img):
        """
        Create an array with lateral positions in milimeters, with
        the center of the image as the origin.
        """
        origin = img.physical_shape[1]/2
        
        self.pixel_positions_mm = np.linspace(
            start = 0,
            stop = img.physical_shape[1],
            num = img.array.shape[1]
            ) - origin


    def _get_zero_dose_intensity(
            self,
            img: TiffImage,
            channel: str,
            at_zero_position: bool = False,
            ) -> tuple[int, dict]:
        """
        Get the intensity of the zero dose film.

        The algorithm finds the film with the highest intensity, 
        then checks if the film is in the center of the scanner.

        Returns the mean intensity of the film and a dictionary 
        with the region of interest used to get a median of the intensities.
        """
        
        if channel == "red":
            intensity_array = img.array[:, :, 0]
        elif channel == "green":
            intensity_array = img.array[:, :, 1]
        elif channel == "blue":
            intensity_array = img.array[:, :, 2]

        properties = regionprops(img.labeled_films, intensity_image = intensity_array)
        
        # Find film with the highest intensity
        zero_film_index = None
        zero_dose_intensity = 0
        for n, p in enumerate(properties):
            if p.intensity_mean > zero_dose_intensity:
                zero_dose_intensity = p.intensity_mean
                zero_film_index = n

        zero_film_properties = properties[zero_film_index]

        # Get the median intensity of the film at the center of the image
        if at_zero_position:

            # Is the film in the center of the image?
            min_row, min_col, max_row, max_col = zero_film_properties.bbox
            img_center_column, img_center_row = img.center()

            if not min_col < img_center_column < max_col:
                print("The film is not in the center of the scaner.")
                print(f"{min_col=}, {img_center_column=}, {max_col=}")
                raise ValueError("The film is not in the center of the scaner.")
            
            # Center of the film
            x0, y0 = zero_film_properties.centroid

            # ROI height of 60% of the minor axis length
            min_lenght = 0.6 * zero_film_properties.axis_minor_length
            min_row_roi = int(x0 - min_lenght / 2)
            max_row_roi = int(x0 + min_lenght / 2)

            ## Get the median intensity
            median_intensity = np.median(
                intensity_array[
                    min_row_roi : max_row_roi,
                    int(img_center_column) - 1 : int(img_center_column) + 1
                    ]
                )
            
            roi = {
                'x': min_row_roi,
                'y': int(img_center_column) - 1,
                'width': 3,
                'height': min_lenght 
            }
        
        # Get median intensity of a ROI of 30% of the film size
        else:  

            width_roi = int( 0.3 * (zero_film_properties.bbox[3] - zero_film_properties.bbox[1]))
            height_roi = int( 0.3 * (zero_film_properties.bbox[2] - zero_film_properties.bbox[0]))
            x = int(zero_film_properties.centroid[0] - height_roi/2)
            y = int(zero_film_properties.centroid[1] - width_roi/2)

            # Print ROI
            print(f"ROI of 30% size for zero dose:")
            print(f"{x=}, {y=}, {width_roi=}, {height_roi=}")

            median_intensity = np.median(
                intensity_array[
                    x : x + height_roi,
                    y : y + width_roi
                ]
            )

            roi = {
                'x': x,
                'y': y,
                'width': width_roi,
                'height': height_roi 
            }
        
            print(f"Median intensity of zero dose film: {median_intensity}")

        return median_intensity, roi
    

    def _get_lateral_intensities_for_zero_dose(
            self,
            img: TiffImage,
            lut: LUT,
            channel: str,
        ) -> tuple[ndarray: ndarray]:
        """
        Get the lateral intensities of the zero dose film that was
        scanned togheter with the film that is going to be converted to dose.

        The algorithm uses normalized intensities of the zero dose film used for calibration.
        """

        # Get intensities of the films used for calibration
        intensities, std = lut.get_intensities(0, "red")

        # Get index of higest intensity (unexposed film)
        index = np.argmax(intensities)

        # Get lateral intensities for zero dose film used for calibration
        lateral_intensities, std, positions = lut.get_lateral_intensity(index, "red")

        # Get intensity at 0 position
        origin_index = np.argwhere(positions == 0)[0]
        
        reference_intensity = lateral_intensities[origin_index]
        
        # Normalize intensities
        relative_intensities = lateral_intensities/reference_intensity

        # Get the intensity of the unexposed film that was scanned togheter 
        # with the film to be converted to dose
        zero_dose_intensity_at_center, roi = self._get_zero_dose_intensity(
            img,
            channel,
            at_zero_position = True,
            )

        # Compute lateral intensities for film with zero dose
        zero_dose_intensities = zero_dose_intensity_at_center * relative_intensities

        return zero_dose_intensities, positions


class PolynomialDoseConverter(DoseConverter):
    """
    Base class to convert a tiff image to a dose array using a polynomial fit function.
    This class implements the DoseConverter interface.
    """

    def __init__(self, channel: str, p0: list):
        super().__init__()
        self.channel = channel
        self.p0 = p0

    def convert2dose(self, img: TiffImage, lut: LUT) -> ArrayImage:
        
        # Film detection
        img.set_labeled_films_and_filters()

        if lut.lut["filter"]:
            img.filter_channel(lut.lut["filter"], channel=self.channel)


        # Without lateral correction
        if not lut.lut["lateral_correction"]:
            calibration_intensities, std = lut.get_intensities(
                lateral_position=0,
                channel=self.channel
            )
            calibration_doses = lut.lut["nominal_doses"]

            cal_responses = optical_density(
                calibration_intensities,
                calibration_intensities[0]
            )

            popt, pcov = curve_fit(
                polynomial_n,
                cal_responses,
                calibration_doses,
                p0=self.p0,
            )

            median_intensity_zero_dose, _ = self._get_zero_dose_intensity(
                img, self.channel, at_zero_position=False
            )

            film_response = optical_density(img.array[:, :, CHANNEL_IDX[self.channel]], median_intensity_zero_dose)

            dose_array = polynomial_n(film_response, *popt)

        # With lateral correction
        else:
            self._set_lateral_positions(img)

            if not self.check_optical_filters(img, lut):
                print("TODO")

            lateral_intensities_for_zero_dose, positions = self._get_lateral_intensities_for_zero_dose(
                img, lut, channel=self.channel
            )

            dose_array = np.empty(img.array[:, :, CHANNEL_IDX[self.channel]].shape)
            width_in_pixels = img.shape[1]

            # Convert each column of the image to dose
            for column in range(0, width_in_pixels):
                pix_position = self.pixel_positions_mm[column]

                rounded_floor_position = math.floor(pix_position)
                rounded_ceil_position = math.ceil(pix_position)

                if rounded_floor_position <= lut.lut["lateral_limits"]["left"] or rounded_ceil_position >= lut.lut["lateral_limits"]["right"]:
                    dose_array[:, column] = 0
                    continue

                calibration_intensities = lut.get_interpolated_intensities_at_position(
                    position=pix_position,
                    channel=self.channel,
                )

                calibration_doses = lut.get_interpolated_doses_at_position(
                    position=pix_position,
                )

                cal_responses = optical_density(calibration_intensities, calibration_intensities[0])

                popt, pcov = curve_fit(
                    polynomial_n,
                    cal_responses,
                    calibration_doses,
                    p0=self.p0,
                )
                #logger.info(f"{popt=} - {self.channel=}")

                idx_pixel_position = np.argwhere(positions == rounded_floor_position)

                film_response = optical_density(
                    img.array[:, column, CHANNEL_IDX[self.channel]],
                    lateral_intensities_for_zero_dose[idx_pixel_position]
                )

                dose_array[:, column] = polynomial_n(film_response, *popt)

        # Remove unphysical values
        dose_array[dose_array < 0] = 0
        dose_array[np.isnan(dose_array)] = 0

        # Limit maximum dose to 1.3 times the maximum dose used for calibration
        max_dose = lut.lut.get("nominal_doses")[-1] * 1.3
        dose_array[dose_array > max_dose] = max_dose

        return ArrayImage(dose_array, dpi=img.dpi)


class RationalDoseConverter(DoseConverter):
    def __init__(self, channel: str, p0: list):
        super().__init__()
        self.channel = channel
        self.p0 = p0

    def convert2dose(self, img: TiffImage, lut: LUT):
        img.set_labeled_films_and_filters()

        if lut.lut["filter"]:
            img.filter_channel(lut.lut["filter"], channel=self.channel)

        if not lut.lut["lateral_correction"]:
            calibration_intensities, std = lut.get_intensities(
                lateral_position=0,
                channel=self.channel
            )

            calibration_doses = lut.lut["nominal_doses"]

            cal_responses = ratio(
                calibration_intensities,
                calibration_intensities[0]
                )
            
            popt, pcov = curve_fit(
                rational_function,
                cal_responses,
                calibration_doses,
                p0 = self.p0
            )

            median_intensity_zero_dose, _ = self._get_zero_dose_intensity(
                img,
                self.channel,
                at_zero_position=False
            )

            film_response = ratio(
                img.array[:, :, CHANNEL_IDX[self.channel]],
                median_intensity_zero_dose
            )

            dose_array = rational_function(film_response, *popt)
        
        else:
            self._set_lateral_positions(img)

            if not self.check_optical_filters(img, lut):
                print("TODO")

            lateral_intensities_for_zero_dose, positions = self._get_lateral_intensities_for_zero_dose(
                img, lut, channel=self.channel
            )

            dose_array = np.empty(img.array[:, :, CHANNEL_IDX[self.channel]].shape)
            width_in_pixels = img.shape[1]

            # Convert each column of the image to dose
            for column in range(0, width_in_pixels):
                pix_position = self.pixel_positions_mm[column]

                rounded_floor_position = math.floor(pix_position)
                rounded_ceil_position = math.ceil(pix_position)

                if rounded_floor_position <= lut.lut["lateral_limits"]["left"] or rounded_ceil_position >= lut.lut["lateral_limits"]["right"]:
                    dose_array[:, column] = 0
                    continue

                calibration_intensities = lut.get_interpolated_intensities_at_position(
                    position=pix_position,
                    channel=self.channel,
                )

                calibration_doses = lut.get_interpolated_doses_at_position(
                    position=pix_position,
                )

                cal_responses = ratio(calibration_intensities, calibration_intensities[0])

                popt, pcov = curve_fit(
                    rational_function,
                    cal_responses,
                    calibration_doses,
                    p0=self.p0,
                )
                #logger.info(f"{popt=} - {self.channel=}")

                idx_pixel_position = np.argwhere(positions == rounded_floor_position)

                film_response = ratio(
                    img.array[:, column, CHANNEL_IDX[self.channel]],
                    lateral_intensities_for_zero_dose[idx_pixel_position]
                )

                dose_array[:, column] = rational_function(film_response, *popt)

        # Remove unphysical values
        dose_array[dose_array < 0] = 0
        dose_array[np.isnan(dose_array)] = 0

        # Limit maximum dose to 1.3 times the maximum dose used for calibration
        max_dose = lut.lut.get("nominal_doses")[-1] * 1.3
        dose_array[dose_array > max_dose] = max_dose

        return ArrayImage(dose_array, dpi=img.dpi)



class RedPolynomialDoseConverter(PolynomialDoseConverter):
    def __init__(self):
        super().__init__(channel="red", p0=[11, 22, 2])


class GreenPolynomialDoseConverter(PolynomialDoseConverter):
    def __init__(self):
        super().__init__(channel="green", p0=[18, 26, 2])


class BluePolynomialDoseConverter(PolynomialDoseConverter):
    def __init__(self):
        super().__init__(channel="blue", p0=[45, 55, 2])


class RedRationalDoseConverter(RationalDoseConverter):
    def __init__(self):
        super().__init__(channel="red", p0=[0.1, 4, 5])


class GreenRationalDoseConverter(RationalDoseConverter):
    def __init__(self):
        super().__init__(channel="green", p0=[0.1, 10, 10])
    

class BlueRationalDoseConverter(RationalDoseConverter):
    def __init__(self):
        super().__init__(channel="blue", p0=[0.1, 21, 22])


dose_converter_factory = Tiff2DoseFactory()
dose_converter_factory.register_method("RP", RedPolynomialDoseConverter)
dose_converter_factory.register_method("GP", GreenPolynomialDoseConverter)
dose_converter_factory.register_method("BP", BluePolynomialDoseConverter)
dose_converter_factory.register_method("RR", RedRationalDoseConverter)
dose_converter_factory.register_method("GR", GreenRationalDoseConverter)
dose_converter_factory.register_method("BR", BlueRationalDoseConverter)

