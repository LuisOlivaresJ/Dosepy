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
from Dosepy.image import TiffImage
from Dosepy.tools.functions import optical_density, uncertainty_optical_density, polynomial_n

import math

# TODO This class will override the Tiff2Dose class
class Tiff2DoseM:
    """
    Class used to convert a tiff image to a dose array.
    Methods
    -------
    get_dose(img: TiffImage, format: str, lut: LUT)
        Get the dose array from a tiff image.
    
    Notes
    -----
    This class implements the Factory Method Pattern. The get_dose method
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
            ):
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
        numpy.ndarray
            The dose
        """
        self._set_dose_converter(format)
        return self.dose_converter.convert2dose(img, lut)
    


class Tiff2Dose:
    """
    Tiff to dose manager to convert a tiff image to a dose map.
    Attributes
    ----------
    img : TiffImage
        The tiff image to convert to dose.
    cal : CalibrationLUT
        The lut to use for curve calibration.
    
    """
    def __init__(self, img: TiffImage, lut: LUT, zero: TiffImage = None):
        """
        Parameters
        ----------
        img : TiffImage
            The tiff image to convert to dose.
        cal : CalibrationLUT
            The lut to use for curve calibration.
        zero : TiffImage
            The tiff image to use as a zero dose reference.
        """
        self.img = img
        self.lut = lut
        if not zero:
            self.zero_img = img
        else:
            self.zero_img = zero

        self.i0 = self._get_zero_dose_intensity(self.zero_img)



    def red(self, fit_function: str):

        #high_pixels = self.img.shape[0]
        width_pixels = self.img.shape[1]

        mask, num_films = self.img.get_labeled_image(
            erosion_pix=int(6*self.lut.lut["resolution"]/MM_PER_INCH)
            )

        if self.lut.lut["filter"]:
            img_array = median(
                self.img.array[:, :, 0],
                footprint = square(self.lut.lut["filter"]),
                mask = mask,
                )
        else:
            img_array = self.img.array[:, :, 0]

        # Buffer array to store dose from img
        dose = np.empty(img_array.shape)

        # Create a list with the lateral positions in milimeters
        origin = self.img.physical_shape[1]/2
        pixel_positions_mm = np.linspace(
            start = 0,
            stop = self.img.physical_shape[1],
            num = self.img.array.shape[1]
            ) - origin

        # Convert image to dose, one column at a time
        for column in range(0, width_pixels):

            # Get pixel positions rounded ceil and floor for interpolation
            pix_position = pixel_positions_mm[column]
            #print("####################")
            #print("Inside Tiff2Dose.red")
            #print(f"{self.lut.lut['lateral_limits']['left']=}")
            #print(f"{self.lut.lut=}")
            rounded_floor_position = math.floor(pix_position)
            print(f"{rounded_floor_position=}")
            if rounded_floor_position < self.lut.lut["lateral_limits"]["left"] or pix_position > self.lut.lut["lateral_limits"]["right"]:
                dose[:, column] = 0
                continue

            rounded_ceil_position = math.ceil(pix_position)
            #position_floor = math.floor(pix_position)

            # Get calibration intensities and calibration doses at pixel positions
            ## Get ceil and floor values to interpolate
            cal_intensities_ceil, _ = self.lut.get_intensities(
                lateral_position = rounded_ceil_position,
                channel = "red",
            )
            cal_intensities_floor, _ = self.lut.get_intensities(
                lateral_position = rounded_floor_position,
                channel = "red",
            )
            cal_doses_ceil = self.lut._get_lateral_doses(position=rounded_ceil_position)
            cal_doses_floor = self.lut._get_lateral_doses(position = rounded_floor_position)

            ## Interpolate values
            interp_intensities = [
                np.interp(
                    pix_position,
                    [rounded_floor_position, rounded_ceil_position],
                    [cal_intensities_floor[i], cal_intensities_ceil[i]],
                )
                for i in range(len(cal_intensities_floor))
            ]
            
            interp_doses = [
                np.interp(
                    pix_position,
                    [rounded_floor_position, rounded_ceil_position],
                    [cal_doses_floor[i], cal_doses_ceil[i]],
                )
                for i in range(len(cal_doses_floor))
            ]

            # Compute dose
            ## _get_dose_from_fit uses the first element of the array to normalize


        return False
    

    def _get_zero_dose_intensity(self, img: TiffImage) -> tuple[int, int]:
        """
        Get mean and standar deviation.

        Parameters
        ----------
        img : TiffImage
            The image to get the zero dose intensity from.
        
        Returns
        -------
        tuple[int, int]
            The mean and standard deviation of the zero dose intensity.
        
        """

        # Find films in the given image

        # Check that the center coordinate of the scaner is inside each film
        # discard films that does not include the center of the scanner

        # Discard films with area less than 400 mm^2

        # Discard films with an intensity TODO How to segment by color?

        # Order films by intensity in descending order

        # Get mean and standard deviation of the first film

        class ZeroDoseIntensity:

            def __init__(self, array: np.ndarray):
                self.array = array
                self.figure = plt.figure()
                self.figure.canvas.mpl_connect('key_press_event', self._on_key_press)


            def _plot_array(self):

                self.ax = self.figure.add_subplot(111)
                self.ax.imshow(self.array/np.max(self.array))
                self.ax.set_title("Select the film with zero dose")
                
                self.roi = RectangleSelector(
                    self.ax,
                    self._on_click,
                    useblit=True,
                    button=[1],
                    minspanx=5,
                    minspany=5,
                    spancoords='pixels',
                    interactive=True
                )

                plt.show()
                

            def _on_click(self, eclick, erelease):
                pass

        
            # Event handler to get the roi selection when the user press enter
            def _on_key_press(self, event):
                print("####################")
                print(event)
                if event.key == 'enter':
                    xmin, xmax, ymin, ymax = self.roi.extents

                    zero_dose_intensity = self.array[xmin: xmax, ymin: ymax]
                    plt.close(self.figure)
                    return zero_dose_intensity


        zero_intensity = ZeroDoseIntensity(img.array)
        zero_intensity._plot_array()



    def plot(dose, ax: plt.Axes = None, show: bool = True, **kwargs) -> None:
        """Plot the dose map.

        Parameters
        ----------
        ax : matplotlib.Axes instance
            The axis to plot the image to. If None, creates a new figure.
        show : bool
            Whether to actually show the image. Set to false when plotting
            multiple items.
        kwargs
            kwargs passed to plt.plot()
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(dose)
        if show:
            plt.show()


class Tiff2DoseFactory:
    """Used to manage tiff to dose converters."""
    def __init__(self):
        self._converters = {}
    
    def register_method(self, format, converter):
        self._converters[format] = converter

    def get_dose_converter(self, format):
        converter = self._converters.get(format)
        if not converter:
            raise ValueError(format)
        return converter()


class DoseConverter(ABC):
    """
    Abstract class to create DoseConverter classes.

    Note: The TiffImage should have a reference film with zero dose.
    """

    def __init__(self):
        self.pixel_positions_mm = None

    
    @abstractmethod
    def convert2dose(self, img: TiffImage, lut: LUT):
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


    def _get_zero_dose_intensity_at_center(
            self,
            img: TiffImage,
            channel: str,
            ) -> tuple[int, dict]:
        """
        Get the intensity of the zero dose film.

        The algorithm finds the film with the highest intensity, 
        then checks if the film is in the center of the scanner.

        Returns the mean intensity of the film and a dictionary 
        with the region of interest used to get a median of the intensities.
        """
        print("Inside RedPolynomialDoseConverter.get_zero_dose_intensity method")
        
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

        # Is the film in the center of the image?
        min_row, min_col, max_row, max_col = zero_film_properties.bbox
        img_center_x, img_center_y = img.center()

        if not min_col < img_center_y < max_col:
            print("The film is not in the center of the scaner.")
            raise ValueError("The film is not in the center of the scaner.")

        # Get the mean intensity of the film
        ## Create a region of interest (ROI) around
        x0, y0 = zero_film_properties.centroid
        # ROI height of 80% of the minor axis length
        min_lenght = 0.8 * zero_film_properties.axis_minor_length
        min_row_roi = int(x0 - min_lenght / 2)
        max_row_roi = int(x0 + min_lenght / 2)

        ## Get the mean intensity
        median_intensity = np.median(
            intensity_array[min_row_roi : max_row_roi, int(img_center_y)]
            )
        
        roi = {
            "min_row": min_row_roi,
            "max_row": max_row_roi,
            "pixel_position": int(y0) 
        }
        
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
        print(f"{origin_index=}")
        reference_intensity = lateral_intensities[origin_index]
        print(f"{reference_intensity=}")
        
        # Normalize intensities
        relative_intensities = lateral_intensities/reference_intensity

        # Get the intensity of the unexposed film that was scanned togheter 
        # with the film that is going to be converted to dose
        zero_dose_intensity_at_center, roi = self._get_zero_dose_intensity_at_center(
            img,
            channel,
            )

        # Compute lateral intensities for film with zero dose
        zero_dose_intensities = zero_dose_intensity_at_center * relative_intensities

        return zero_dose_intensities, positions


class RedPolynomialDoseConverter(DoseConverter):
    
    def convert2dose(self, img: TiffImage, lut: LUT) -> np.ndarray:

        # Without lateral correction
        if not lut.lut["lateral_correction"]:

            # Get calibration intensities and calibration doses at the center

            calibration_intensities, std = lut.get_intensities(
                lateral_position = 0,
                channel = "red",
            )
            calibration_doses = lut.lut["nominal_doses"]

            # Compute fit coefficients
            cal_responses = optical_density(calibration_intensities, calibration_intensities[0])
            popt, pcov = curve_fit(polynomial_n, cal_responses, calibration_doses)

            # Mean intensity of zero dose
            mean_intensity, std = img.get_stat(ch = "R", roi = (8, 8))
            mean_intensity = sorted(mean_intensity, reverse=True)

            # Compute film response as optical density
            film_response = optical_density(img.array[:, :, 0], mean_intensity[0])
            
            # Compute dose
            dose_array = polynomial_n(film_response, *popt)
        
        # With lateral correction
        else:
            # Create lateral positions in milimeters
            self._set_lateral_positions(img)

            # Apply filter if it was usesd for calibration
            if lut.lut["filter"]:
                img.filter_channel(lut.lut["filter"], channel="R")

            # TODO Check that mean intensity of filters are equal (considering standar deviation)
            ## Get mask for films and filters
            if img.labeled_films.size == 0 or img.labeled_optical_filters.size == 0:
                img.set_labeled_films_and_filters()

            if not self.check_optical_filters(img, lut):
                print("The mean intensity of the filters are not equal.")

            # Get lateral intensities for the unexposed film scanned with the film to be converted to dose
            lateral_intensities_for_zero_dose, positions = self._get_lateral_intensities_for_zero_dose(img, lut, channel="red")

            # Convert image to dose, one column at a time
            # Buffer array to store dose array
            dose_array = np.empty(img.array[:, :, 0].shape)
            width_in_pixels = img.shape[1]
            for column in range(0, width_in_pixels):

                # Get pixel position in milimeters
                pix_position = self.pixel_positions_mm[column]

                # Round pixel position to work inside LUT limits.
                rounded_floor_position = math.floor(pix_position)
                rounded_ceil_position = math.ceil(pix_position)
                if rounded_floor_position <= lut.lut["lateral_limits"]["left"] or rounded_ceil_position >= lut.lut["lateral_limits"]["right"]:
                    dose_array[:, column] = 0
                    continue

                # Get calibration intensities and calibration doses at pixel position

                calibration_intensities = lut.get_interpolated_intensities_at_position(
                    position = pix_position,
                    channel = "red",
                )
                calibration_doses = lut.get_interpolated_doses_at_position(
                    position = pix_position,
                )

                # Compute fit coefficients
                cal_responses = optical_density(calibration_intensities, calibration_intensities[0])
                popt, pcov = curve_fit(polynomial_n, cal_responses, calibration_doses)

                # index for pixel position
                idx_pixel_position = np.argwhere(positions == rounded_floor_position)

                # Compute film response as optical density
                film_response = optical_density(
                    img.array[:, column, 0],
                    lateral_intensities_for_zero_dose[idx_pixel_position]
                    )
                
                # Compute dose
                dose_array[:, column] = polynomial_n(film_response, *popt)

        # Remove unphysical values
        dose_array[dose_array < 0] = 0
        dose_array[np.isnan(dose_array)] = 0

        return dose_array


dose_converter_factory = Tiff2DoseFactory()
dose_converter_factory.register_method("RP", RedPolynomialDoseConverter)

