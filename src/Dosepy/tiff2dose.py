from abc import ABC, abstractmethod

from skimage.filters.rank import median
from skimage.morphology import square
from skimage.measure import label, regionprops

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from Dosepy.calibration import LUT, MM_PER_INCH
from Dosepy.image import TiffImage
from Dosepy.tools.functions import optical_density, uncertainty_optical_density

import math


class Tiff2DoseM:
    """
    Converts a tiff image to dose map.

    This class implements the Factory Method Pattern.
    """
    def get_dose(
            img: TiffImage,
            format: str,
            lut: LUT
            ):
        dose_converter = dose_converter_factory.get_dose_converter(format)
        return dose_converter.get_dose(img, lut)
    


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

            position_ceil = math.ceil(pix_position)
            #position_floor = math.floor(pix_position)

            # Get calibration intensities and calibration doses at pixel positions
            ## Get ceil and floor values to interpolate
            cal_intensities_ceil, _ = self.lut._get_intensities(
                lateral_position = position_ceil,
                channel = "red",
            )
            cal_intensities_floor, _ = self.lut._get_intensities(
                lateral_position = rounded_floor_position,
                channel = "red",
            )
            cal_doses_ceil = self.lut._get_lateral_doses(position=position_ceil)
            cal_doses_floor = self.lut._get_lateral_doses(position = rounded_floor_position)

            ## Interpolate values
            interp_intensities = [
                np.interp(
                    pix_position,
                    [rounded_floor_position, position_ceil],
                    [cal_intensities_floor[i], cal_intensities_ceil[i]],
                )
                for i in range(len(cal_intensities_floor))
            ]
            
            interp_doses = [
                np.interp(
                    pix_position,
                    [rounded_floor_position, position_ceil],
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
    def get_dose(img, lut):
        pass


    def create_positions(self, img):
        # Create a list with lateral positions in milimeters, with the center of the image as the origin.
        origin = img.physical_shape[1]/2
        self.pixel_positions_mm = np.linspace(
            start = 0,
            stop = img.physical_shape[1],
            num = img.array.shape[1]
            ) - origin
        

    def apply_filter(self, array, lut):

        if lut.lut["filter"]:
            mask = "TODO"

            filtered_array =  median(
                array,
                footprint = square(lut.lut["filter"]),
                mask = mask,
                )
            
        else:
            filtered_array = array

        return filtered_array
    

    def check_optical_filters(img: TiffImage, lut: LUT):
        # TODO Obtener mean and std de la intensidad de los filtros al momento
        # de la calibración. Deberán almacenarse en LUT
        return True
    

    def get_zero_dose_intensity(img: TiffImage) -> int:
        pass



class RedPolynomialDoseConverter(DoseConverter):
    
    def get_dose(self, img: TiffImage, lut: LUT):

        # TODO LUT does not have data for no-latera-correction
        if not lut.lut["lateral_correction"]:
            doses = lut.lut["nominal_doses"]
            mean_intensity, std = img.get_stat(ch = "R", roi = (5,5), show=True)

            film_response = optical_density(img.array[:, :, 0], mean_intensity[0])
            # Get popt
            #dose_image = polynomial_g3(x, *cal.popt)
            # Get sorted intensities from unexposed (high intensity) to exposed (low intensity) doses.
            mean_intensity = sorted(mean_intensity, reverse=True)
            return "TODO"
        
        # Crear una lista con las posiciones en mm
        self.create_positions(img)

        # Aplicar filtro si lut se creo con filtro
        if lut.lut["filter"]:
            red = self.apply_filter(img.array[:, :, 0], lut)

        # Check that mean intensity of filters are equal (considering standar deviation)
        ## Get mask for films and filters
        img.set_labeled_films_and_filters()  ## TODO use cache or something to check if it is aldready calculated
        if not self.check_optical_filters(img):
            print("The mean intensity of the filters are not equal.")

        # Convertir cada pixel a dosis
        ## Obtener intensidad a cero Grays
        zero_dose_intensity = self.get_zero_dose_intensity(img)
        ### En img, identificar film con cero grays
        ### Revisar film contiene el centro de la imagen
        ### Obtener intensidad en el centro del scaner, en el film zero
        
        print("TODO")
        return "Dose"


    def get_zero_dose_intensity(img: TiffImage) -> int:
        # En img, identificar film con cero grays
        
        properties = regionprops(img.labeled_films, intensity_image = img.array[:, :, 0])
        zero_film_index = None
        zero_dose_intensity = None
        for n, p in enumerate(properties, start = 1):
            if p.intensity_mean > zero_dose_intensity:
                zero_dose_intensity = p.intensity_mean
                zero_film_index = n - 1

        zero_film_properties = properties[zero_film_index]

        # Revisar si film contiene el centro de la imagen
        min_row, min_col, max_row, max_col = zero_film_properties.bbox
        center_x, center_y = img.center()

        if max_col < center_y or min_col > center_y:
            print("The film is not in the center of the scaner.")
            return False

        # Obtener intensidad en el centro del scaner, en el film zero
        ## crear box con la imagen
        x0, y0 = zero_film_properties.centroid
        min_lenght = 0.5 * zero_film_properties.axis_minor_length
        min_row_roi = int(x0 - min_lenght*0.9)
        max_row_roi = int(x0 + min_lenght*0.9)

        ## Get the mean intensity
        median_intensity = np.median(
            img[min_row_roi : max_row_roi, y0, 0])
        
        roi = {
            "min_row": min_row_roi,
            "max_row": max_row_roi,
            "pixel_position": y0 
        }
        
        return median_intensity, roi


dose_converter_factory = Tiff2DoseFactory()
dose_converter_factory.register_method("RP", RedPolynomialDoseConverter)

