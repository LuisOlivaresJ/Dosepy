"""
NAME
    Calibration module

DESCRIPTION
    Module for the management of the calibration curve. Here are the functions
    to be used for fitting. See Calibration class for details.
"""

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import square, erosion
from skimage.measure import label, regionprops
from skimage.filters.rank import mean
from skimage.transform import rotate

from Dosepy.image import TiffImage


"""Functions used for film calibration."""


def polynomial_g3(x, a, b, c, d):
    """
    Polynomial function of degree 3.
    """
    return a + b*x + c*x**2 + d*x**3


def rational_func(x, a, b, c):
    """
    Rational function.
    """
    return -c + b/(x-a)

class CalibrationLUT:
    """
    Class used to represent a calibration curve.

    Attributes
    ----------
    film_response : list
        The response of the film to the doses.
    doses : list
        The doses values used to expose the films for calibration.
    dose_unit : str
        The unit of the dose.
    fit_function : str
        The model function used for dose-film response relationship.
        "P3": Polynomial function of degree 3.
        "RF": Rational function.
    channel : str
        Color channel. "R": Red, "G": Green and "B": Blue.
    """

    def __init__(self):
        self.film_response = []
        self.doses = []
        self.dose_unit = ""
        self.fit_function = ""
        self.channel = ""


    def compute_lut(self, img: TiffImage, doses: list, rois: list, channel: str):
        """
        Compute the look-up table (LUT) for the calibration curve.

        Parameters
        ----------
        img : TiffImage
            The image used for calibration.
        doses : list
            The doses values used to expose the films for calibration.
        rois : list
            The response of the film to the doses.
        channel : str
            Color channel. "R": Red, "G": Green and "B": Blue.
        """
        pass

    

    pass

class CalibrationLUT:
    """
    Class used to store data used for film calibration.
    This class is hevily inspired by the LUT class from OMG Dosimetry package
    (https://omg-dosimetry.readthedocs.io/en/latest/_modules/omg_dosimetry/calibration.html#LUT)
    
    Attributes
    ----------
    tiff_image : TiffImage
        The image used for calibration.

    lut : dict
        The look-up table (LUT) used to store data as a nested dictionary.
        At every milimeter in the lateral direction, the lut stores the corrected dose, the mean pixel value, 
        and the standard deviation of the pixel values for each color channel. A new calibration curve will
        be computed at every lateral position.
        The LUT is organized as follows:
        {
            'author' : str,
            'film_lote' : str,
            'scanner' : str,
            'date_exposed' : str,
            'date_read' : str,
            'wait_time' : str,
            'nominal_doses' : list,
            'resolution' : float,  # The resolution of the image in DPI.
            'lateral_limits' : list[float, float],
            (lateral_position : float, nominal_dose : float) : {
                'corrected_dose' : float,  # Contains the output and beam profile corrected doses.
                'I_red' : float,  # Mean pixel value of the red channel.
                'S_red' : float,  # Standard deviation of the red channel.
                'I_green' : float,
                'S_green' : float,
                'I_blue' : float,
                'S_blue' : float
                'I_mean' : float
                'S_mean' : float
                },
            ...
        }
    """
    
    def __init__(
            self,
            tiff_image: TiffImage,
            doses : list,
            lateral_correction : bool = False,
            beam_profile : ndarray = None,
            filter : int = None,
            metadata : dict = None,
            ):
        """
        Parameters
        ----------
        tiff_image : TiffImage
            The image used for calibration.
        doses : list of floats
            List of nominal doses values that were delivered on the films.
        lateral_correction : bool
            True: A LUT is computed for every milimeter in the scanner lateral direction
            False: A single LUT is computed for the scanner.
            As currently implemented, lateral correction is performed by exposing
            long strips of calibration films with a large uniform field (30 cm x 30 cm).
            By scanning the strips perpendicular to the scanner direction, a LUT is computed
            for each milímeter in the scanner lateral direction. If this method is
            used, it is recommended that beam profile correction be applied also,
            so as to remove the contribution of beam inhomogeneity.
        beam_profile : ndarray of size (n, 2)
            Beam profile  that will be used to correct the doses at each milimeter position.
            The array must contain the position and relative profile value.
            First column should be a position, given in mm, with 0 being at center.
            Second column should be the measured profile relative value [%], normalised to 100 in the center.
            Corrected doses are defined as dose_corrected(position) = dose * profile(position),
            where profile(y) is the beam profile, normalized to 100% at beam center
            axis, which is assumed to be aligned with scanner center.
        filter : int
            If filt > 0, a median filter of size (filt, filt) is applied to 
            each channel of the scanned image prior to LUT creation.            
            This feature might affect the automatic detection of film strips if
            they are not separated by a large enough gap (~ 5 mm). In this case, you can
            either use manual ROIs selection, or apply filtering to the LUT during
            the conversion to dose (see tiff2dose module).
        metadata : dict
            Dictionary with metadata information about the calibration.
            The following keys are required:
            'author' : str,
            'film_lote' : str,
            'scanner' : str,
            'date_exposed' : str,
            'date_read' : str,
            'wait_time' : str,
        """
        self.tiff_image = tiff_image
        self.lut = {}

    def create_central_rois(self, size : tuple) -> list:
        """
        Create a list of ROIs for the central region of the image.

        Parameters
        ----------
        size : tuple[int, int]
            The size of the ROIs in milimeters, width x height.

        Returns
        -------
        list
            A list of ROIs. 
            
            [
                (x0, y0, x1, y1),
                ...
            ]

            where:
            x0 and y0 are the coordinates of the top-left corner
            x1 and y1 are the coordinates of the bottom-right corner
            
        """
        # Check if the image is loaded.
        if self.tiff_image is None:
            raise Exception("No image loaded.")

        # Get the image size in mm.
        width, height = self.tiff_image.physical_shape

        # Get the image size in pixels.
        width_px, height_px = self.tiff_image.shape

        # Get the image resolution in mm.
        dpmm = self.tiff_image.dpmm

        # Calculate the size of the ROIs in pixels.
        width_roi = size[0] * dpmm
        height_roi = size[1] * dpmm

        # Get labeled image
        label_image, num_films_detected = self._get_labeled_image()


    def _get_labeled_image(self, threshold: float = None) -> tuple[ndarray, int]:
        """
        Get the labeled image of the films.

        Parameters
        ----------
        threshold : float
            The threshold value used to detect film. Pixel values below the threshold are considered films.
             If None, the Otsu method is used to define a threshold.
        
        Returns
        -------
        ndarray : 
            The labeled image, where all connceted regions are assigned the same integer value.
        num : int
            The number of films detected.
        """

        gray_scale = rgb2gray(self.tiff_img.array)

        if not threshold:
            thresh = threshold_otsu(gray_scale)  # Used for films identification.
        else:
            thresh = threshold * np.amax(gray_scale)

        # Number of pixels used for erosion. 
        # Used to remove the irregular borders of the films.
        # https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_erosion

        erosion_pix = int(6*self.tiff_img.dpmm)
        binary = erosion(gray_scale < thresh, square(erosion_pix))

        labeled_image, number_of_films = label(binary, return_num=True)

        return labeled_image, number_of_films


    def compute_lut(self, doses: list, rois: list, channel: str):
        """
        Compute the look-up table (LUT) for the calibration curve.

        Parameters
        ----------
        doses : list
            The doses values used to expose the films for calibration.
        rois : list
            The response of the film to the doses.
        channel : str
            Color channel. "R": Red, "G": Green and "B": Blue.
        """
        pass
    



class Calibration:
    """Class used to represent a calibration curve.

        Attributes
        ----------
        y : list
            The doses values that were used to expose films for calibration.
        x : list
            Optical density if "P3" fit function is used, or normalized pixel value
            for "RF" fit function.
        func : str
            The model function used for dose-film response relationship.
            "P3": Polynomial function of degree 3.
            "RF": Rational function.
        channel : str
            Color channel. "R": Red, "G": Green and "B": Blue.
        popt : array
            Parameters of the function.
        pcov : 2-D array
            The estimated approximate covariance of popt. The diagonals provide
            the variance of the parameter estimate. To compute one standard
            deviation errors on the parameters, use perr = np.sqrt(np.diag(pcov)).
        """

    def __init__(self, y: list, x: list, func: str = "P3", channel: str = "R"):

        self.doses = sorted(y)

        if func in ["P3", "Polynomial"]:
            self.x = sorted(x)  # Film response.
        elif func in ["RF", "Rational"]:
            self.x = sorted(x, reverse=True)

        self.func = func

        if self.func in ["P3", "Polynomial"]:
            self.popt, self.pcov = curve_fit(polynomial_g3, self.x, self.doses)
        elif self.func in ["RF", "Rational"]:
            self.popt, self.pcov = curve_fit(
                                            rational_func,
                                            self.x,
                                            self.doses,
                                            p0=[0.1, 200, 500]
                                            )
        else:
            raise Exception("Invalid fit function.")
        self.channel = channel

    def plot(self, ax: plt.Axes = None, show: bool = True, **kwargs) -> plt.Axes:
        """Plot the calibration curve.

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

        x = np.linspace(self.x[0], self.x[-1], 100)
        if self.func in ["P3", "Polynomial"]:
            y = polynomial_g3(x, *self.popt)
            ax.set_xlabel("Optical density")
        elif self.func in ["RF", "Rational"]:
            y = rational_func(x, *self.popt)
            ax.set_xlabel("Normalized pixel value")

        if self.channel in ["R", "Red", "r", "red"]:
            color = "red"
        elif self.channel in ["G", "Green", "g", "green"]:
            color = "green"
        elif self.channel in ["B", "Blue", "b", "blue"]:
            color = "blue"
        elif self.channel in ["M", "Mean", "m", "mean"]:
            color = "black"
        
        ax.plot(
            self.x,
            self.doses,
            color = color,
            marker = '*',
            linestyle = 'None',
            **kwargs
            )
        ax.plot(
            x,
            y,
            color=color,
        )
        ax.set_ylabel("Dose [Gy]")
        if show:
            plt.show()
        return ax
