"""
NAME
    Calibration module

DESCRIPTION
    Module for the management of the calibration curve. 
    See CalibrationLUT class for details.
    The CalibrationLUT class is an addaptation of the LUT class from 
    OMG Dosimetry package (https://omg-dosimetry.readthedocs.io/en/latest/_modules/omg_dosimetry/calibration.html#LUT)
    Main differences:
    - The data structure (LUT) to store intensites from pixels is a nested dictionary.
    - The data is created for every milimeter in the lateral direction of the scanner, instead of each pixel.
    - The data contains the standard deviation of the pixel values.
    - Automatic roi detection is implemented with scikit-image.regionprops() function.

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
from Dosepy.i_o import load_beam_profile


BIN_WIDTH = 1  # Width of the bin in milimeters. Used to compute the LUT.

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


def _get_dose_from_fit(xdata, ydata, x, fit_function):
    if fit_function == "rational":

        popt, pcov = curve_fit(rational_func, xdata, ydata, p0=[0.1, 200, 500])
        #popt, pcov = curve_fit(rational_func, xdata, ydata)
        return rational_func(x, *popt)


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
            'date_scanned' : str,
            'wait_time' : str,
            'nominal_doses' : list,
            'rois' : list[{  # List of ROIs used to compute the calibration curve.
                "x" : int,  # The x coordinate (row) of the top-left corner of the ROI.
                "y" : int,  # The y coordinate (column) of the top-left corner of the ROI.
                "width" : int,  # The width of the ROI in pixels.
                "height" : int,  # The height of the ROI in pixels.
                }],  
            'resolution' : float,  # The resolution of the image in dots per inch.
            'lateral_limits' : {"left": int, "right": int},  # Left and rigth lateral limits of the scanner where calibration is valid, in milimeters from the center of the image.
            (lateral_position : int, roi_number : int) : {
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
            doses : list = None,
            lateral_correction : bool = False,
            #beam_profile : ndarray = None,
            #filter : int = None,
            metadata : dict = {},
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
            'date_scanned' : str,
            'wait_time' : int,
        """
        self.tiff_image = tiff_image
        self.lut = {}

        self.lut['author'] = metadata.get('author')
        self.lut['film_lote'] = metadata.get('film_lote')
        self.lut['scanner'] = metadata.get('scanner')
        self.lut['date_exposed'] = metadata.get('date_exposed')
        self.lut['date_scanned'] = metadata.get('date_scanned')
        self.lut['wait_time'] = metadata.get('wait_time')
        self.lut['resolution'] = tiff_image.dpi
        self.lut['lateral_limits'] = {
            'left': None,
            'right': None,
        }
        self.lut['rois'] = None

        # Check if the doses are provided.
        if doses:
            self.set_doses(doses)
        #self.set_beam_profile(beam_profile)
        

    def set_doses(self, doses: list):
        """
        Set the nominal doses values that were delivered on the films.
        """
        # Check if the doses are numeric.
        if not all(isinstance(dose, (int, float)) for dose in doses):
            raise Exception("Doses must be numeric.")
        # Check if doses are positive.
        if not all(dose >= 0 for dose in doses):
            raise Exception("Doses must be positive.")
        # Check if doses are unique.
        if len(doses) != len(set(doses)):
            raise Exception("Doses must be unique.")

        self.lut['nominal_doses'] = sorted(doses)

    
    def set_beam_profile(self, beam_profile: str):
        """
        Beam profile  that will be used to correct the doses at each milimeter in lateral position.

        Parameters
        ----------
        beam_profile : str
            The path to the file containing the beam profile.

        Returns
        -------
        ndarray of size (n, 2)
        
        The array must contain the position and relative profile value.
        First column should be a position, given in mm, with 0 being at center.
        Second column should be the measured profile relative value [%], normalised to 100 in the center.
        Corrected doses are defined as dose_corrected(position) = dose * profile(position),
        where profile(y) is the beam profile, normalized to 100% at beam center
        axis, which is assumed to be aligned with scanner center.
        """

        # Check if the beam profile is a string.
        if not isinstance(beam_profile, str):
            raise Exception("Beam profile must be a string.")

        # Load the beam profile.
        self.lut['beam_profile'] = load_beam_profile(beam_profile)

        # Print the beam profile.
        #print(self.lut['beam_profile'])
        #type(self.lut['beam_profile'])


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
            A list of ROIs as dictionaries.     
            [{'x': int, 'y': int, 'width': int, 'height': int}, ...]
            where: x and y are the coordinates of the top-left corner 
            (x for row and y for column),
            and width and height are the size of the ROI in pixels.
            
        """
        # Check if the image is loaded.
        if self.tiff_image is None:
            raise Exception("No image loaded.")

        # Get the image size in mm.
        height, width = self.tiff_image.physical_shape

        # Check if the size of the ROIs is valid.
        if size[0] > width or size[1] > height:
            raise Exception("Invalid ROI size. The size of the ROIs must be smaller than the image.")

        # Get the image resolution in mm.
        dpmm = self.tiff_image.dpmm

        # Calculate the size of the ROIs in pixels.
        width_roi = size[0] * dpmm
        height_roi = size[1] * dpmm

        # Get labeled image
        label_image, num_films_detected = self._get_labeled_image()

        rois = []

        # Get the central region of the image.
        for region in regionprops(label_image):
            if region.area > 0.5*width_roi*height_roi:
                rois.append(
                    {
                        'x': int(region.centroid[0] - height_roi/2),
                        'y': int(region.centroid[1] - width_roi/2),
                        'width': int(width_roi),
                        'height': int(height_roi),
                    }
                )
        self.lut["rois"] = rois

        # Find maximum y values in milimeters.
        self.lut["lateral_limits"]["left"] = int(max([roi['y'] for roi in rois])/dpmm - width/2)

        # Find minimum y + width values in milimeters.
        self.lut["lateral_limits"]["right"] = int(min([roi['y'] + roi["width"] for roi in rois])/dpmm - width/2)

        # Plot for visualization testing purposes.
        #self._plot_rois(dpmm = dpmm, origin = -width/2)


    def compute_lateral_lut(self):
        """
        Compute the look-up table (LUT) in lateral positions at every bin of size BIN_WIDTH.
        """
        # Check if the image is loaded.
        if self.tiff_image is None:
            raise Exception("No image loaded.")
        # Check if rois are created.
        if not self.lut.get("rois"):
            raise Exception("No ROIs created. Use the create_central_rois method to set the ROIs.")


        # Create a list with the lateral positions in milimeters
        origin_side2half = self.tiff_image.physical_shape[1]/2

        lateral_positions_half = np.linspace(
            start = 0,
            stop = self.tiff_image.physical_shape[1],
            num = self.tiff_image.array.shape[1]
            ) - origin_side2half


        for roin_num, roi in enumerate(self.lut["rois"]):

            # Define a bin_step limit to append pixels in a bin of size BAND_WIDTH.
            bin_limit = int(self.lut["lateral_limits"]["left"]) + BIN_WIDTH
            bin_buffer_red = np.array([])
            bin_buffer_green = np.array([])
            bin_buffer_blue = np.array([])

            for column_pixel in range(self.tiff_image.array.shape[1]):

                rounded_position = int(lateral_positions_half[column_pixel])

                # Populate the LUT with None if the pixel is outside the lateral limits.
                if lateral_positions_half[column_pixel] < self.lut["lateral_limits"]["left"] or lateral_positions_half[column_pixel] > self.lut["lateral_limits"]["right"]:
                    self.lut[(rounded_position, roin_num)] = None

                # Join pixel values in bin_width into a band buffer.
                elif lateral_positions_half[column_pixel] <= bin_limit:
                    bin_buffer_red = np.concatenate(
                        (
                            bin_buffer_red,
                            self.tiff_image.array[roi['x'] : roi['x'] + roi['height'], column_pixel, 0]
                        ),
                        axis=None,
                    )
                    bin_buffer_green = np.concatenate(
                        (
                            bin_buffer_green,
                            self.tiff_image.array[roi['x'] : roi['x'] + roi['height'], column_pixel, 1]
                        ),
                        axis=None,
                    )
                    bin_buffer_blue = np.concatenate(
                        (
                            bin_buffer_blue,
                            self.tiff_image.array[roi['x'] : roi['x'] + roi['height'], column_pixel, 2]
                        ),
                        axis=None,
                    )
                
                else:
                    #print(rounded_position, roin_num)
                    # Populate the LUT with the mean pixel value and standard deviation of the band.
                    self.lut[(rounded_position, roin_num)] = {
                        'I_red': int(np.mean(bin_buffer_red)),
                        'S_red': int(np.std(bin_buffer_red)),
                        'I_green': int(np.mean(bin_buffer_green)),
                        'S_green': int(np.std(bin_buffer_green)),
                        'I_blue': int(np.mean(bin_buffer_blue)),
                        'S_blue': int(np.std(bin_buffer_blue)),
                    }
                    self.lut[(rounded_position, roin_num)]["I_mean"] = int(
                        (
                            self.lut[(rounded_position, roin_num)]["I_red"] +
                            self.lut[(rounded_position, roin_num)]["I_green"] +
                            self.lut[(rounded_position, roin_num)]["I_blue"]
                            ) / 3
                        )
                    self.lut[(rounded_position, roin_num)]["S_mean"] = int(
                        (
                            self.lut[(rounded_position, roin_num)]["S_red"]**2 +
                            self.lut[(rounded_position, roin_num)]["S_green"]**2 +
                            self.lut[(rounded_position, roin_num)]["S_blue"]**2
                            )**0.5 / 3
                        )
                    # Update bin_step and bin_buffer.

                    bin_limit += BIN_WIDTH

                    bin_buffer_red = np.array([])
                    np.concatenate(
                        (
                            bin_buffer_red,
                            self.tiff_image.array[roi['x'] : roi['x'] + roi['height'], column_pixel, 0]
                        ),
                        axis=None,
                    )

                    bin_buffer_green = np.array([])
                    np.concatenate(
                        (
                            bin_buffer_green,
                            self.tiff_image.array[roi['x'] : roi['x'] + roi['height'], column_pixel, 1]
                        ),
                        axis=None,
                    )

                    bin_buffer_blue = np.array([])
                    np.concatenate(
                        (
                            bin_buffer_blue,
                            self.tiff_image.array[roi['x'] : roi['x'] + roi['height'], column_pixel, 2]
                        ),
                        axis=None,
                    )
        #self._plot_rois(dpmm=self.tiff_image.dpmm, origin=-self.tiff_image.physical_shape[1]/2)


    def plot_lateral_response(self, channel: str = "red"):
        """
        Plot the lateral response of the scanner for each ROI (film).

        Parameters
        ----------
        channel : str
            The color channel to plot. "red", "green", "blue" or "mean".

        Examples
        --------
        >>> cal = CalibrationLUT(tiff_image)
        >>> cal.create_central_rois((180, 8))
        >>> cal.compute_lateral_lut()
        >>> cal.plot_lateral_response(channel = "red")
        >>> plt.show()
        """
        if channel.lower() in ["red", "r"]:
            channel = "I_red"
            color = "red"
        elif channel.lower() in ["green", "g"]:
            channel = "I_green"
            color = "green"
        elif channel.lower() in ["blue", "b"]:
            channel = "I_blue"
            color = "blue"
        elif channel.lower() in ["mean", "m"]:
            channel = "I_mean"
            color = "black"

        # Get positions from lut
        positions = self._get_calibration_positions()

        # Number of ROIs
        num_rois = len(self.lut["rois"])

        # Set up the figure.
        fig, axes = plt.subplots(num_rois, 1, sharex=True)
        fig.suptitle("Lateral response [%] of the scanner.")
        #fig.suptitle("Another title", fontsize=10)
        axes[0].set_title(
            f"Error bars represent the standard deviation of the pixel values in the ROI.",
            fontsize = 9,
            )
        axes[-1].set_xlabel("Lateral position [mm]")

        for ax in axes:
            ax.set_ylim(-10, 5)
            #ax.set_ylabel("[%]")
            ax.grid(True)

        # Get the lateral pixel values and coordinate, for each roi.
        for roi_counter in range(num_rois):

            intensity = np.array([
                self.lut[(position, roi_counter)][channel]
                for position in positions
                if self.lut[(position, roi_counter)]
                ])

            std = np.array([
                self.lut[(position, roi_counter)][channel.replace("I", "S")]
                for position in positions
                if self.lut[(position, roi_counter)]
            ])

            coordinate = np.array([
                position
                for position in positions
                if self.lut[(position, roi_counter)]
            ])

            # Normalize the pixel values to the central pixel value.
            I_central = intensity[int(len(intensity)/2)]
            I_relative = intensity / I_central * 100 - 100

            # Standar deviation of the pixel values.
            std_Ir = std / I_central * 100

            # Plot the lateral response.
            #line, = axes[roi_counter].plot(coordinate, I_relative, color = color, marker = '.')
            #line, = axes[roi_counter].errorbar(coordinate, I_relative, yerr = std_Ir)
            axes[roi_counter].errorbar(
                coordinate, I_relative,
                yerr = std_Ir,
                color = color,
                ecolor = color,
                label = f"ROI: {roi_counter + 1}",
                )
            #line.set_label(f"ROI: {roi_counter + 1}")
            axes[roi_counter].legend()


    def plot_fit(
        self,
        fit_type: str = 'rational',
        channel: str = 'red',
        position: float = 0,
        ax: plt.Axes = None,
        ):
        """
        Plot the fit of the calibration curve.

        Parameters
        ----------
        fit_type : str
            The type of fit to use. "rational" or "polynomial".
        channel : str
            The color channel to plot. "red", "green", "blue" or "mean".
        position : float
            The lateral position in milimeters.
        ax : plt.Axes
            The axis to plot the image to. If None, creates a new figure.
        """
        if fit_type.lower() not in ["rational", "polynomial"]:
            raise Exception("Invalid fit type. Choose between 'rational' or 'polynomial'.")
        
        if channel.lower() not in ["red", "green", "blue", "mean"]:
            raise Exception("Invalid channel. Choose between 'red', 'green', 'blue' or 'mean'.")

        if position < self.lut["lateral_limits"]["left"] or position > self.lut["lateral_limits"]["right"]:
            raise Exception("Position out of lateral limits.")

        # Round the position to the nearest integer.
        position = round(position)

        # Get the pixel values of the channel at the given lateral position.
        intensities = self._get_intensities(position, channel)

        if fit_type == "rational":
            responses = np.array(intensities) / intensities[0]
        elif fit_type == "polynomial":
            #TODO: Create responses.
            pass

        # Get the doses values that were used to expose the films for calibration
        # at a given position.
        doses = self._get_lateral_doses(position)

        # Create the calibration curve.
        response_curve = np.linspace(responses[0], responses[-1], 100)
        dose_curve = _get_dose_from_fit(responses, doses, response_curve, fit_type)

        if ax is None:
            fig, axe = plt.subplots()
        else: 
            axe = ax

        axe.plot(responses, doses, marker = '*')
        axe.plot(response_curve, dose_curve)


    def _plot_rois(self, ax: plt.Axes = None):
        """
        Plot the ROIs on the image.
        """
        dpmm = self.tiff_image.dpmm
        origin = -self.tiff_image.physical_shape[1]/2

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(self.tiff_image.array/np.max(self.tiff_image.array))
        for roi in self.lut["rois"]:
            rect = plt.Rectangle(
                (roi['y'], roi['x']),
                roi['width'],
                roi['height'],
                edgecolor = 'r',
                fill = False,
                linestyle = '--',
            )
            ax.add_patch(rect)
        
        # r0' = r0 - r00', change of origin.
        #print(self.lut["lateral_limits"])
        y_left_limit_pix =int((self.lut["lateral_limits"]["left"] - origin)*dpmm)
        y_right_limit_pix = int((self.lut["lateral_limits"]["right"] - origin)*dpmm)

        ax.axvline(y_left_limit_pix)
        ax.axvline(y_right_limit_pix)
        #plt.show()


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

        gray_scale = rgb2gray(self.tiff_image.array)

        if not threshold:
            thresh = threshold_otsu(gray_scale)  # Used for films identification.
        else:
            thresh = threshold * np.amax(gray_scale)

        # Number of pixels used for erosion. 
        # Used to remove the irregular borders of the films.
        # https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_erosion

        erosion_pix = int(6*self.tiff_image.dpmm)  # 6 mlimiters.
        binary = erosion(gray_scale < thresh, square(erosion_pix))

        labeled_image, number_of_films = label(binary, return_num=True)

        return labeled_image, number_of_films


    def _get_calibration_positions(self) -> list:
        """
        Get the calibration positions in milimeters.
        """
        positions = [key[0] for key in self.lut.keys() if isinstance(key, tuple)]
        return sorted(set(positions))
    

    def _get_intensities(self, lateral_position: float, channel: str) -> list:
        """
        Get the pixel values of the channel at a given lateral position, for each film.

        Parameters
        ----------
        lateral_position : float
            The lateral position in milimeters.

        channel : str
            The color channel. "red", "green", "blue" or "mean".

        Returns
        -------
        list
            A list of pixel values, sorted in descending order.
        """
        if channel.lower() not in ["red", "green", "blue", "mean"]:
            raise Exception("Invalid channel. Choose between 'red', 'green', 'blue' or 'mean'.")
        
        position = int(lateral_position)
        intensities = []

        for roi in range(len(self.lut["rois"])):
            intensities.append(self.lut[(position, roi)]["I_" + channel])

        return sorted(intensities, reverse=True)

    
    def _get_lateral_doses(self, position: float) -> list:
        """
        Get lateral doses at a given position for each film corrected by beam profile.

        Parameters
        ----------
        position : float
            The lateral position in milimeters.
        
        Returns
        -------
        list
            A list of doses for each film.
        """
        # Check if the beam profile is loaded.
        #if self.lut.get("beam_profile"):
        #    raise Exception("No beam profile loaded.")

        # Check if the doses are set.
        #if not self.lut.get("nominal_doses"):
        #    raise Exception("No doses set.")
        
        profile = np.interp(
            position,
            self.lut['beam_profile'][:, 0],
            self.lut['beam_profile'][:, 1]) / 100

        lateral_doses = sorted([dose * profile for dose in self.lut["nominal_doses"]])

        return lateral_doses


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
