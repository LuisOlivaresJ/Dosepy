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
    - The data contains the standard deviation of the pixel values for uncertainty analysis.
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
from skimage.filters.rank import mean, median
from skimage.transform import rotate

from Dosepy.image import TiffImage
from Dosepy.i_o import load_beam_profile

import logging


BIN_WIDTH = 1  # Width of the bin in milimeters used to compute the calibration LUT.

"""Functions used for film calibration."""


def polynomial_g3(x, a, b, c, d):
    """
    Polynomial function of degree 3.
    """
    return a + b*x + c*x**2 + d*x**3


def polynomial_n(x, a, b, n):
    """
    Polynomial function of degree n.
    """
    return a*x + b*x**n


def rational_func(x, a, b, c):
    """
    Rational function.
    """
    return -c + b/(x-a)


def _get_dose_from_fit(calib_film_response, calib_dose, response, fit_function):

    if fit_function == "rational":

        xdata = sorted(calib_film_response, reverse=True)
        ydata = sorted(calib_dose)

        popt, pcov = curve_fit(
            rational_func,
            xdata,
            ydata,
            p0=[0.1, 4, 4],
            maxfev=1500,
            )
        
        return rational_func(response, *popt)

    elif fit_function == "polynomial":
        
        xdata = sorted(calib_film_response)
        ydata = sorted(calib_dose)

        popt, pcov = curve_fit(
            polynomial_n,
            xdata,
            ydata,
            maxfev=1500,
            )

        return polynomial_n(response, *popt)


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
            'filter' : int,  # The size in pixels of the median filter applied to the image.
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
        self.filter = None

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


    def compute_lateral_lut(self, filter: int = None):
        """
        Compute the look-up table (LUT) in lateral positions at every bin of size BIN_WIDTH.

        Parameters
        ----------
        filter : int
            If filter > 0, a median filter of size (filter, filter) is applied to 
            each channel of the scanned image prior to LUT creation.            

        """
        # Check if the image is loaded.
        if self.tiff_image is None:
            raise Exception("No image loaded.")
        # Check if rois are created.
        if not self.lut.get("rois"):
            raise Exception("No ROIs created. Use the create_central_rois method to set the ROIs.")
        # Check if filter is valid.
        if filter and filter < 0:
            raise Exception("Filter must be a positive integer.")
        # Check if filter is a integer.
        if filter and not isinstance(filter, int):
            raise Exception("Filter must be an integer.")


        # Apply filter to the image.
        if filter:
            self.filter = filter
            # Labeled area of the films.
            mask, _ = self._get_labeled_image()
            # Array buffer to store the filtered image.
            array_img = np.empty(
                shape = (
                    self.tiff_image.array.shape[0],
                    self.tiff_image.array.shape[1],
                    self.tiff_image.array.shape[2]
                    ),
                dtype = np.uint16
                )
            for i in range(3):
                
                array_img[:,:,i] = median(
                    self.tiff_image.array[:,:,i],
                    footprint = square(filter),
                    mask = mask,
                    )
        else:
            # Unfiltered image.
            array_img = self.tiff_image.array

        # Create a list with the lateral positions in milimeters
        origin = self.tiff_image.physical_shape[1]/2
        pixel_position = np.linspace(
            start = 0,
            stop = self.tiff_image.physical_shape[1],
            num = self.tiff_image.array.shape[1]
            ) - origin


        for roi_num, roi in enumerate(self.lut["rois"]):
            
            target_position = int(self.lut["lateral_limits"]["left"])

            for column_pixel in range(self.tiff_image.array.shape[1]):

                # Populate the LUT with None if the pixel is outside the lateral limits.
                if pixel_position[column_pixel] < self.lut["lateral_limits"]["left"] or pixel_position[column_pixel] > self.lut["lateral_limits"]["right"]:
                    rounded_position = round(pixel_position[column_pixel])
                    self.lut[(rounded_position, roi_num)] = None

                # Populate the LUT with the pixel values.
                else:
                    diff = abs(pixel_position[column_pixel] - target_position)
                    #print(diff)
                    #print(self.tiff_image.dpmm)
                    if diff <= 1/self.tiff_image.dpmm:
                        #print(f"Target position: {target_position}")
                        self.lut[(target_position, roi_num)] = {
                            'I_red': int(np.mean(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel, 0])),
                            'S_red': int(np.std(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel, 0])),
                            'I_green': int(np.mean(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel, 1])),
                            'S_green': int(np.std(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel, 1])),
                            'I_blue': int(np.mean(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel, 2])),
                            'S_blue': int(np.std(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel, 2])),
                        }
                            
                        self.lut[(target_position, roi_num)]["I_mean"] = int(
                            (
                                self.lut[(target_position, roi_num)]["I_red"] +
                                self.lut[(target_position, roi_num)]["I_green"] +
                                self.lut[(target_position, roi_num)]["I_blue"]
                                ) / 3
                            )
                        self.lut[(target_position, roi_num)]["S_mean"] = int(
                            (
                                self.lut[(target_position, roi_num)]["S_red"]**2 +
                                self.lut[(target_position, roi_num)]["S_green"]**2 +
                                self.lut[(target_position, roi_num)]["S_blue"]**2
                                )**0.5 / 3
                            )
                        # Update target position.
                        target_position += 1

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

        if channel == "mean":
            color = "black"
        color = channel

        # Number of ROIs
        num_rois = len(self.lut["rois"])

        # Set up the figure.
        fig, axes = plt.subplots(num_rois, 1, sharex=True)
        fig.suptitle("Lateral response [%] of the scanner.")
        axes[0].set_title(
            f"Error bars represent the standard deviation of the pixel values in the ROI.",
            fontsize = 9,
            )
        axes[-1].set_xlabel("Lateral position [mm]")

        for ax in axes:
            ax.set_ylim(-10, 5)
            ax.grid(True)

        # Get the lateral pixel values, standar deviation and coordinate, for each roi.
        for roi_counter in range(num_rois):

            intensity, std, coordinate = self.get_lateral_respose(roi_counter, channel)

            # Normalize the pixel values to the central pixel value.
            I_central = intensity[int(len(intensity)/2)]
            I_relative = intensity / I_central * 100 - 100

            # Standar deviation of the pixel values.
            std_Ir = std / I_central * 100

            # Plot the lateral response.
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
        Plot the fit function of the calibration curve at a given lateral position.

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

        # Round the position.
        position = int(position)

        # Get the pixel values of the channel at the given lateral position.
        intensities, std = self._get_intensities(position, channel)

        if fit_type == "rational":
            response = intensities / intensities[0]
            #print(response)
            # Uncertainty propagation.
            std_response = response * np.sqrt( (std/intensities)**2 + (std[0]/intensities[0])**2 )
        elif fit_type == "polynomial":
            response = -np.log10(intensities/intensities[0])
            #print(response)
            # Uncertainty propagation.
            std_response = (1/np.log(10))*np.sqrt( (std/intensities)**2 + (std[0]/intensities[0])**2 )

        # Get the corrected doses used to expose the films for calibration
        # at a given position.
        doses = self._get_lateral_doses(position)

        # Create the calibration curve.
        response_curve = np.linspace(response[0], response[-1], 100)
        dose_curve = _get_dose_from_fit(response, doses, response_curve, fit_type)

        if ax is None:
            fig, axe = plt.subplots()
        else: 
            axe = ax

        #axe.plot(response, doses, marker = '*', linestyle="None", color = channel)
        axe.errorbar(response, doses, xerr = std_response, color = channel, marker = '*', linestyle="None")
        axe.plot(response_curve, dose_curve, color = channel)


    def plot_dose_fit_uncertainty(
        self,
        position: float,
        channel: str,
        fit_function: str,
        ax: plt.Axes = None,
        **kwargs
        ):
        """
        Plot the dose fit uncertainty at a given lateral position and channel.

        Parameters
        ----------
        position : float
            The lateral position in milimeters.
        channel : str
            The color channel to plot. "red", "green", "blue" or "mean".
        fit_function : str
            The type of fit to use. "rational" or "polynomial".
        """
        doses = self._get_lateral_doses(position)
        intensities, std = self._get_intensities(position, channel)
        uncertainty = self._get_dose_uncertainty(intensities, std, doses, fit_function)
        u_percent = uncertainty[1:] / doses[1:] * 100
        
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(doses[1:], u_percent, marker = '*', linestyle = '--', color = channel, **kwargs)
        ax.set_xlabel("Dose [Gy]")
        ax.set_ylabel("Dose uncertainty [%]")


    def get_lateral_respose(self, roi: int, channel: str) -> tuple[ndarray, ndarray, ndarray]:
        """
        Get the lateral response of the scanner for a given ROI and channel.

        Parameters
        ----------
        roi_number : int
            The number of the ROI (or film).
        channel : str
            The color channel to plot. "red", "green", "blue" or "mean".

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]
            A tuple with the pixel values, standar deviation and coordinates.
        """
        # Check if the channel is valid.
        if channel.lower() not in ["red", "green", "blue", "mean"]:
            raise Exception("Invalid channel. Choose between 'red', 'green', 'blue' or 'mean'.")
        
        # Check if there is ROIs in the lut.
        if not self.lut["rois"]:
            raise Exception("No ROIs created. Use the create_central_rois method to set the ROIs.")
        
        # Check if the roi is a valid number for the lut.
        if roi < 0 or roi >= len(self.lut["rois"]):
            raise Exception("Invalid ROI number.")
        
        if channel.lower() in ["red", "r"]:
            channel = "I_red"
        elif channel.lower() in ["green", "g"]:
            channel = "I_green"
        elif channel.lower() in ["blue", "b"]:
            channel = "I_blue"
        elif channel.lower() in ["mean", "m"]:
            channel = "I_mean"

        # Get the calibration positions.
        positions = self._get_calibration_positions()

        # Get the pixel values, standar and coordinate for a valid calibration region.
        intensity = np.array([
            self.lut[(position, roi)][channel]
            for position in positions
            if self.lut[(position, roi)]
            ])

        std = np.array([
            self.lut[(position, roi)][channel.replace("I", "S")]
            for position in positions
            if self.lut[(position, roi)]
        ])

        coordinate = np.array([
            position
            for position in positions
            if self.lut[(position, roi)]
        ])

        return np.array(intensity), np.array(std), np.array(coordinate)


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
    

    def _get_intensities(self, lateral_position: float, channel: str) -> ndarray:
        """
        Get the pixel values and standar deviation of the channel at a given lateral position,
        for each film, in descending order. 

        Parameters
        ----------
        lateral_position : float
            The lateral position in milimeters.

        channel : str
            The color channel. "red", "green", "blue" or "mean".

        Returns
        -------
        ndarray
            Arrays with the pixel values and standar deaviation of the channel at the given lateral position.
        """
        # Check if the channel is valid.
        if channel.lower() not in ["red", "green", "blue", "mean"]:
            raise Exception("Invalid channel. Choose between 'red', 'green', 'blue' or 'mean'.")
        
        # Check if position is between the lateral limits.
        if lateral_position < self.lut["lateral_limits"]["left"] or lateral_position > self.lut["lateral_limits"]["right"]:
            raise Exception("Position out of lateral limits.")
        
        position = round(lateral_position)
        intensities = []
        std = []

        for roi in range(len(self.lut["rois"])):
            intensities.append(self.lut[(position, roi)]["I_" + channel])
            std.append(self.lut[(position, roi)]["S_" + channel])

        data = zip(intensities, std)
        # Sort by intensities in descending order.
        sorted_data = sorted(data, reverse=True)

        intensities, std = zip(*sorted_data)

        return np.array(intensities), np.array(std)

    
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

        lateral_doses = sorted([float(dose * profile) for dose in self.lut["nominal_doses"]])

        return lateral_doses


    def _get_dose_uncertainty(self, intensities: ndarray, std: ndarray, doses: list, fit_function: str) -> ndarray:
        """
        Get the uncertainty of the dose fit at a given lateral position and channel.

        Parameters
        ----------
        intensities : ndarray
            The pixel values at a given lateral position.
        std : ndarray
            The standard deviation of the pixel values at a given lateral position.
        doses : list
            The corrected doses used to expose the films for calibration at a given lateral position.
        fit_function : str
            The type of fit to use. "rational" or "polynomial".

        Returns
        -------
        ndarray
            The uncertainty of the dose fit.
        """
        if fit_function == "rational":

            response = intensities / intensities[0]
            # Uncertainty propagation.
            std_response = response * np.sqrt( (std/intensities)**2 + (std[0]/intensities[0])**2 )

            popt, pcov = curve_fit(
                rational_func,
                response,
                doses,
                p0=[0.1, 4.0, 4.0],
                maxfev=1500,
                #method='trf',
                )
            print("\nInside _get_dose_uncertainty\n")
            print("response:")
            print(response)
            print("doses:")
            print(doses)
            print("Coefficients")
            print(popt)
            print("Covariance")
            print(np.sqrt(np.diag(pcov)))

            a = popt[0]
            b = popt[1]
            ua = np.sqrt(np.diag(pcov))[0]
            ub = np.sqrt(np.diag(pcov))[1]

            u_exp = b*std_response/(response-a)**2
            u_fit = np.sqrt( (b*ua/(response-a)**2)**2 + (ub/(response-a))**2 )
            u_d = np.sqrt( u_exp**2 + u_fit**2 )

        
        elif fit_function == "polynomial":
            response = -np.log10(intensities/intensities[0])
            # Uncertainty propagation.
            std_response = (1/np.log(10))*np.sqrt( (std/intensities)**2 + (std[0]/intensities[0])**2 )

            popt, pcov = curve_fit(
                polynomial_n,
                response,
                doses,
                #p0=[0.1, 4.0, 4.0, 4.0],
                maxfev=1500,
                #method='trf',
                )

            a = popt[0]
            b = popt[1]
            n = popt[2]
            ua = np.sqrt(np.diag(pcov))[0]
            ub = np.sqrt(np.diag(pcov))[1]

            u_exp = (a + n*b*response**(n-1))*std_response
            u_fit = np.sqrt( response**2*ua**2 + response**(2*n)*ub**2 )
            u_d = np.sqrt( u_exp**2 + u_fit**2 )

        return u_d

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
