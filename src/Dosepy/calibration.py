"""
NAME
    Calibration module

DESCRIPTION
    Module for the management of the film calibration. 
    See LUT class for details.
    The LUT class is an adoption of the LUT class from 
    OMG Dosimetry package (https://omg-dosimetry.readthedocs.io/en/latest/_modules/omg_dosimetry/calibration.html#LUT)
    Main differences:
    - The data structure (LUT) to store intensites from pixels uses a dictionary and get methods.
    - The data is created at every milimeter in the lateral direction of the scanner, instead of each pixel.
    - The data contains the standard deviation of the pixel values for uncertainty analysis.
    - Automatic roi detection is implemented with scikit-image methods.
    - Added a plot function for fit and scanner uncertainty.

"""

from __future__ import annotations
import math

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import square, erosion
from skimage.measure import label, regionprops
from skimage.filters.rank import median
from skimage.transform import rotate

from Dosepy.image import TiffImage, MM_PER_INCH
from Dosepy.i_o import load_beam_profile
from Dosepy.tools.functions import optical_density, uncertainty_optical_density, ratio, uncertainty_ratio
from Dosepy.tools.functions import polynomial_g3, rational_function, polynomial_n

import yaml


BIN_WIDTH = 1  # Width of the bin in milimeters used to compute the calibration LUT.


class LUT:
    """
    Class to store data for film calibration.
    This class is an adoption of the LUT class from OMG Dosimetry package 
    (https://omg-dosimetry.readthedocs.io/en/latest/_modules/omg_dosimetry/calibration.html#LUT)
    
    Attributes
    ----------
    tiff_image : TiffImage
        The image used for calibration.

    lut : dict
        The look-up table (LUT) used to store data as a nested dictionary.
        At every milimeter (BIN_WIDTH constant) in the lateral direction
        (perpendicular to the scanning direction), the lut stores the corrected dose,
        the mean, and the standard deviation of the pixel values for each color channel.
        The LUT is organized as follows:
        {
            'author' : str,
            'film_lote' : str,
            'scanner' : str,
            'date_exposed' : str,
            'date_scanned' : str,
            'wait_time' : str,
            'nominal_doses' : list, # List of sorted doses that were delivered on the films.
            'filter' : int,  # The size in pixels of the median filter applied to the image.
            'rois' : list[{  # List of ROIs used to compute the calibration curve.
                "x" : int,  # The x coordinate (row) of the top-left corner of the ROI.
                "y" : int,  # The y coordinate (column) of the top-left corner of the ROI.
                "width" : int,  # The width of the ROI in pixels.
                "height" : int,  # The height of the ROI in pixels.
                }],  
            'resolution' : float,  # The resolution of the image in dots per inch.
            'lateral_limits' : {"left": int, "right": int},  # Left and rigth lateral limits of the scanner where calibration is valid, in milimeters from the center of the image.
            'lateral_correction' : bool,  # True if a LUT is computed for every milimeter in the scanner lateral direction.
            'rois_for_optical_filters' : list of dictionaries containing coordinate and radius of circular roi(s).
            'intensities_of_optical_filters : list, # Sorted red channel intensities of optical filters.
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
            tiff_image: TiffImage = None,
            doses : list = None,
            metadata : dict = {},
            ):
        """
        Parameters
        ----------
        tiff_image : TiffImage
            The image used for calibration.

        doses : list of floats
            List of nominal doses values that were delivered on the films.

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

        self.lut = {}

        if tiff_image is not None and not isinstance(tiff_image, TiffImage):
            raise Exception("Image must be a TiffImage object.")

        if tiff_image is not None:
            self.tiff_image = tiff_image
            self.lut['resolution'] = tiff_image.dpi

        else:
            self.tiff_image = None
            self.lut['resolution'] = None

        self.lut['author'] = metadata.get('author')
        self.lut['film_lote'] = metadata.get('film_lote')
        self.lut['scanner'] = metadata.get('scanner')
        self.lut['date_exposed'] = metadata.get('date_exposed')
        self.lut['date_scanned'] = metadata.get('date_scanned')
        self.lut['wait_time'] = metadata.get('wait_time')
        self.lut['lateral_limits'] = {
            'left': None,
            'right': None,
        }
        self.lut['lateral_correction'] = False
        self.lut['rois'] = None

        self.lut['filter'] = None
        self.lut['rois_for_optical_filters'] = None
        self.lut['intensities_of_optical_filters'] = None

        # Check if the doses are provided.
        if doses:
            self.set_doses(doses)
        

    def set_doses(self, doses: list) -> None:
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

    
    def set_beam_profile(self, beam_profile: str) -> None:
        """
        Beam profile that will be used to correct the doses at each milimeter in lateral position
        due to beam horns.

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
            raise Exception("Beam profile must be a string that has the path to the file.")

        # Load the beam profile.
        self.lut['beam_profile'] = load_beam_profile(beam_profile)


    def set_central_rois(self, size: tuple = None, show = False) -> None:
        """
        Used for film and optical filter identification on the image.
        This allows to create regions of interest and lateral limits.

        Parameters
        ----------
        size : tuple[width: int, height int]
            The size of the ROIs in milimeters. If None, an automatic roi size is performed.

        Note
        -------
        lut.["rois"] is created with the following structure:    
        [
            {'x': int, 'y': int, 'width': int, 'height': int},
            {...},
            ...
        ]
        where: x and y are the coordinates of the top-left corner 
        (x for row and y for column),
        width and height are the size of the ROI in pixels.
            
        """
        # Check if the image is loaded.
        if self.tiff_image is None:
            raise Exception("No image loaded.")
        
        # Get labeled objects
        self.tiff_image.set_labeled_films_and_filters()
        labeled_films = self.tiff_image.labeled_films

        # Get the image resolution in mm.
        dpmm = self.lut["resolution"]/MM_PER_INCH

        # Get the image size in mm.
        height, width = self.tiff_image.physical_shape

        rois = []
        if size:
            # Check size of ROIs.
            if size[0] > width or size[1] > height:
                raise Exception("Invalid ROI size. The size of the ROIs must be smaller than the image.")

            # Calculate the size of the ROIs in pixels.
            width_roi = size[0] * dpmm
            height_roi = size[1] * dpmm

            # Get the central region of each film.
            for region in regionprops(labeled_films):
                # TODO Is this check necesary? set_labeled_films_and_filters filters for small objects
                #if region.area > 0.5*width_roi*height_roi:
                rois.append(
                    {
                        'x': int(region.centroid[0] - height_roi/2),
                        'y': int(region.centroid[1] - width_roi/2),
                        'width': int(width_roi),
                        'height': int(height_roi),
                    }
                )

        else:  # Roi size based of region properties
            for region in regionprops(labeled_films):

                # Create rois based on region properties

                pix_in_3_mm = int(3*dpmm)  # Used to remove borders

                x = region.bbox[0] + pix_in_3_mm
                y = region.bbox[1] + pix_in_3_mm
                width_roi = region.bbox[3] - region.bbox[1] - 2*pix_in_3_mm
                height_roi = region.bbox[2]- region.bbox[0] - 2*pix_in_3_mm

                rois.append(
                    {
                        'x': x,
                        'y': y,
                        'width': width_roi,
                        'height': height_roi 
                    }
                )

        self.lut["rois"] = rois

        # Find maximum y values in milimeters.
        self.lut["lateral_limits"]["left"] = int(max([roi['y'] for roi in rois])/dpmm - width/2)

        # Find minimum y + width values in milimeters.
        self.lut["lateral_limits"]["right"] = int(min([roi['y'] + roi["width"] for roi in rois])/dpmm - width/2)

        # Plot rois.
        if show:
            self._plot_rois()


    def compute_lateral_lut(self, filter: int = None) -> None:
        """
        Compute the look-up table (LUT) in lateral positions at every bin of size BIN_WIDTH.

        Parameters
        ----------
        filter : int
            If filter > 0, a median filter of size (filter, filter) is applied to 
            each channel of the scanned image prior to LUT creation.            

        """
        self._check_before_compute_lut(filter)
        self.lut['lateral_correction'] = True

        # Apply a filter to the image.
        if filter:
            self.lut['filter'] = filter
            self.tiff_image.filter_channel(filter, "median", "R")
            self.tiff_image.filter_channel(filter, "median", "G")
            self.tiff_image.filter_channel(filter, "median", "B")

            array_img = self.tiff_image.array

        else: # Unfiltered image.
            array_img = self.tiff_image.array

        # Create a list with the lateral positions in milimeters
        origin = self.tiff_image.physical_shape[1]/2  # Origin in the middle of the image.
        pixel_position = np.linspace(
            start = 0,
            stop = self.tiff_image.physical_shape[1],
            num = self.tiff_image.array.shape[1]
            ) - origin

        # Iterate over rois and columns to populate the LUT.
        for roi_num, roi in enumerate(self.lut["rois"]):
            
            target_position = int(self.lut["lateral_limits"]["left"])

            for column_pixel in range(self.tiff_image.array.shape[1]):

                # Populate the LUT with None if the pixel is outside the lateral limits.
                if pixel_position[column_pixel] < self.lut["lateral_limits"]["left"] or pixel_position[column_pixel] > self.lut["lateral_limits"]["right"]:
                    rounded_position = round(pixel_position[column_pixel])
                    self.lut[(rounded_position, roi_num)] = None

                # Populate the LUT with the pixel values.
                else:
                    # Used to check if the target position is reached.
                    diff = abs(pixel_position[column_pixel] - target_position)

                    if diff <= 1/(self.lut["resolution"]/MM_PER_INCH):
                        self.lut[(target_position, roi_num)] = {
                            'I_red': int(np.mean(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel-1: column_pixel + 1, 0])),
                            'S_red': int(np.std(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel-1: column_pixel + 1, 0])),
                            'I_green': int(np.mean(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel-1: column_pixel + 1, 1])),
                            'S_green': int(np.std(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel-1: column_pixel + 1, 1])),
                            'I_blue': int(np.mean(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel-1: column_pixel + 1, 2])),
                            'S_blue': int(np.std(array_img[roi['x'] : roi['x'] + roi['height'], column_pixel-1: column_pixel + 1, 2])),
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

        self._set_roi_and_intensity_of_optical_filters()


    def compute_central_lut(self, filter: int = None) -> None:
        """
        Compute the look-up table (LUT) for the central region of the scanner.

        Parameters
        ----------
        filter : int
            If filter > 0, a median filter of size (filter, filter) is applied to 
            each channel of the scanned image prior to LUT creation.            
        """

        self._check_before_compute_lut(filter)
        self.lut['lateral_limits']['left'] = -self.tiff_image.physical_shape[1]/2
        self.lut['lateral_limits']['right'] = self.tiff_image.physical_shape[1]/2

        self.lut['lateral_correction'] = False

        # Apply a filter to the image.
        if filter:
            self.lut['filter'] = filter
            self.tiff_image.filter_channel(filter, "median", "R")
            self.tiff_image.filter_channel(filter, "median", "G")
            self.tiff_image.filter_channel(filter, "median", "B")

            array_img = self.tiff_image.array

        else: # Unfiltered image.
            array_img = self.tiff_image.array

        for roi_num, roi in enumerate(self.lut["rois"]):
            
            position = 0

            self.lut[(position, roi_num)] = {
                'I_red': int(np.mean(array_img[roi['x'] : roi['x'] + roi['height'], roi['y'] : roi['y'] + roi['width'], 0])),
                'S_red': int(np.std(array_img[roi['x'] : roi['x'] + roi['height'], roi['y'] : roi['y'] + roi['width'], 0])),
                'I_green': int(np.mean(array_img[roi['x'] : roi['x'] + roi['height'], roi['y'] : roi['y'] + roi['width'], 1])),
                'S_green': int(np.std(array_img[roi['x'] : roi['x'] + roi['height'], roi['y'] : roi['y'] + roi['width'], 1])),
                'I_blue': int(np.mean(array_img[roi['x'] : roi['x'] + roi['height'], roi['y'] : roi['y'] + roi['width'], 2])),
                'S_blue': int(np.std(array_img[roi['x'] : roi['x'] + roi['height'], roi['y'] : roi['y'] + roi['width'], 2])),
            }

            self.lut[(position, roi_num)]["I_mean"] = int(
                (
                    self.lut[(position, roi_num)]["I_red"] +
                    self.lut[(position, roi_num)]["I_green"] +
                    self.lut[(position, roi_num)]["I_blue"]
                    ) / 3
                )
            self.lut[(position, roi_num)]["S_mean"] = int(
                (
                    self.lut[(position, roi_num)]["S_red"]**2 +
                    self.lut[(position, roi_num)]["S_green"]**2 +
                    self.lut[(position, roi_num)]["S_blue"]**2
                    )**0.5 / 3
                )

        self._set_roi_and_intensity_of_optical_filters()


    def plot_lateral_response(self, channel: str = "red"):
        """
        Plot the lateral response of the scanner for each ROI (film).

        Parameters
        ----------
        channel : str
            The color channel to plot. "red", "green", "blue" or "mean".

        Examples
        --------
        >>> cal = LUT(tiff_image)
        >>> cal.set_central_rois((180, 8))
        >>> cal.compute_lateral_lut()
        >>> cal.plot_lateral_response(channel = "red")
        >>> plt.show()
        """

        # Check if lateral limits are set.
        if not self.lut["lateral_limits"]["left"]:
            raise Exception("Plotting lateral response requieres a lateral LUT. Use the compute_lateral_lut method.")

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

            intensity, std, coordinate = self.get_lateral_intensity(roi_counter, channel)

            # Normalize the pixel values to the central pixel value.
            I_central = intensity[int(len(intensity)/2)] # Central pixel value.
            I_relative = intensity / I_central * 100 - 100

            # Standar deviation of the pixel values.
            std_Ir = std / I_central * 100

            # Plot the lateral response.
            axes[roi_counter].errorbar(
                coordinate,
                I_relative,
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
        calib_intensities, std = self.get_intensities(position, channel)
        calib_intensities_curve = np.linspace(calib_intensities[0], calib_intensities[-1], endpoint=True, num=100)

        if fit_type == "rational":
            response = ratio(calib_intensities, calib_intensities[0])
            response_curve = ratio(calib_intensities_curve, calib_intensities_curve[0])
            std_response = uncertainty_ratio(calib_intensities, std, calib_intensities[0], std[0])

        elif fit_type == "polynomial":
            response = optical_density(calib_intensities, calib_intensities[0])
            response_curve = optical_density(calib_intensities_curve, calib_intensities_curve[0])
            std_response = uncertainty_optical_density(calib_intensities, std, calib_intensities[0], std[0])

        # Get the corrected doses at a given position.
        if self.lut["lateral_correction"]:
            doses = self._get_lateral_doses(position)
        else:
            doses = self.lut.get("nominal_doses")

        dose_curve, _, _ = self._get_dose_from_fit(
            calib_film_intensities = calib_intensities,
            calib_dose = doses,
            fit_function = fit_type,
            intensities = calib_intensities_curve,
        )

        if ax is None:
            fig, axe = plt.subplots()
        else: 
            axe = ax

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
        if self.lut["lateral_correction"]:
            doses = self._get_lateral_doses(position)
        else:
            doses = self.lut.get("nominal_doses")

        intensities, std = self.get_intensities(position, channel)
        uncertainty = self._get_dose_fit_uncertainty(intensities, std, doses, fit_function)
        u_percent = uncertainty[1:] / doses[1:] * 100
        
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(doses[1:], u_percent, marker = '*', linestyle = '--', color = channel, **kwargs)
        ax.set_xlabel("Dose [Gy]")
        ax.set_ylabel("Dose uncertainty [%]")


    def get_lateral_intensity(self, roi: int, channel: str) -> tuple[ndarray, ndarray, ndarray]:
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
            raise Exception("No ROIs created. Use the set_central_rois method to set the ROIs.")
        
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


    def get_intensities(
            self,
            lateral_position: float = 0,
            channel: str = "red"
        ) -> tuple[ndarray, ndarray]:
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
        tuple[ndarray, ndarray]
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


    def get_intensities_of_optical_filters(self) -> list[float]:
        """
        Return the intensities of the optical filters in the red channel.
        """
       
        return self.lut.get("intensities_of_optical_filters")
    

    def get_rois_of_optical_filters(self) -> list[dict]:
        """
        Return the rois of the optical filters as a list of dictionaries.
        [
            {
                'x': int,  # The x coordinate (row) of the top-left corner of the ROI.
                'y': int,  # The y coordinate (column) of the top-left corner of the ROI.
                'radius': int,
            }
        ]
        """
        return self.lut.get("rois_for_optical_filters")
        

    def get_interpolated_intensities_at_position(self, position: float, channel: str) -> ndarray:
        """
        Get the interpolated pixel values at a given lateral position.

        Parameters
        ----------
        position : float
            The lateral position in milimeters.
        channel : str
            The color channel. "red", "green" or "blue".

        Returns
        -------
        ndarray
            An array with the interpolated pixel values at the given lateral position.
        """
        # Check if the channel is valid.
        if channel.lower() not in ["red", "green", "blue", "mean"]:
            raise Exception("Invalid channel. Choose between 'red', 'green', 'blue' or 'mean'.")
        
        rounded_floor_position = math.floor(position)
        rounded_ceil_position = math.ceil(position)

        # Check if position is between the lateral limits.
        if rounded_floor_position < self.lut["lateral_limits"]["left"] or rounded_ceil_position > self.lut["lateral_limits"]["right"]:
            raise Exception("Position out of lateral limits.")
        

        cal_intensities_ceil, _ = self.get_intensities(
                        lateral_position = rounded_ceil_position,
                        channel = channel,
                    )
        cal_intensities_floor, _ = self.get_intensities(
                lateral_position = rounded_floor_position,
                channel = channel,
            )
        
        calibration_intensities = np.array([
                np.interp(
                    position,
                    [rounded_floor_position, rounded_ceil_position],
                    [cal_intensities_floor[i], cal_intensities_ceil[i]],
                )
                for i in range(len(cal_intensities_floor))
            ])

        return calibration_intensities


    def get_interpolated_doses_at_position(self, position: float) -> ndarray:
        """
        Get the interpolated doses at a given lateral position.

        Parameters
        ----------
        position : float
            The lateral position in milimeters.

        Returns
        -------
        ndarray
            An array with the interpolated doses at the given lateral position.
        """
        
        rounded_floor_position = math.floor(position)
        rounded_ceil_position = math.ceil(position)

        # Check if position is between the lateral limits.
        if rounded_floor_position < self.lut["lateral_limits"]["left"] or rounded_ceil_position > self.lut["lateral_limits"]["right"]:
            raise Exception("Position out of lateral limits.")
        
        cal_doses_ceil = self._get_lateral_doses(position=rounded_ceil_position)
        cal_doses_floor = self._get_lateral_doses(position = rounded_floor_position)
        
        # Interpolate
        calibration_doses = np.array([
                np.interp(
                    position,
                    [rounded_floor_position, rounded_ceil_position],
                    [cal_doses_floor[i], cal_doses_ceil[i]],
                )
                for i in range(len(cal_doses_floor))
            ])

        return calibration_doses


    def to_yaml_file(self, path: str):
        """
        Save the calibration data to a YAML file.

        Parameters
        ----------
        path : str
            The path to the file.
        """
        path_file = path + ".yaml"
        print(f"Saving lut to: {path_file}")
        with open(path_file, mode = "wt", encoding = "utf-8") as file:
            yaml.dump(self.lut, file)


    @classmethod
    def from_yaml_file(cls, path: str) -> LUT:
        """
        Load the calibration data from a YAML file.

        Parameters
        ----------
        path : str
            The path to the file.
        """
        with open(path, mode = "r", encoding = "utf-8") as file:
            cal = LUT()
            cal.lut = yaml.full_load(file)
        return cal


    def _check_before_compute_lut(self, filter: int) -> None:
        """
        Check if the conditions are met before computing the LUT.
        """
        # Check if the image is loaded.
        if self.tiff_image is None:
            raise Exception("No image loaded.")
        # Check if rois are created.
        if not self.lut.get("rois"):
            raise Exception("No ROIs created. Use the set_central_rois method to set the ROIs.")
        # Check if filter is valid.
        if filter and filter < 0:
            raise Exception("Filter must be a positive integer.")
        # Check if filter is a integer.
        if filter and not isinstance(filter, int):
            raise Exception("Filter must be an integer.")
        # Check if the doses are set.


    def _plot_rois(self, ax: plt.Axes = None):
        """
        Plot the ROIs on the image.
        """
        dpmm = self.lut["resolution"]/MM_PER_INCH
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
        
        if self.lut["lateral_correction"]:
            
            # r0' = r0 - r00', change of origin.
            #print(self.lut["lateral_limits"])
            y_left_limit_pix =int((self.lut["lateral_limits"]["left"] - origin)*dpmm)
            y_right_limit_pix = int((self.lut["lateral_limits"]["right"] - origin)*dpmm)

            ax.axvline(y_left_limit_pix)
            ax.axvline(y_right_limit_pix)
        #plt.show()


    def _get_calibration_positions(self) -> list:
        """
        Get the calibration positions in milimeters.
        """
        positions = [key[0] for key in self.lut.keys() if isinstance(key, tuple)]
        return sorted(set(positions))

    
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
            np.array(self.lut['beam_profile']['positions']),
            np.array(self.lut['beam_profile']['doses']) / 100,
            )

        lateral_doses = sorted([float(dose * profile) for dose in self.lut["nominal_doses"]])

        return lateral_doses


    def _set_roi_and_intensity_of_optical_filters(self):

        if self.tiff_image.number_of_optical_filters == 0:
            return
        
        # Get the central region of each filter.
        rois = []
        intensities = []
        
        for region in regionprops(self.tiff_image.labeled_optical_filters, self.tiff_image.array[:, :, 0]):
            rois.append(
                {
                    'x': int(region.centroid[0]),
                    'y': int(region.centroid[1]),
                    'radius': int(region.axis_minor_length),
                }
            )
            intensities.append(region.intensity_mean)

        self.lut["rois_for_optical_filters"] = rois
        self.lut["intensities_of_optical_filters"] = sorted(intensities)


    @staticmethod
    def _get_dose_from_fit(
        calib_film_intensities,
        calib_dose,
        intensities,
        fit_function,
        ) -> tuple:

        if fit_function == "rational":

            calib_response = ratio(calib_film_intensities, calib_film_intensities[0])

            popt, pcov = curve_fit(
                rational_function,
                calib_response,
                calib_dose,
                p0=[0.1, 4, 4],
                maxfev=1500,
                )
            
            response = ratio(intensities, intensities[0])
            dose = rational_function(response, *popt)

        elif fit_function == "polynomial":
            
            calib_response = optical_density(calib_film_intensities, calib_film_intensities[0])
            print("_get_dose_from_fit")
            print("calib_response")
            print(calib_response)

            popt, pcov = curve_fit(
                polynomial_n,
                calib_response,
                calib_dose,
                p0=[10, 35, 2.5],
                maxfev=1500,
                )
            
            response = optical_density(intensities, intensities[0])
            dose = polynomial_n(response, *popt)

        return dose, popt, np.sqrt(np.diag(pcov))


    @staticmethod
    def _get_dose_fit_uncertainty(
        calib_film_intensities: ndarray,
        std_calib_film_intensities: ndarray,
        calib_doses: list,
        fit_function: str) -> tuple:
        """
        Get the dose uncertainty realated to fit.

        Parameters
        ----------
        calib_film_intensities : ndarray
            The intensities of each film used for calibraiton.
        std_calib_film_intensities : ndarray
            The standard deviation of the intensities of each film used for calibration.
        calib_doses : list
            The doses used to expose the films for calibration.
        fit_function : str
            The type of fit to use. "rational" or "polynomial".

        Returns
        -------
        ndarray
            Dose fit uncertanties
        """

        #intensities_calibration, std = self.get_intensities(position, channel)
        if fit_function == "rational":

            response_cal = ratio(calib_film_intensities, calib_film_intensities[0])

            popt, pcov = curve_fit(
                rational_function,
                response_cal,
                calib_doses,
                p0=[0.1, 4.0, 4.0],
                maxfev=1500,
                )

            a = popt[0]
            b = popt[1]
            ua = np.sqrt(np.diag(pcov))[0]
            ub = np.sqrt(np.diag(pcov))[1]

            std_response = uncertainty_ratio(
                calib_film_intensities,
                std_calib_film_intensities,
                calib_film_intensities[0],
                std_calib_film_intensities[0]
                )
            u_exp = b*std_response/(response_cal-a)**2
            u_fit = np.sqrt( (b*ua/(response_cal-a)**2)**2 + (ub/(response_cal-a))**2 )
            u_dose = np.sqrt( u_exp**2 + u_fit**2 )


        elif fit_function == "polynomial":

            response_cal = optical_density(calib_film_intensities, calib_film_intensities[0])

            popt, pcov = curve_fit(
                polynomial_n,
                response_cal,
                calib_doses,
                p0=[10, 35, 2.5],
                maxfev=1500,
                )

            a = popt[0]
            b = popt[1]
            n = popt[2]
            ua = np.sqrt(np.diag(pcov))[0]
            ub = np.sqrt(np.diag(pcov))[1]

            std_response = uncertainty_optical_density(
                calib_film_intensities,
                std_calib_film_intensities,
                calib_film_intensities[0],
                std_calib_film_intensities[0]
                )
            u_exp = (a + n*b*response_cal**(n-1))*std_response
            u_fit = np.sqrt( response_cal**2*ua**2 + response_cal**(2*n)*ub**2 )
            u_dose = np.sqrt( u_exp**2 + u_fit**2 )


        return u_dose
