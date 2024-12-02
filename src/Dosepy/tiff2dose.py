from skimage.filters.rank import median
from skimage.morphology import square

import numpy as np
import matplotlib.pyplot as plt

from Dosepy.calibration import LUT, MM_PER_INCH
from Dosepy.image import TiffImage

import math

class Tiff2Dose:
    """ Tiff to dose manager to convert a tiff image to a dose map.
    Attributes
    ----------
    img : TiffImage
        The tiff image to convert to dose.
    cal : CalibrationLUT
        The lut to use for curve calibration.
    
    """
    def __init__(self, img: TiffImage, cal: LUT, zero: TiffImage = None):
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
        self.cal = cal
        self.zero_img = zero
        self.zero_intensity = None
        self.u_zero_intensity = None


    def red(self, fit_function: str):

        #high_pixels = self.img.shape[0]
        width_pixels = self.img.shape[1]

        mask, num_films = self.img.get_labeled_image(
            erosion_pix=int(6*self.cal.lut["resolution"]/MM_PER_INCH)
            )

        if self.cal.lut["filter"]:
            img_array = median(
                self.img.array[:, :, 0],
                footprint = square(self.cal.lut["filter"]),
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

        # Convert image to dose one column at a time
        for column in range(0, width_pixels):

            # Get pixel positions rounded ceil and floor for interpolation
            pix_position = pixel_positions_mm[column]
            position_ceil = math.ceil(pix_position)
            position_floor = math.floor(pix_position)

            # Get calibration intensities and calibration doses at pixel positions
            ## Get ceil and floor values to interpolate
            cal_intensities_ceil = self.cal._get_intensities(
                lateral_position=position_ceil,
                channel = "red",
            )
            cal_intensities_floor = self.cal._get_intensities(
                lateral_position=position_floor,
                channel = "red",
            )
            cal_doses_ceil = self.cal._get_lateral_doses(position=position_ceil)
            cal_doses_floor = self.cal._get_lateral_doses(position=position_floor)

            ## Interpolate values
            interp_intensities = [
                np.interp(
                    pix_position,
                    [position_floor, position_ceil],
                    [cal_intensities_floor[i], cal_intensities_ceil[i]],
                )
                for i in range(len(cal_intensities_floor))
            ]
            
            interp_doses = [
                np.interp(
                    pix_position,
                    [position_floor, position_ceil],
                    [cal_doses_floor[i], cal_doses_ceil[i]],
                )
                for i in range(len(cal_doses_floor))
            ]

            if not self.zero_img:
                # Generate a qt widget to show the image
                # The user should be able to create a roi selection
                # Add a event handler to get the roi selection when the user press enter
                # In the event handler function, change the Tiff2Dose.zero_intenstie attribute

                pass

            # Compute dose
            ## _get_dose_from_fit uses the first element of the array to normalize


        return dose
    

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


