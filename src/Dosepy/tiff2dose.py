from skimage.filters.rank import median
from skimage.morphology import square

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

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

        # Convert image to dose one column at a time
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


