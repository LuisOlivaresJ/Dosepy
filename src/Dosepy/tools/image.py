"""This module holds classes for image loading and manipulation.
The content is heavily based from pylinac (https://pylinac.readthedocs.io/en/latest/_modules/pylinac/core/image.html),
with some modification to focus on tff files used for film dosimetry.
"""

from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import os.path as osp
from typing import Any, Union
from tifffile import TiffFile
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square, erosion
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

from calibration import polymonial_g3, Calibration

MM_PER_INCH = 25.4

ImageLike = Union["ArrayImage", "TiffImage", "CalibImage"]

def load(path: str | Path | np.ndarray, for_calib: bool = False) -> "ImageLike":
    r"""Load a TIFF image or numpy 2D array.

    Parameters
    ----------
    path : str, file-object
        The path to the image file or data stream or array.

    for_calib : bool, default = False
        True if the image is going to be used to get a calibration curve.

    Returns
    -------
    ::class:`~dosepy.image.BaseImage`

    Examples
    --------
    Load an image from a file::

        >>> from dosepy.image import load
        >>> path_to_image = r"C:\QA\image.tif"
        >>> img = load(path_to_image)  # returns a TiffImage

    Loading from an array is just like loading from a file::

        >>> arr = np.arange(36).reshape(6, 6)
        >>> img = load(arr)  # returns a ArrayImage
    """
    if isinstance(path, BaseImage):
        return path

    if _is_array(path):
        return ArrayImage(path)
    elif _is_image_file(path):
        if for_calib:
            return CalibImage(path)
        else:
            return TiffImage(path)
    else:
        raise TypeError(
            f"The argument `{path}` was not found to be a valid TIFF file or an array."
        )

def _is_array(obj: Any) -> bool:
    """Whether the object is a numpy array."""
    return isinstance(obj, np.ndarray)

def _is_image_file(path: str | Path) -> bool:
    """Whether the file is a readable image file via tifffile."""
    try:
        TiffFile(path)
        return True
    except:
        return False


class BaseImage:
    """Base class for the Image classes.

    Attributes
    ----------
    path : str
        The path to the image file.
    array : numpy.ndarray
        The actual image pixel array.
    """

    array: np.ndarray
    path: str | Path

    def __init__(self, path: str | Path | np.ndarray):
        """
        Parameters
        ----------
        path : str
            The path to the image.
        """

        if isinstance(path, (str, Path)) and not osp.isfile(path):
            raise FileExistsError(
                f"File `{path}` does not exist. Verify the file path name."
            )
        else:
            self.path = path
            self.base_path = osp.basename(path)


class TiffImage(BaseImage):
    """An image from a tiff file.

    Attributes
    ----------
    sid : float
        The SID value as passed in upon construction.
    tag : dict
        All tags in the TIFF file.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dpi: float | None = None,
        sid: float | None = None,
    ):
        """
        Parameters
        ----------
        path : str, file-object
            The path to the file or a data stream.
        dpi : int, float
            The dots-per-inch of the image, defined at isocenter.

            .. note:: If a X and Y Resolution tag is found in the image, that value will override the parameter, 
                otherwise this one will be used.
        sid : int, float
            The Source-to-Image distance in mm.
        tag : dict
            All tags in the TIFF file.
        """
        super().__init__(path)
        with TiffFile(path) as tif:
            self.array = tif.asarray()
            try:
                self.tags = {tag.name: tag.value for tag in tif.pages[0].tags}

                if self.tags['XResolution'][0] != self.tags['YResolution'][0]:
                    raise Exception(
                        """
                        XResolution should be equal to YResolution. The value of x and y was: 
                        {}""".format(self.tags['XResolution'][0], self.tags['YResolution'][0])
                        )
                else:
                    dpi = self.tags['XResolution'][0]
                
            except AttributeError:
                pass

        self._dpi = dpi
        self.sid = sid

    @property
    def dpi(self) -> float | None:
        """The dots-per-inch of the image, defined at isocenter."""

        if self.sid is not None:
            dpi *= self.sid / 1000
        else:
            dpi = self._dpi
        return dpi

    @property
    def dpmm(self) -> float | None:
        """The Dots-per-mm of the image, defined at isocenter. E.g. if an EPID image is taken at 150cm SID,
        the dpmm will scale back to 100cm."""
        try:
            return self.dpi / MM_PER_INCH
        except TypeError:
            return

    def region_properties(self, film_detect = True, crop = 8, channel = "R"):
        """Measure properties of films used for calibration.

        Parameters
        ----------
        film_detect : str
            Define if automatic film position detection is performed. True: The films are detected automatically. Flase: Manual user selection. 
        crop : int = 8
            Removes all edges of the image in-place (in milimeters).
        channel : str
            "R": Red, "G": Green, "B": Blue, "Mean": Arithmetic mean from the three channels.

        Returns
        -------
        ::class:`~skimage.measure.regionprops`
            Measure properties of labeled image regions. For more information see: 
            https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

        Examples
        --------
        Load an image from a file::

        >>> from dosepy.image import load
        >>> path_to_image = r"C:\QA\image.tif" 
        >>> cal_image = load(path_to_image, for_calib = True)  # returns a TiffImage used for calibration.
        >>> regions = cal_image.region_properties(crop = 8, channel = "G")
        >>> for region in regions:
        ...     print(region.intensity_mean)    # Print the value with the mean intensity in the region.
        """

        gray_scale = rgb2gray(self.array)
        thresh = threshold_otsu(gray_scale)
        # Apply threshold and close small holes with binary closing.
        # remove_pixels = 75dpi * 3 mm, is used to close smaller holes than 3 mm.
        pixels_to_remove_holes = int(self.dpmm * 3)
        binary = closing(gray_scale < thresh, square(pixels_to_remove_holes))
        # remove artifacts connected to image border
        cleared = clear_border(binary)
        # label image regions
        lb = label(cleared)
        pixels_to_remove_border = int(self.dpmm * crop)
        label_image = erosion(lb, square(pixels_to_remove_border))
        #regions = regionprops(label_image, np.mean(self.array, axis = 2))
        if channel == "R":
            regions = regionprops(label_image, self.array[:,:,0])
        elif channel == "G":
            regions = regionprops(label_image, self.array[:,:,1])
        elif channel == "B":
            regions = regionprops(label_image, self.array[:,:,2])
        elif channel == "Mean":
            regions = regionprops(label_image, np.mean(self.array, axis = 2))
        else:
            raise Exception(
                        """
                        {} is not a valid channel. Use "R" for red, "G" for green or "B" for blue.
                        """.format(channel)
                        )
        
        return regions

    def plot(
        self, ax: plt.Axes = None, show: bool = True, clear_fig: bool = False, **kwargs
        ) -> plt.Axes:
        """Plot the image.

        Parameters
        ----------
        ax : matplotlib.Axes instance
            The axis to plot the image to. If None, creates a new figure.
        show : bool
            Whether to actually show the image. Set to false when plotting multiple items.
        clear_fig : bool
            Whether to clear the prior items on the figure before plotting.
        kwargs
            kwargs passed to plt.imshow()
        """
        if ax is None:
            fig, ax = plt.subplots()
        if clear_fig:
            plt.clf()
        ax.imshow(self.array/np.max(self.array), **kwargs)
        #ax.imshow(self.array[:,:,0], **kwargs)
        if show:
            plt.show()
        return ax

class ArrayImage(BaseImage):
    """An image constructed solely from a numpy array."""

    def __init__(
        self,
        array: np.ndarray,
        *,
        dpi: float = None,
        sid: float = None,
        dtype=None,
    ):
        """
        Parameters
        ----------
        array : numpy.ndarray
            The image array.
        dpi : int, float
            The dots-per-inch of the image, defined at isocenter.

            .. note:: If a DPI tag is found in the image, that value will override the parameter, otherwise this one
                will be used.
        sid : int, float
            The Source-to-Image distance in mm.
        dtype : dtype, None, optional
            The data type to cast the image data as. If None, will use whatever raw image format is.
        """
        if dtype is not None:
            self.array = np.array(array, dtype=dtype)
        else:
            self.array = array
        self._dpi = dpi
        self.sid = sid

    @property
    def dpmm(self) -> float | None:
        """The Dots-per-mm of the image, defined at isocenter. E.g. if an EPID image is taken at 150cm SID,
        the dpmm will scale back to 100cm."""
        try:
            return self.dpi / MM_PER_INCH
        except:
            return

    @property
    def dpi(self) -> float | None:
        """The dots-per-inch of the image, defined at isocenter."""
        dpi = None
        if self._dpi is not None:
            dpi = self._dpi
            if self.sid is not None:
                dpi *= self.sid / 1000
        return dpi


class CalibImage(TiffImage):
    """A tiff image used for calibration."""

    def __init__(self, path: str | Path):
        """
        Parameters
        ----------
        path : str, file-object
            The path to the file.
        """
        super().__init__(path)
        self.calibration_curve_computed = False    

    def get_calibration(self, doses: list, func = "P3", channel = "R"):
        """Computes calibration curve. Use non-linear least squares to fit a function, func, to data. 
        For more information see scipy.optimize.curve_fit.

        Parameter
        ---------
        doses : list
            The doses values that were used to expose films for calibration.
        func : string
            P3: Polynomial function of degree 3.
        channel : str
            Color channel. "R": Red, "G": Green and "B": Blue. 

        Returns
        -------
        ::class:`~dosepy.calibration.Calibration`
            Instance of a Calibration class.

        Examples
        --------
        Load an image from a file and compute a calibration curve using green channel::

        >>> from dosepy.image import load
        >>> path_to_image = r"C:\QA\image.tif" 
        >>> cal_image = load(path_to_image, for_calib = True)
        >>> cal = cal_image.get_calibration(doses = [0, 0.5, 1, 2, 4, 6, 8, 10], channel = "G")
        >>> # Plot the calibration curve
        >>> cal.plot(color = "green")
        """

        doses = sorted(doses)
        regions = self.region_properties(film_detect = True, crop = 8, channel = channel)
        # Higest intensity represents lowest dose.
        intensities = sorted([properties.intensity_mean for properties in regions], reverse = True)
        optical_density = -np.log10(intensities/intensities[0])
        
        return Calibration(doses, optical_density, func = func, channel = channel)

        