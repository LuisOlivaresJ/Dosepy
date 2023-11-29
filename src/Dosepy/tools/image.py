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
import matplotlib.patches as mpatches

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square, erosion
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.filters.rank import mean

from calibration import polynomial_g3, Calibration

MM_PER_INCH = 25.4

ImageLike = Union["ArrayImage", "TiffImage", "CalibImage"]

def load(path: str | Path | np.ndarray, for_calib: bool = False, filter: int | None = None) -> "ImageLike":
    r"""Load a TIFF image or numpy 2D array.

    Parameters
    ----------
    path : str, file-object
        The path to the image file or data stream or array.

    for_calib : bool, default = False
        True if the image is going to be used to get a calibration curve.

    filter : int
        If None (default), no filtering will be done to the image.
        If an int, will perform median filtering over image of size ``filter``.

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
        array_image = ArrayImage(path)
        if isinstance(filter, int):
            array_image.array = mean(array_image.array, footprint = square(filter))
        return array_image
        
    elif _is_image_file(path):
        if for_calib:
            calib_image = CalibImage(path)
            if isinstance(filter, int):
                for i in range(3):
                    calib_image.array[:,:,i] = mean(calib_image.array[:,:,i], footprint = square(filter))
            return calib_image
        else:
            tiff_image = TiffImage(path)
            if isinstance(filter, int):
                for i in range(3):
                    tiff_image.array[:,:,i] = mean(tiff_image.array[:,:,i], footprint = square(filter))
            return tiff_image
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

    def get_stat(self, ch = 'G', field_in_film = False, ar = 0.5, show = False, threshold = None):
        """Get average and standar deviation from pixel values inside film's roi.
        
        Parameter
        ---------
        ch : str
            Color channel. "R": Red, "G": Green and "B": Blue.
        field_in_film : bool
            True to show the rois used in the image.
        ar : float
            Area ratio used to define the roi size relative to the film. Use 1 for a roi size equal to the film.
        show : bool
            Whether to actually show the image and rois.

        Returns
        -------
        list
            mean, std

        Examples
        --------
        Load an image from a file and compute a calibration curve using green channel::

        >>> from dosepy.image import load
        >>> path_to_image = r"C:\QA\image.tif" 
        >>> cal_image = load(path_to_image, for_calib = True)
        >>> mean, std = cal_image.get_stat(ch = 'G', field_in_film = True, ar = 0.4, show = True)
        >>> list(zip(mean, std))

        """

        n_crop_pix = int(6*self.dpmm) # Number of pixels used to remove film border.

        gray_scale = rgb2gray(self.array)
        if not threshold:
            thresh = threshold_otsu(gray_scale) # Used for films identification.
        else: 
            thresh = threshold
        binary = erosion(gray_scale < thresh, square(n_crop_pix))
        label_image, num = label(binary, return_num = True)

        if show == True:
                
            fig, axes = plt.subplots(ncols=1)
            #ax = axes.ravel()
            axes = plt.subplot(1, 1, 1)
            axes.imshow(gray_scale, cmap = "gray")
            
        print(f"Number of images detected: {num}")

        # Films
        if ch == "R":
            films = regionprops(label_image, intensity_image = self.array[:,:,0])
        if ch == "G":
            films = regionprops(label_image, intensity_image = self.array[:,:,1])
        if ch == "B":
            films = regionprops(label_image, intensity_image = self.array[:,:,2])
        if ch == "mean":
            films = regionprops(label_image, intensity_image = np.mean(self.array, axis = 2))

        # Find the unexposed film.
        mean_pixel = []
        for film in films:
            mean_pixel.append(film.intensity_mean)
        index_ref = mean_pixel.index(max(mean_pixel))
        #end Find the unexposed film.

        mean = []
        std = []
        
        if not field_in_film:
            for film in films:

                mean.append(film.intensity_mean)
                std.append(np.std(film.image_intensity))

        else:
            for film in films:

                minr_film, minc_film, maxr_film, maxc_film  = film.bbox # Used to get film patch rectangle.
                
                if show == True:
                    rect_film = mpatches.Rectangle(
                    (minc_film, minr_film), maxc_film - minc_film, maxr_film - minr_film,
                    fill = False, 
                    edgecolor = 'yellow',
                    linewidth = 1,
                    )

                    axes.add_patch(rect_film)
                
                film.image_intensity[film.image_intensity == 0] = np.max(film.image_intensity) # Fill white pixels.
                th = threshold_otsu(film.image_intensity)

                # Used for field detection inside the film
                bin = erosion(film.image_intensity < th)
                lb = label(bin)
                field = regionprops(lb, intensity_image = film.image_intensity)

                minr_field, minc_field, maxr_field, maxc_field  = field[0].bbox

                a = maxr_film - minr_film # Film box height
                b = maxc_film - minc_film # Film box width.
                rows_to_crop = int(0.5*a*(1-ar**0.5))
                colums_to_crop = int(0.5*b*(1-ar**0.5))

                minr_roi = int(minr_film + minr_field + rows_to_crop)
                minc_roi = int(minc_film + minc_field + colums_to_crop)
                maxr_roi = int(maxr_film - minr_field - rows_to_crop)
                maxc_roi = int(maxc_film - minc_field - colums_to_crop)

                if film.label == (index_ref + 1): # Continues with the next iteration of the loop, if unexposed film.
                    roi = film.image_intensity[rows_to_crop : -rows_to_crop, colums_to_crop : -colums_to_crop]
                else:
                    roi = field[0].image_intensity[rows_to_crop : -rows_to_crop, colums_to_crop : -colums_to_crop]

                mean.append(int(np.mean(roi)))
                std.append(int(np.std(roi)))

                if show:
                    rect_roi = mpatches.Rectangle(
                    (minc_roi, minr_roi), maxc_roi - minc_roi, maxr_roi - minr_roi,
                    fill = False, 
                    edgecolor = 'red',
                    linewidth = 1,
                    )

                    axes.add_patch(rect_roi)
        plt.show()
        
        return mean, std
    
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
    
    def to_dose(self, cal):
        mean_pixel, _ = self.get_stat(ch = cal.channel, field_in_film = False, ar = 0.4, show = False)
        mean_pixel = sorted(mean_pixel, reverse = True)

        if cal.channel == "R":                        
            optical_density = -np.log10(self.array[:,:,0]/mean_pixel[0])
        elif cal.channel == "G":
            optical_density = -np.log10(self.array[:,:,1]/mean_pixel[1])
        else:
            optical_density = -np.log10(self.array[:,:,2]/mean_pixel[2])
        
        dose_image = polynomial_g3(optical_density, *cal.popt)
        dose_image[dose_image < 0] = 0 # Remove doses < 0
        
        return dose_image

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

    def get_calibration(self, doses: list, func = "P3", channel = "R", field_in_film = False, threshold = None):
        """Computes calibration curve. Use non-linear least squares to fit a function, func, to data. 
        For more information see scipy.optimize.curve_fit.

        Parameter
        ---------
        doses : list
            The doses values that were used to expose films for calibration.
        func : string
            P3: Polynomial function of degree 3.
        channel : str
            Color channel. "R": Red, "G": Green and "B": Blue, "mean": mean.

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
        #regions = self.region_properties(film_detect = True, crop = 8, channel = channel)
        # Higest intensity represents lowest dose.
        #intensities = sorted([properties.intensity_mean for properties in regions], reverse = True)
        mean_pixel, _ = self.get_stat(ch = channel, field_in_film = field_in_film, ar = 0.4, threshold = threshold)
        mean_pixel = sorted(mean_pixel, reverse = True)
        mean_pixel = np.array(mean_pixel)
        #optical_density = -np.log10(intensities/intensities[0])
        optical_density = -np.log10(mean_pixel/mean_pixel[0])
        
        return Calibration(doses, optical_density, func = func, channel = channel)

