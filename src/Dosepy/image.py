"""
NAME
    Image module

DESCRIPTION
    This module holds functionalities for tif image loading and manipulation.
    ArryaImage class is used as representation of dose distributions.
    The content is heavily based from 
    `pylinac <https://pylinac.readthedocs.io/en/latest/_modules/pylinac/core/image.html>`_, and `omg_dosimetry <https://omg-dosimetry.readthedocs.io/en/latest/>`_

"""

from pathlib import Path
import numpy as np
from numpy import ndarray
import os.path as osp
from typing import Any, Union
import imageio.v3 as iio
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import square, erosion
from skimage.measure import label, regionprops
from skimage.filters.rank import mean
from skimage.transform import rotate
from .tools.resol import equate_resolution
from .tools.files_to_image import equate_array_size
from .tools.array_utils import filter_array

import math

#from .calibration import polynomial_g3, rational_func, Calibration
from .i_o import retrieve_dicom_file, is_dicom_image

MM_PER_INCH = 25.4
MIN_AREA_FOR_FILMS = 400  # 20x20 mm^2

ImageLike = Union["ArrayImage", "TiffImage", "CalibImage"]


def load(path: str | Path | np.ndarray,
         for_calib: bool = False,
         filter: int | None = None,
         **kwargs) -> "ImageLike":
    r"""Load a DICOM image, TIF image, or numpy 2D array.

    Parameters
    ----------
    path : str, file-object
        The path to the image file or array.

    for_calib : bool, default = False
        True if the image is going to be used to get a calibration curve.

    filter : int
        If None (default), no filtering will be done to the image.
        If an int, will perform median filtering over image of size ``filter``.

    kwargs
        See :class:`~Dosepy.image.ArrayImage`, :class:`~Dosepy.image.TiffImage`,
        or :class:`~Dosepy.image.CalibImage` for keyword arguments.

    Returns
    -------
    ::class:`~Dosepy.image.ArrayImage`, :class:`~Dosepy.image.TiffImage`

    Examples
    --------
    Load an image from a file::

        >>> from Dosepy.image import load
        >>> path_to_image = r"C:\QA\image.tif"
        >>> img = load(path_to_image)  # returns a TiffImage

    Loading from an array is just like loading from a file::

        >>> arr = np.arange(36).reshape(6, 6)
        >>> img = load(arr)  # returns an ArrayImage
    """
    if isinstance(path, BaseImage):
        return path

    if _is_array(path):
        array_image = ArrayImage(path, **kwargs)
        if isinstance(filter, int):
            array_image.array = mean(array_image.array, footprint=square(filter))
        return array_image
    
    elif _is_dicom(path):
        ds = retrieve_dicom_file(path)

        array = ds.pixel_array
        #image_orientation = DS.ImageOrientationPatient
        if array.ndim != 2:
            raise Exception("The DICOM file must have 2D dose distribution.")
        
        dgs = ds.DoseGridScaling
        d_array = array * dgs
        resolution_mm = ds.PixelSpacing
        if resolution_mm[0] != resolution_mm[1]:
            raise Exception("Pixel spacing must be equal in both dimensions.")

        return ArrayImage(d_array, dpi=MM_PER_INCH/resolution_mm[0])
    
    elif _is_image_file(path):
        if _is_tif_file(path):
            if _is_RGB(path):
                if for_calib:
                    calib_image = CalibImage(path, **kwargs)
                    if isinstance(filter, int):
                        for i in range(3):
                            calib_image.array[:, :, i] = mean(
                                calib_image.array[:, :, i],
                                footprint=square(filter))
                    return calib_image
                else:
                    tiff_image = TiffImage(path, **kwargs)
                    if isinstance(filter, int):
                        for i in range(3):
                            tiff_image.array[:, :, i] = mean(
                                tiff_image.array[:, :, i],
                                footprint=square(filter))
                    return tiff_image
            else:
                raise TypeError(f"The argument '{path}' was not found to be\
                                a RGB TIFF file.")
        else:
            raise TypeError(f"The argument '{path}' was not found to be\
                            a valid TIFF file.")
    else:
        raise TypeError(f"The argument '{path}' was not found to be\
                        a valid file.")


def load_images(paths: list):
    """
    Parameters
    ----------
    paths : list
        List with the paths to the TIFF files.

    Return
    ------
    list of TIffImage
    """
    images = []

    for file in paths:
        images.append(load(file))
    
    return images


def _is_array(obj: Any) -> bool:
    """Whether the object is a numpy array."""
    return isinstance(obj, np.ndarray)


def _is_dicom(path: str | Path) -> bool:
    """Whether the file is a readable DICOM file via pydicom."""
    return is_dicom_image(file=path)


def _is_image_file(path: str | Path) -> bool:
    """Whether the file is a readable image file via imageio.v3."""
    try:
        iio.improps(path)
        return True
    except:
        return False


def _is_tif_file(path: str | Path) -> bool:
    """Whether the file is a tif image file."""
    if Path(path).suffix in (".tif", ".tiff"):
        return True
    else:
        return False


def _is_RGB(path: str | Path) -> bool:
    """Whether the image is RGB."""
    img_props = iio.improps(path)
    if len((img_props.shape)) == 3 and img_props.shape[2] == 3:
        return True
    else:
        return False


class BaseImage(ABC):
    """Base abstract class for the Image classes.

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
                f"File `{path}` does not exist. Verify the file path name.")
        else:
            self.path = path

        super().__init__()

    
    @property
    def physical_shape(self) -> tuple[float, float]:
        """
        The physical size of the image in mm.

        Returns
        -------
        tuple[float, float]
            The physical size in mm. The first element is the height, the second the width.
        """
        return self.shape[0] / self.dpmm, self.shape[1] / self.dpmm
    

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape


    def as_type(self, dtype: np.dtype) -> np.ndarray:
        return self.array.astype(dtype)


    def crop(
        self,
        pixels: int = 15,
        edges: tuple[str, ...] = ("top", "bottom", "left", "right"),
    ) -> None:
        """Removes pixels on all edges of the image in-place.

        Parameters
        ----------
        pixels : int
            Number of pixels to cut off all sides of the image.
        edges : tuple
            Which edges to remove from. Can be any combination of the four edges.
        """
        if pixels <= 0:
            raise ValueError("Pixels to remove must be a positive number")
        if "top" in edges:
            self.array = self.array[pixels:, :]
        if "bottom" in edges:
            self.array = self.array[:-pixels, :]
        if "left" in edges:
            self.array = self.array[:, pixels:]
        if "right" in edges:
            self.array = self.array[:, :-pixels]


    def flipud(self) -> None:
        """Flip the image array upside down. Wrapper for np.flipud()"""
        self.array = np.flipud(self.array)

    def fliplr(self) -> None:
        """Flip the image array in the left/right direction. Wrapper for np.fliplr()"""
        self.array = np.fliplr(self.array)

    def rotate(self, angle: float, mode: str = "edge", *args, **kwargs):
        """Rotate the image counter-clockwise. Simple wrapper for scikit-image. See https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate.
        All parameters are passed to that function."""
        self.array = rotate(self.array, angle, mode=mode, *args, **kwargs)

    def center(self) -> tuple[float, float]:
        """
        Return the center position of the image array as a tuple.
        Even-length arrays will return the midpoint between central two indices. Odd will return the central index.
        """
        x_center = (self.shape[1] / 2) - 0.5
        y_center = (self.shape[0] / 2) - 0.5

        return x_center, y_center

    def get_labeled_image(
            self,
            threshold: float = None,
            erosion_pix: int = 3,
            ) -> tuple[np.ndarray, int]:
        """
        Get the labeled image of the films.
        Function used to identify the films in the image using skimage.measure.label.

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

        gray_scale = rgb2gray(self.array)

        if not threshold:
            thresh = threshold_otsu(gray_scale)  # Used for films identification.

        else:
            thresh = threshold * np.amax(gray_scale)

        # Number of pixels used for erosion. 
        # Used to remove the irregular borders of the films.
        # https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_erosion

        #erosion_pix = int(6*self.lut["resolution"]/MM_PER_INCH)  # 6 mlimiters.
        binary = erosion(gray_scale < thresh, square(erosion_pix))

        labeled_image, number_of_films = label(binary, return_num=True)

        return labeled_image, number_of_films


class TiffImage(BaseImage):
    """An image from a tiff file.

    Attributes
    ----------
    sid : float
        The SID value as passed in upon construction.
    props : imageio.core.v3_plugin_api.ImageProperties
        Image properties via imageio.v3.
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
        dpi : float
            The dots-per-inch of the image, defined at isocenter.

            .. note:: If a X and Y Resolution tag is found in the image, that
            value will override the parameter, otherwise this one will be used.
        sid : float
            The Source-to-Image distance in mm.
        label_img : numpy.adarray
            Label image regions.
        """
        super().__init__(path)
        self.props = iio.improps(path)
        self.array = iio.imread(path)

        try:
            dpi = self.props.spacing[0]

        except AttributeError:
            pass

        self._dpi = dpi
        self.sid = sid

        self.labeled_films = np.array([])
        self.number_of_films = None


    @property
    def dpi(self) -> float | None:
        """The dots-per-inch of the image, defined at isocenter."""

        dpi = None

        if self.sid is not None:
            dpi *= self.sid / 1000
        else:
            dpi = self._dpi

        return dpi


    @property
    def dpmm(self) -> float | None:
        """The Dots-per-mm of the image, defined at isocenter. E.g. if an EPID
        image is taken at 150cm SID, the dpmm will scale back to 100cm."""
        try:
            return self.dpi / MM_PER_INCH
        except TypeError:
            return


    def set_labeled_films_and_filters(self):
        """
        Set the labeled films and optical filters in the image.
        """

        # Get labeled objects
        labeled_objects, number_of_objects = self.get_labeled_objects(return_num = True)

        # Set the labeled films and filters
        #film_counter = 0
        films = np.copy(labeled_objects)
        #filter_counter = 0
        filters = np.copy(labeled_objects)

        min_area_in_pixels = int(MIN_AREA_FOR_FILMS * (1/MM_PER_INCH)**2 * (self.dpi)**2)
        
        properties = regionprops(label_image = labeled_objects)

        for n, p in enumerate(properties, start = 1):
            
            if p.area < min_area_in_pixels:
                #filter_counter += 1
                films[films == n] = 0
            else:
                #film_counter += 1
                filters[filters == n] = 0

        self.labeled_films = label(films)
        self.labeled_optical_filters = label(filters)


    def get_labeled_objects(
        self,
        return_num: bool = False,
        threshold: tuple[float, float] = (0.1, 0.8),
        min_area: float = 100,
        show: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, int]:
        """
        Get a labeled array and the number of regions in an image.

        Parameters
        ----------
        return_num : bool
            If True, the number of labeled regions is returned.
        threshold : tuple
            The threshold values used to detect film.
            The first value is used as a threshold for dark regions (< 0.1) and the second value for bright regions (> 0.9).
        min_area : float
            The minimum area in mm^2 of a region to be considered a film.
        show : bool
            If True, the image and histogram are shown.

        Returns
        -------
        labeled_img : ndarray
            Image with labeled regions
        num_labels : int
            Number of labeled regions if return_num is True

        """
        # Convert to HSV
        hsv_img = rgb2hsv(self.array)
        #h = hsv_img[:, :, 0]
        #s = hsv_img[:, :, 1]
        v = hsv_img[:, :, 2]

        # Get binary with thresholding
        ## If v is LOW, it is a DARK region
        ## If v is HIGH, it is a BRIGHT region
        binary_img = np.logical_and(
            v > threshold[0],
            v < threshold[1]
            )

        # Filter for small bright spots
        ## The erosion_pix parameter is set to 6 times the resolution in mm.
        #erosion_pix = int(6*self.dpi/MM_PER_INCH)
        bi_img_filtered = skimage.morphology.binary_erosion(
            binary_img,
            mode="min",
            footprint=square(3)
            )

        # Get labeled regions and properties
        labeled_img = label(bi_img_filtered)
        properties = regionprops(label_image = labeled_img)

        # Remove objects with area less than 10 x 10 mm^2
        ## What is the number of pixels in 10 x 10 mm^2?
        ## [10 mm (1 inch / 25.4 mm) (300 pixels / 1 inch)]**2
        ##               ^- mm to inch       ^- inch to pixels

        minimum_area = int(min_area * (1/MM_PER_INCH)**2 * (self.dpi)**2)
        print(f"The minimum area in pixels is: {minimum_area}")

        film_counter = 0  # Used to reset label number

        for n, p in enumerate(properties, start = 1):
            
            if p.area < minimum_area:  # Remove small regions
                labeled_img[labeled_img == n] = 0

            else:
                film_counter += 1
                print(f"Object num. {film_counter}")
                labeled_img[labeled_img == n] = film_counter
                

        # Plot histogram if show is True
        if show:
            fig = plt.figure(tight_layout=True)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.imshow(v)
            ax1.set_title("Value")
            ax2.hist(v.ravel(), 512)

        if return_num:
            return (labeled_img, film_counter)
        else:
            return labeled_img


    def get_stat(
            self,
            ch='G',
            roi=(5, 5),
            show=False,
            ) -> list:
        r"""Get average and standar deviation from pixel values at a central ROI in each film.

        Parameter
        ---------
        ch : str
            Color channel. "R": Red, "G": Green, "B": Blue and "M": mean.
        field_in_film : bool
            True to show the rois used in the image.
        roi : tuple
            Width and height of a region of interest (roi) in millimeters, at the
            center of the film.
        show : bool
            Whether to actually show the image and rois.

        Returns
        -------
        list
            mean, std

        Examples
        --------
        Load an image from a file and compute a calibration curve using
        green channel::

        >>> from dosepy.image import load
        >>> path_to_image = r"C:\QA\image.tif"
        >>> cal_image = load(path_to_image, for_calib = True)
        >>> mean, std = cal_image.get_stat(ch = 'G', roi=(5,5), show = True)
        >>> list(zip(mean, std))
        """

        if not self.labeled_films.any():
            self.set_labeled_films_and_filters()

        if show:
            fig, axes = plt.subplots(ncols=1)
            axes = plt.subplot(1, 1, 1)
            axes.imshow(self.array/np.max(self.array))         

        # Films
        if ch in ["R", "Red", "r", "red"]:
            films = regionprops(self.labeled_films, intensity_image=self.array[:, :, 0])
        elif ch in ["G", "Green", "g", "green"]:
            films = regionprops(self.labeled_films, intensity_image=self.array[:, :, 1])
        elif ch in ["B", "Blue", "b", "blue"]:
            films = regionprops(self.labeled_films, intensity_image=self.array[:, :, 2])
        elif ch in ["M", "Mean", "m", "mean"]:
            films = regionprops(self.labeled_films,
                                intensity_image=np.mean(self.array, axis=2)
                                )
        else:
            print("Channel not founded")

        mean = []
        std = []

        height_roi_pix = int(roi[1]*self.dpmm)
        #print(f"height_roi: {height_roi_pix}")
        width_roi_pix = int(roi[0]*self.dpmm)
        #print(f"width_roi: {width_roi_pix}")

        for film in films:
            x0, y0 = film.centroid
            #print(f"(x0: {x0}, y0: {y0}")

            # Used to get film rectangle.
            #minr_film, minc_film, maxr_film, maxc_film  = film.bbox

            minc_roi = int(y0 - width_roi_pix/2)
            minr_roi = int(x0 - height_roi_pix/2)

            if ch in ["R", "Red", "r", "red"]:
                roi = self.array[
                    int(x0 - height_roi_pix/2): int(x0 + height_roi_pix/2),
                    int(y0 - width_roi_pix/2): int(y0 + width_roi_pix/2),
                    0,
                ]
            elif ch in ["G", "Green", "g", "green"]:
                roi = self.array[
                    int(x0 - height_roi_pix/2): int(x0 + height_roi_pix/2),
                    int(y0 - width_roi_pix/2): int(y0 + width_roi_pix/2),
                    1,
                ]
            elif ch in ["B", "Blue", "b", "blue"]:
                roi = self.array[
                    int(x0 - height_roi_pix/2): int(x0 + height_roi_pix/2),
                    int(y0 - width_roi_pix/2): int(y0 + width_roi_pix/2),
                    2,
                ]
            elif ch in ["M", "Mean", "m", "mean"]:
                array = np.mean(self.array, axis=2)
                roi = array[
                    int(x0 - height_roi_pix/2): int(x0 + height_roi_pix/2),
                    int(y0 - width_roi_pix/2): int(y0 + width_roi_pix/2)
                ]

            mean.append(int(np.mean(roi)))
            std.append(int(np.std(roi)))

            if show:
                rect_roi = mpatches.Rectangle(
                    (minc_roi, minr_roi),
                    width_roi_pix,
                    height_roi_pix,
                    fill=False,
                    edgecolor='red',
                    linewidth=1,
                )

                axes.add_patch(rect_roi)

        if show:
            plt.show()

        return mean, std


    def plot(
        self,
        ax: plt.Axes = None,
        show: bool = True,
        clear_fig: bool = False,
        **kwargs
    ) -> plt.Axes:
        """Plot the image.

        Parameters
        ----------
        ax : matplotlib.Axes instance
            The axis to plot the image to. If None, creates a new figure.
        show : bool
            Whether to actually show the image. Set to false when plotting
            multiple items.
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


    def to_dose(self, cal, clip=False):
        """Convert the tiff image to a dose distribution. The tiff file image
        has to contain an unirradiated film used as a reference for zero Gray.

        Parameters
        ----------
        cal : :class:`~Dosepy.calibration.Calibration`
            Instance of a Calibration class

        clip : bool, default: False
            If True, limit the maximum dose to the greatest used for calibration. Useful to avoid very high doses.

        Returns
        -------
        :class:`~Dosepy.image.ArrayImage`
            Dose distribution.
        """
        mean_pixel, _ = self.get_stat(
            ch=cal.channel,
            roi=(5, 5),
            show=False,
            )
        mean_pixel = sorted(mean_pixel, reverse=True)

        if cal.channel in ["R", "Red", "r", "red"]:
            if cal.func in ["P3", "Polynomial"]:
                x = -np.log10(self.array[:, :, 0]/mean_pixel[0])
            elif cal.func in ["RF", "Rational"]:
                x = self.array[:, :, 0]/mean_pixel[0]

        elif cal.channel in ["G", "Green", "g", "green"]:
            if cal.func in ["P3", "Polynomial"]:
                x = -np.log10(self.array[:, :, 1]/mean_pixel[0])
            elif cal.func in ["RF", "Rational"]:
                x = self.array[:, :, 1]/mean_pixel[0]

        elif cal.channel in ["B", "Blue", "b", "blue"]:
            if cal.func in ["P3", "Polynomial"]:
                x = -np.log10(self.array[:, :, 2]/mean_pixel[0])
            elif cal.func in ["RF", "Rational"]:
                x = self.array[:, :, 2]/mean_pixel[0]

        elif cal.channel in ["M", "Mean", "m", "mean"]:
            array = np.mean(self.array, axis=2)
            if cal.func in ["P3", "Polynomial"]:
                x = -np.log10(array/mean_pixel[0])
            elif cal.func in ["RF", "Rational"]:
                x = self.array/mean_pixel[0]

        if cal.func in ["P3", "Polynomial"]:
            dose_image = polynomial_g3(x, *cal.popt)
        elif cal.func in ["RF", "Rational"]:
            dose_image = rational_func(x, *cal.popt)

        dose_image[dose_image < 0] = 0  # Remove unphysical doses < 0

        if clip:  # Limit the maximum dose
            max_calib_dose = cal.doses[-1]
            dose_image[dose_image > max_calib_dose] = max_calib_dose

        return load(dose_image, dpi=self.dpi)


    def doses_in_central_rois(self, cal, roi, show):
        """Dose in central film rois.

        Parameters
        ----------
        cal : Dosepy.calibration.Calibration
            Instance of a Calibration class
        roi : tuple
            Width and height of a region of interest (roi) in millimeters (mm), at the
            center of the film.
        show : bool
            Whether to actually show the image and rois.

        Returns
        -------
        array : numpy.ndarray
            Doses on heach founded film.
        """
        mean_pixel, _ = self.get_stat(ch=cal.channel, roi=roi, show=show)
        #
        if cal.func in ["P3", "Polynomial"]:
            mean_pixel = sorted(mean_pixel)  # Film response.
            optical_density = -np.log10(mean_pixel/mean_pixel[0])
            dose_in_rois = polynomial_g3(optical_density, *cal.popt)

        elif cal.func in ["RF", "Rational"]:
            # Pixel normalization 
            mean_pixel = sorted(mean_pixel, reverse = True)
            norm_pixel = np.array(mean_pixel)/mean_pixel[0]
            dose_in_rois = rational_func(norm_pixel, *cal.popt)

        return dose_in_rois
    

    def filter_channel(
        self,
        size: float | int = 0.05,
        kind: str = "median",
        channel: str = "R"
    ) -> None:
        """Apply a filter to the given channel.

        Parameters
        ----------
        size : int, float
            Size of the median filter to apply.
            If a float, the size is the ratio of the length. Must be in the range 0-1.
            E.g. if size=0.1 for a 1000-element array, the filter will be 100 elements.
            If an int, the filter is the size passed.
        kind : {'median', 'gaussian'}
            The kind of filter to apply. If gaussian, *size* is the sigma value.
        channel : {'R', 'G', 'B'}
            The color channel to filter

        Notes
        -----
        This function was adapted from the `pylinac` library filter fuction.
        https://github.com/jrkerns/pylinac/blob/f16b70a1c70e15061211c853942296287cb865d3/pylinac/core/image.py#L618
        """
        if channel in ["R", "Red", "r", "red"]:
            self.array[:, :, 0] = filter_array(self.array[:, :, 0], size=size, kind=kind)
        elif channel in ["G", "Green", "g", "green"]:
            self.array[:, :, 1] = filter_array(self.array[:, :, 1], size=size, kind=kind)
        elif channel in ["B", "Blue", "b", "blue"]:
            self.array[:, :, 2] = filter_array(self.array[:, :, 2], size=size, kind=kind)
        else:
            raise ValueError("Channel not suported. Use 'R', 'G' or 'B'.")
        

## TODO Delete this class
class CalibImage(TiffImage):
    """A tiff image used for calibration."""

    def __init__(self, path: str | Path, **kwargs):
        """
        Parameters
        ----------
        path : str, file-object
            The path to the file.

        dpi : float
            The dots-per-inch of the image, defined at isocenter.

            .. note:: If a DPI tag is found in the image, that value will
            override the parameter, otherwise this one will be used.
        """
        super().__init__(path, **kwargs)
        self.calibration_curve_computed = False

    def get_calibration(
        self,
        doses: list,
        func="P3",
        channel="R",
        roi=(5, 5),
        threshold=None,
        ):
        r"""Computes calibration curve. Use non-linear least squares to
        fit a function, func, to data. For more information see
        scipy.optimize.curve_fit.

        Parameter
        ---------
        doses : list
            Doses values used to expose films for calibration.
        func : string
            "P3": Polynomial function of degree 3, using optical density as film response. "RF" or "Rational": Rational function, using normalized pixel value relative to the unexposed film.
        channel : str
            Color channel. "R": Red, "G": Green and "B": Blue, "M": mean.
        roi : tuple
            Width and height region of interest (roi) in millimeters, at the
            center of the film.

        Returns
        -------
        ::class:`~Dosepy.calibration.Calibration`
            Instance of a Calibration class.

        Examples
        --------
        Load an image from a file and compute a calibration curve using green
        channel::

        >>> from Dosepy.image import load
        >>> path_to_image = r"C:\QA\image.tif"
        >>> cal_image = load(path_to_image, for_calib = True)
        >>> cal = cal_image.get_calibration(doses = [0, 0.5, 1, 2, 4, 6, 8, 10], channel = "G")
        >>> # Plot the calibration curve
        >>> cal.plot()
        """

        doses = sorted(doses)
        mean_pixel, _ = self.get_stat(ch=channel, roi=roi)
        mean_pixel = sorted(mean_pixel, reverse=True)
        mean_pixel = np.array(mean_pixel)

        if func in ["P3", "Polynomial"]:
            x = -np.log10(mean_pixel/mean_pixel[0])  # Optical density

        elif func in ["RF", "Rational"]:
            x = mean_pixel/mean_pixel[0]

        return Calibration(y=doses, x=x, func=func, channel=channel)
    

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

            .. note:: If a DPI tag is found in the image, that value will
            override the parameter, otherwise this one will be used.
        sid : int, float
            The Source-to-Image distance in mm.
        dtype : dtype, None, optional
            The data type to cast the image data as. If None, will use
            whatever raw image format is.
        """
        if dtype is not None:
            self.array = np.array(array, dtype=dtype)
        else:
            self.array = array
        self._dpi = dpi
        self.sid = sid

    @property
    def dpmm(self) -> float | None:
        """The Dots-per-mm of the image, defined at isocenter."""
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

    @property
    def physical_shape(self):
        """The physical size of the image in mm."""
        return self.shape[0] / self.dpmm, self.shape[1] / self.dpmm


    def save_as_tif(self, file_name):
        """Used to save a dose distribution (in Gy) as a tif file (in cGy).
        
        Parameters
        ----------
        file_name : str
            File name as a string

        """
        np_tif = self.array.astype(np.uint16)
        #np_tif = self.array
        tif_encoded = iio.imwrite(
            "<bytes>",
            np_tif,
            extension=".tif",
            resolution = (self.dpi, self.dpi),
            )
        with open(file_name, 'wb') as f:
            f.write(tif_encoded)


    def gamma2D(self,
                reference,
                dose_ta=3,
                dist_ta=3,
                *,
                dose_threshold=10,
                dose_ta_Gy=False,
                local_norm=False,
                mask_radius=10,
                max_as_percentile=True,
                exclude_above=None
                ):
        '''
        Calculate gamma between the current image against a reference image.
        The images must have the same spatial resolution (dpi) to be comparable. The size of the images must also be the same.
        An array is ​​obtained. It represents the gamma indices at each position of the dose distribution,
        as well as the approval rate defined as the percentage of gamma values ​​that are less or equal to 1.
        The registration of dose distributions is assumed, i.e. the spatial coordinate of a point in the
        reference dose distribution is equal to the coordinate of the same point in the distribution to be evaluated.

        Parameters
        ----------
        reference ::class:`~Dosepy.image.ArrayImage`
            The reference image.

        dose_ta : float, default = 3
            Dose-to-agreement.
            This value can be interpreted in 3 different ways depending on the dose_ta_Gy,
            local_norm and max_as_percentile  parameters, which are described below.

        dist_ta : float, default = 3
            Distance-to-agreement in mm.

        dose_threshold : float, default = 10
            Dose threshold in percentage (0 to 100) with respect to the maximum dose of the
            reference distribution (or to 99th percentile if max_as_percentile = True).
            Any point in the dose distribution with a value less than the dose threshold,
            is excluded from the analysis.
            
        dose_ta_Gy : bool, default: False
            If True, then "dose_ta" (the tolerance dose) is interpreted as an absolute value in Gray.
            If False (default), "dose_ta" is interpreted as a percentage.

        local_norm : bool, default: False
            If the argument is True (local normalization), the tolerance dose percentage "dose_ta" is interpreted with respect to the local dose
            at each point of the reference distribution.
            If the argument is False (global normalization), the tolerance dose percentage "dose_ta" is interpreted with respect to the
            maximum of the distribution to be evaluated.
            * The dose_ta_Gy and local_norm arguments must NOT be selected as True simultaneously.
            * If you want to use the maximum of the distribution directly, use the parameter max_as_percentile = False (see explanation below).

        mask_radius : float, default: 10
            Physical distance in millimeters used to limit the calculation to positions that are within a neighborhood given by mask_radius.

            The use of this mask allows reducing the calculation time due to the following process:
            
                For each point in the reference distribution, the calculation of the Gamma function is performed only
                with those points or positions of the distribution to be evaluated that are at a relative distance
                less than or equal to mask_radius, that is, with the points that are within the neighborhood given by mask_radius.
                The length of one side of the square mask is 2*mask_radius + 1.

            On the other hand, if you prefer to compare with all the points of the distribution to be evaluated, it is enough to enter
            a distance greater than the dimensions of the dose distribution (for example mask_radius = 1000).

        max_as_percentile : bool, default: True
            If the argument is True, 99th percentile is used as an approximation of the maximum value of the
            dose distribution. This allows us to exclude artifacts or errors in specific positions.
            If the argument is False, the maximum value of the distribution is used.

        exclude_above : float, default: None
            Dose limit in Gy. Any point in the evaluated distribution greater than exclude_above, is not accounted in the pass rate. dose_ta_Gy should be set as True.

        Returns
        -------

        gamma_map : numpy.ndarray
            The calculated gamma distribution.

        pass_rate : float
            Approval rate. It is calculated as the percentage of gamma values ​​<= 1. 
            Points with dose below than the dose threshold are not accounted.

        Notes
        -----

        Percentile 99th of the dose distribution can be used as an approximation of the maximum value.
        This allows us to avoid artifacts or errors in specific positions of the distribution.
        (useful for example for spot labels are used in films).

        It is assumed that both distributions have exactly the same physical dimensions, and the positions
        for each point coincide with each other, that is, the images are registered.

        Interpolation is not supported yet.

        **References**
        
        For more information about the operating mechanisms, effectiveness and accuracy of the gamma tool:

        [1] M. Miften, A. Olch, et. al. "Tolerance Limits and Methodologies for IMRT Measurement-Based
        Verification QA: Recommendations of AAPM Task Group No. 218" Medical Physics, vol. 45, nº 4, pp. e53-e83, 2018.

        [2] D. Low, W. Harms, S. Mutic y J. Purdy, «A technique for the quantitative evaluation of dose distributions,»
        Medical Physics, vol. 25, nº 5, pp. 656-661, 1998.

        [3] L. A. Olivares-Jimenez, "Distribución de dosis en radioterapia de intensidad modulada usando películas de tinte
        radiocrómico : irradiación de cerebro completo con protección a hipocampo y columna con protección a médula"
        (Tesis de Maestría) Posgrado en Ciencias Físicas, IF-UNAM, México, 2019

        Examples
        --------

        Numpy arrays as dose distributions::
        
        >>> # We import the Dosepy packages as well as numpy to create example arrays representing two dose distributions.
        >>> from Dosepy.image import load
        >>> import numpy as np

        >>> # We generate the arrays, A and B, with the values 96 and 100 in all their elements.
        >>> A = np.zeros((30, 30)) + 96
        >>> B = np.zeros((30, 30)) + 100

        >>> # We generate the dose distributions
        >>> D_ref = load(A, dpi = 25.4)
        >>> D_eval = load(B, dpi = 25.4)

        >>> # On the variable D_eval, we apply the gamma2D method providing as arguments the reference distribution, D_ref, and the criteria (3%, 1 mm).
        >>> gamma_distribution, pass_rate = D_eval.gamma2D( D_ref, 3, 1) 
        >>> print(f"Pass rate: {pass_rate:.1f} %")

        CSV files (comma separated values)::

        >>> from Dosepy.image import load

        >>> # Load "D_TPS.csv" y "D_FILM.csv"
        >>> # The example .csv files are located within the Dosepy package, in the src/Dosepy/data
        >>> np_film = np.genfromtxt('../D_FILM.csv', delimiter = ",", comments = "#")
        >>> np_tps = np.genfromtxt('../D_TPS.csv', delimiter = ",", comments = "#")
        >>> d_film = load(np_film, dpi=25.4)
        >>> d_tps = load(np_tps, dpi=25.4)

        We call the method gamma2D, with criteria 3%, 2 mm::

        >>> g, pass_rate = d_tps.gamma2D(d_film, 3, 2)

        >>> # Print the result
        >>> print(f'Pass rate: {pass_rate:.1f} %')
        >>> plt.imshow(g, vmax = 1.4)
        >>> plt.show()
        >>> # Pass rate: 98.9 %
        '''

        #%%

        # error checking
        if reference.shape != self.shape:
            raise AttributeError(
                f"The images are not the same size: {self.shape} vs. {reference.shape}"
                )

        if local_norm and dose_ta_Gy:
            raise AttributeError(
                "Simultaneous selection of dose_ta_Gy and local_norm is not possible."
                )

        if not self.dpi:
            raise AttributeError(
                "The distribution has no associated spatial resolution."
                )

        if reference.dpi != self.dpi:
            raise AttributeError(
                f"The image DPIs to not match: {self.dpi:.2f} vs. {reference.dpi:.2f}"
                )

        #%%

        D_ref = reference.array
        D_eval = self.array

        if max_as_percentile:
            maximum_dose = np.percentile(D_eval, 99)
        else:
            maximum_dose = np.amax(D_eval)
        #print(f'Maximum dose: {maximum_dose:.1f}')
        #  Umbral de dosis
        Dose_threshold = (dose_threshold/100)*maximum_dose
        #print(f'Dose_threshold: {Dose_threshold:.1f}')

        # Absolute or relative dose-to-agreement
        if dose_ta_Gy:
            pass
        elif local_norm:
            pass
        else:
            dose_ta = (dose_ta/100) * maximum_dose

        # Number of pixels that will be used to define a neighborhood 
        # over which the gamma index will be calculated.
        neighborhood = round(mask_radius*self.dpmm)

        # Array that will store the result of the gamma index.
        gamma = np.zeros( (self.array.shape[0], self.array.shape[1]) )



        #%%
        for i in np.arange( D_ref.shape[0] ):
            # Code that allows including points near the border of the dose distribution
            mi = -(neighborhood - max(0, neighborhood - i))
            mf = neighborhood - max(0, neighborhood - (D_eval.shape[0] - (i+1))) + 1

            for j in np.arange( D_ref.shape[1] ):
                ni = -(neighborhood - max(0, neighborhood - j))
                nf = neighborhood - max(0, neighborhood - (D_eval.shape[1] - (j+1))) + 1

                # To temporarily store the Gamma function values ​​for 
                # each point in the reference distribution
                Gamma = []

                for m in np.arange(mi , mf):
                    for n in np.arange(ni, nf):

                        # Row physical distance (mm)
                        dm = m*(1./self.dpmm)
                        # Column physical distance (mm)
                        dn = n*(1./self.dpmm)
                        
                        # Distance between two points 
                        distance = np.sqrt(dm**2 + dn**2)

                        # Dose difference
                        dose_dif = D_eval[i + m, j + n] - D_ref[i,j]


                        if local_norm:
                            # The dose-to-agreement is updated to the percentage 
                            # with respect to the value
                            # of local dose in the reference distribution.
                            dose_t_local = dose_ta * D_ref[i,j] / 100

                            Gamma.append(
                                np.sqrt(
                                    (distance**2) / (dist_ta**2)
                                    + (dose_dif**2) / (dose_t_local**2))
                                        )

                        else :
                            Gamma.append(
                                np.sqrt(
                                    (distance**2) / (dist_ta**2)
                                    + (dose_dif**2) / (dose_ta**2))
                                        )

                gamma[i,j] = min(Gamma)

                # For the position in question, if the dose is below than the dose threshold, or above exclude_above,
                # then this point is not taken into account in the approval percentage.
                if D_eval[i,j] < Dose_threshold:
                    gamma[i,j] = np.nan

                if exclude_above:
                    if D_eval[i,j] > exclude_above:
                        gamma[i,j] = np.nan

        # Returns the coordinates where the gamma values ​​are less than or equal to 1
        less_than_1_coordinate = np.where(gamma <= 1)
        # Counts the number of coordinates where gamma <= 1 is True
        less_than_1 = np.shape(less_than_1_coordinate)[1]
        # Number of values that are not np.nan
        total_points = gamma.size - np.isnan(gamma).sum(where=True)

        # Pass rate
        gamma_percent = float(less_than_1)/total_points*100
        return gamma, gamma_percent            


    def reduce_resolution_as(self, reference):
        """
        Reduce the spatial resolution of the image to have the same a reference image. Usefull for gamma analysis.
        The physical dimensions of the images must be the same (within half of the reference resolution).
        The algorithm averages a number of pixels given by reference_resolution // image_resolution.


        Parameters
        ----------
        reference : :class:`~Dosepy.image.ArrayImage`
            The reference image that has the target resolution.

        Raises
        ------
        AttributeError
            If the physical dimensions of the images are not the same.

        Examples
        --------
        Create two images with different resolutions and reduce the resolution of one of them::
        
        >>> from Dosepy.image import load
        >>> import numpy as np

        >>> # Generate the arrays, A and B.
        >>> A = np.random.rand(100, 100)
        >>> B = np.random.rand(10, 10)

        >>> # Create the dose distributions.
        >>> D_eval = load(A, dpi = 10)
        >>> D_ref = load(B, dpi = 1)

        >>> # Reduce the resolution of the image D_eval to have the same resolution as D_ref.
        >>> D_eval.reduce_resolution_as(D_ref)

        >>> # Print the new shape of the D_eval array.
        >>> print(D_eval.shape) # (10, 10)
        """

        # Check that reference has a higher resolution
        if reference.dpi > self.dpi:
            raise AttributeError(
                "The reference image must have a higher resolution than the image to be reduced."
            )
        elif reference.dpi == self.dpi:
            print("The spatial resolution of both images is the same.")
            return

        ## Check if the physical dimensions are the same within a tolerance
        if not math.isclose(self.physical_shape[0], reference.physical_shape[0], abs_tol = 1./reference.dpmm/2):
            raise AttributeError(
                "The physical dimensions of the images are not the same."
                )

        # Average pixels to reduce resolution
        self.array = equate_resolution(
            array = self.array,
            array_resolution = 1/self.dpmm,
            target_resolution = 1./reference.dpmm
            )
        
        # Equate array shape 
        reduced_img, _ = equate_array_size([self, reference])
        self.array = reduced_img.array

        self._dpi = reference.dpi


class DoseImage(ArrayImage):
    """ A dose distribution image."""

    def __init__(
            self,
            array: np.ndarray,
            dpi: float,
            reference_point: list[float, float],
            orientation: tuple[int, int, int, int, int, int],
            dose_unit: str,
            ):
        super().__init__(array, dpi=dpi)
        self._reference_point  = reference_point
        self._orientation = orientation
        self._dose_unit = dose_unit
