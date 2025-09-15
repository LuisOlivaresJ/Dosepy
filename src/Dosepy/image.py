"""
NAME
    Image module

DESCRIPTION
    This module holds functionalities for TIFF image loading and manipulation.
    ArrayImage class is used as a representation of dose distributions.
    The content is heavily based on 
    `pylinac <https://pylinac.readthedocs.io/en/latest/_modules/pylinac/core/image.html>`_,
    and `omg_dosimetry <https://omg-dosimetry.readthedocs.io/en/latest/>`_

"""

from pathlib import Path
import numpy as np
from numpy import ndarray
import os.path as osp
from typing import Any, Union, BinaryIO, Tuple
import imageio.v3 as iio
from abc import ABC, abstractmethod

import copy
import math
import logging

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, footprint_rectangle
from skimage.measure import label, regionprops
from skimage.filters.rank import mean
from skimage.transform import rotate
from .tools.resol import equate_resolution

from .tools.array_utils import filter_array

from .i_o import retrieve_dicom_file, is_dicom_image

MM_PER_INCH = 25.4
MIN_AREA_FOR_FILMS = 400  # 20x20 mm^2

FILE_TYPE = "file"
STREAM_TYPE = "stream"

ImageLike = Union["ArrayImage", "TiffImage"]

logging.getLogger(__name__)


def load(path: str | Path | np.ndarray | BinaryIO, **kwargs) -> ImageLike:
    r"""Load a DICOM image, TIFF image, or numpy 2D array.

    Parameters
    ----------
    path : str, file-object
        The path to the image file or array.

    kwargs
        See :class:`~Dosepy.image.ArrayImage` or :class:`~Dosepy.image.TiffImage`,
        for keyword arguments.

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
            array_image.array = mean(array_image.array, footprint=footprint_rectangle((filter, filter)))
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
    
    elif _is_tif_file(path):
        return TiffImage(path, **kwargs)

    else:
        raise TypeError(f"The argument '{path}' was not found to be\
                        a valid file.")


def load_multiples(files: list[str | Path | BinaryIO]) -> ImageLike:
    """
    Load multiple images into a single one. 
    Equate TIFF files to have the same array size.
    Average images with the same file name and stack imges with different name.
     
    Parameters
    ----------
    files : list[str]
        List of paths to the images.
        
    Returns
    -------
    ImageLike
        The merged image.
    """
    
    if len(files) == 1:
        return load(files[0])
    
    else:
        # Load images
        images = []
        for file in files:
            images.append(load(file))
        
        img = load(files[0]) # Placeholder
        equated_images = equate_array_size(images, axis=("width"))
        averaged_images = average_tiff_images(equated_images)
        stacked = stack_images(averaged_images, padding=6)
        img.array = stacked.array
        return img


def _is_array(obj: Any) -> bool:
    """Whether the object is a numpy array."""
    return isinstance(obj, np.ndarray)


def _is_dicom(path: str | Path) -> bool:
    """Whether the file is a readable DICOM file via pydicom."""
    return is_dicom_image(file=path)


def _is_tif_file(path: str | Path | BinaryIO) -> bool:
    """Whether the file is an RGB TIFF image file."""
    try:
        img_props = iio.improps(path)
        if len((img_props.shape)) == 3 and img_props.shape[2] == 3:
            return True
    except:
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
    base_path: str  # Name of the file
    source: str

    def __init__(self, path: str | Path | np.ndarray):
        """
        Parameters
        ----------
        path : str
            The path to the image.
        """

        # Check for a file
        if isinstance(path, (str, Path)) and not osp.isfile(path):
            raise FileExistsError(
                f"File `{path}` does not exist. Verify the file path name.")
        
        elif isinstance(path, (str, Path)) and osp.isfile(path):
            self.path = path
            self.base_path = osp.basename(path)
            self.source = FILE_TYPE
        
        else:
            self.source = STREAM_TYPE

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
        """Flip the image array upside down. Wrapper for np.flipud()."""
        self.array = np.flipud(self.array)

    def fliplr(self) -> None:
        """Flip the image array in the left/right direction. Wrapper for np.fliplr()."""
        self.array = np.fliplr(self.array)

    def rotate(self, angle: float, mode: str = "edge", *args, **kwargs):
        """Rotate the image counter-clockwise. Simple wrapper for scikit-image. See https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate.
        All parameters are passed to that function."""
        self.array = rotate(self.array, angle, mode=mode, *args, **kwargs)

    def center(self) -> tuple[float, float]:
        """
        Return the center position of the image array as a tuple.
        Even-length arrays will return the midpoint between the central two indices. Odd will return the central index.
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
            The threshold value used to detect films. Pixel values below the threshold are considered films.
            If None, the Otsu method is used to define a threshold.
        
        Returns
        -------
        ndarray : 
            The labeled image, where all connected regions are assigned the same integer value.
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
        binary = erosion(gray_scale < thresh, footprint_rectangle((erosion_pix, erosion_pix)))

        labeled_image, number_of_films = label(binary, return_num=True)

        return labeled_image, number_of_films


class TiffImage(BaseImage):
    """An image from a TIFF file.

    Attributes
    ----------
    sid : float
        The SID value as passed in upon construction.
    props : imageio.core.v3_plugin_api.ImageProperties
        Image properties via imageio.v3.
    """

    def __init__(
        self,
        path: str | Path | BinaryIO,
        *,
        dpi: float | None = None,
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
        label_img : numpy.adarray
            Label image regions.
        """
        super().__init__(path)
        self.props = iio.improps(path, extension=".tif")
        if not self.props:
            logging.WARNING("It was not possible to read image properties.")
            print("It was not possible to read image properties.")
        self.array = iio.imread(path, extension=".tif")

        try:
            dpi = self.props.spacing[0]

        except:
            logging.WARNING("Image properties have no spacing attributes.")
            print("Image properties have no spacing attributes.")
            pass

        self._dpi = dpi

        # Use set_labeled_films_and_filters() method to fill these attributes.
        self._is_labeled = False
        self.labeled_films = None
        self.labeled_optical_filters = None

        self.number_of_films = None
        self.number_of_optical_filters = None


    @property
    def dpi(self) -> float | None:
        """The dots-per-inch of the image, defined at isocenter."""

        return self._dpi


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

        This method sets the following class attributes: labeled_films, number_of_films, 
        labeled_optical_filters, and number_of_filters.
        """

        # Get labeled objects
        labeled_objects, number_of_objects = self.get_labeled_objects(return_num = True)

        # Set the labeled films and filters
        films = np.copy(labeled_objects)
        
        filters = np.copy(labeled_objects)

        min_area_in_pixels = int(MIN_AREA_FOR_FILMS * (1/MM_PER_INCH)**2 * (self.dpi)**2)
        
        properties = regionprops(label_image = labeled_objects)

        film_counter = 0
        filter_counter = 0
        for n, p in enumerate(properties, start = 1):
            
            if p.area < min_area_in_pixels:
                filter_counter += 1
                films[films == n] = 0
            else:
                film_counter += 1
                filters[filters == n] = 0

        self.labeled_films = label(films)
        self.number_of_films = film_counter
        self.labeled_optical_filters = label(filters)
        self.number_of_optical_filters = filter_counter

        self._is_labeled = True


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
            The threshold values used to detect objects.
            The first value is used as a threshold for dark regions (< 0.1) and the second value for bright regions (> 0.9).
        min_area : float
            The minimum area in mm^2 of a region to be considered an object.
        show : bool
            If True, the image and histogram are shown.

        Returns
        -------
        labeled_img : ndarray
            Image with labeled regions.
        num_labels : int
            Number of labeled regions if return_num is True.
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
        bi_img_filtered = skimage.morphology.binary_erosion(
            binary_img,
            mode="min",
            footprint=footprint_rectangle((3, 3))
            )

        # Get labeled regions and properties
        labeled_img = label(bi_img_filtered)
        properties = regionprops(label_image = labeled_img)

        # Remove objects with area less than 10 x 10 mm^2
        ## What is the number of pixels in 10 x 10 mm^2?
        ## [10 mm (1 inch / 25.4 mm) (300 pixels / 1 inch)]**2
        ##               ^- mm to inch       ^- inch to pixels

        minimum_area = int(min_area * (1/MM_PER_INCH)**2 * (self.dpi)**2)

        film_counter = 0  # Used to reset label number

        for n, p in enumerate(properties, start = 1):
            
            if p.area < minimum_area:  # Remove small regions
                labeled_img[labeled_img == n] = 0

            else:
                film_counter += 1
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
        r"""Get average and standard deviation from pixel values at a central ROI in each film.

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

        if not self._is_labeled:
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
            Whether to actually show the image. Set to False when plotting
            multiple items.
        clear_fig : bool
            Whether to clear the prior items on the figure before plotting.
        kwargs
            kwargs passed to plt.imshow().
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
    

    def filter_channel(
        self,
        size: float | int = 0.05,
        kind: str = "median",
        channel: str = "red"
    ) -> None:
        """Apply a filter to the given channel.

        Parameters
        ----------
        size : int, float
            Size of the median filter to apply.
            If a float, the size is the ratio of the length. Must be in the range 0-1.
            E.g., if size=0.1 for a 1000-element array, the filter will be 100 elements.
            If an int, the filter is the size passed.
        kind : {'median', 'gaussian'}
            The kind of filter to apply. If gaussian, *size* is the sigma value.
        channel : {'R', 'G', 'B'}
            The color channel to filter.

        Notes
        -----
        This function was adapted from the `pylinac` library filter function.
        https://github.com/jrkerns/pylinac/blob/f16b70a1c70e15061211c853942296287cb865d3/pylinac/core/image.py#L618
        """
        if channel in ["R", "Red", "r", "red"]:
            self.array[:, :, 0] = filter_array(self.array[:, :, 0], size=size, kind=kind)
        elif channel in ["G", "Green", "g", "green"]:
            self.array[:, :, 1] = filter_array(self.array[:, :, 1], size=size, kind=kind)
        elif channel in ["B", "Blue", "b", "blue"]:
            self.array[:, :, 2] = filter_array(self.array[:, :, 2], size=size, kind=kind)
        else:
            raise ValueError("Channel not supported. Use 'red', 'green' or 'blue'.")
        
    
    def get_optical_filters(self) -> dict:
        """
        Return the ROIs and mean intensities of the optical filters in the red channel.

        Returns
        -------
        dict
            A dictionary with the ROIs as a dict and intensities as an array of the optical filters.

        Example
        -------
        >>> from Dosepy.image import load
        >>> path_to_image = "image.tif"
        >>> img = load(path_to_image)
        >>> optical_filters = img.get_optical_filters()

        >>> optical_filters["rois_for_optical_filters"]
        >>> optical_filters["intensities_of_optical_filters"]
        """

        optical_filters = {}

        if not self._is_labeled:
            self.set_labeled_films_and_filters()

        if self.number_of_optical_filters == 0:
            print("Optical filters not found")
        
        # Get the central region of each filter.
        rois = []
        intensities = []
        
        for region in regionprops(self.labeled_optical_filters, self.array[:, :, 0]):

            rois.append(
                {
                    'x': int(region.centroid[0]),
                    'y': int(region.centroid[1]),
                    'radius': int(region.axis_minor_length/2)
                }
            )
            intensities.append(region.intensity_mean)

        optical_filters["rois_for_optical_filters"] = rois
        optical_filters["intensities_of_optical_filters"] = sorted(intensities)

        return optical_filters
            

class ArrayImage(BaseImage):
    """An image constructed solely from a numpy array."""

    def __init__(
        self,
        array: np.ndarray,
        *,
        dpi: float = None,
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

        return self._dpi

    @property
    def physical_shape(self):
        """The physical size of the image in mm."""
        return self.shape[0] / self.dpmm, self.shape[1] / self.dpmm


    def save_as_tif(self, file_name):
        """Used to save a dose distribution (in Gy) as a TIFF file (in cGy).
        
        Parameters
        ----------
        file_name : str
            File name as a string.
        """
        np_tif = self.array.astype(np.float32)
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
        local_dose=False,
        mask_radius=10,
        max_as_percentile=True,
        exclude_above=None
        ) -> Tuple[ndarray, float]:
        '''
        Calculate gammas between the current image and a reference image.
        A tuple is ​​obtained. First elmenet represents the gamma indices at each position of the reference dose distribution.
        The second element represents the approval rate defined as the percentage of gamma values ​​that are less or equal to 1.

        The images must have the same spatial resolution (dpi) and array size.
        The registration of dose distributions is assumed, i.e. the spatial coordinate of a point in the
        reference dose distribution is equal to the coordinate of the same point in the distribution to be evaluated.

        Parameters
        ----------
        reference ::class:`~Dosepy.image.ArrayImage`
            The reference image.

        dose_ta : float, default = 3
            Dose tolerance as a percentage (0 - 100).

        dist_ta : float, default = 3
            Distance tolerance (distance-to-agrement) in mm.

        dose_threshold : float, default = 10
            Dose threshold in percentage (0 to 100) with respect to the maximum dose of the
            evaluated distribution (or to 99th percentile if max_as_percentile = True).
            Any point in the dose distribution with a value less than the dose threshold,
            is excluded from the analysis.
            
        dose_ta_Gy : bool, default: False
            If True, then dose_ta is interpreted as an absolute value in Gray.
            If False (default), dose_ta is interpreted as a percentage.

        local_dose : bool, default: False
            If True, the dose tolerance percentage (dose_ta) is interpreted with respect to the local dose
            (dose at the reference distribution).
            If the argument is False, the dose tolerance percentage is interpreted with respect to the
            maximum of the distribution to be evaluated. The maximum from reference is not used because of film uncertainties.
            * The dose_ta_Gy and local_dose arguments must NOT be selected as True simultaneously.
            * If you want to use the maximum of the distribution directly, use the parameter max_as_percentile = False (see explanation below).

        mask_radius : float, default: 10
            Physical distance in millimeters used to limit the calculation to positions that are within a neighborhood given by mask_radius.

            The use of this mask allows reducing the calculation time due to the following process:
            
                For each point in the reference distribution, the calculation of the Gamma function is performed only
                with those points or positions of the evaluated distribution that are at a relative distance
                less than or equal to mask_radius, that is, with the points that are within the neighborhood given by mask_radius.
                The length of one side of the square mask is 2*mask_radius + 1.

            On the other hand, if you prefer to compare with all the points of the distribution to be evaluated, it is enough to enter
            a distance greater than the dimensions of the dose distribution (for example mask_radius = 1000).

        max_as_percentile : bool, default: True
            If the argument is True, 99th percentile is used as an approximation of the maximum value of the
            evaluated dose distribution. This allows us to exclude artifacts.
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

        if local_dose and dose_ta_Gy:
            raise AttributeError(
                "Simultaneous selection of dose_ta_Gy and local_dose is not possible."
                )

        if not self.dpi:
            raise AttributeError(
                "The distribution has no associated spatial resolution."
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

        # Absolute or relative tolerance dose
        if not (dose_ta_Gy or local_dose):
            dose_ta = (dose_ta/100) * maximum_dose

        # Number of pixels that will be used to define a neighborhood on the evaluated dose distribution
        # over which the gamma index will be calculated.
        neighborhood = round(mask_radius*self.dpmm)

        # Array that will store the result of the gamma index.
        gamma = np.zeros( (self.array.shape[0], self.array.shape[1]) )



        #%%
        # Perform gamma for each point in reference distribution
        for i in np.arange( D_ref.shape[0] ):
            # The next two lines allows to exlude evaluated points outside the image if the reference point is 
            # near a border.
            mi = -(neighborhood - max(0, neighborhood - i))
            mf = neighborhood - max(0, neighborhood - (D_eval.shape[0] - (i+1))) + 1

            for j in np.arange( D_ref.shape[1] ):
                ni = -(neighborhood - max(0, neighborhood - j))
                nf = neighborhood - max(0, neighborhood - (D_eval.shape[1] - (j+1))) + 1

                # Place holder for the Gamma function values ​​at
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


                        if local_dose:
                            # The tolerance dose is updated as a percentage 
                            # of local dose in the reference distribution.
                            dose_t_local = dose_ta * D_ref[i,j] / 100

                            Gamma.append(
                                np.sqrt(
                                    (distance**2)/(dist_ta**2) + (dose_dif**2)/(dose_t_local**2)
                                )
                            )

                        else :
                            Gamma.append(
                                np.sqrt(
                                    (distance**2)/(dist_ta**2) + (dose_dif**2)/(dose_ta**2)
                                )
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
        Reduce the spatial resolution of the image to match a reference image. Useful for gamma analysis.
        The physical dimensions of the images must be the same (within half of the reference resolution).
        The algorithm averages a number of pixels given by image_resolution [dpi] / reference_resolution [dpi].
        For example, if the image comes from a TIFF file with a resolution of 75 dpi, and the reference image
        comes from a treatment planning system with a resolution of 25.4 dpi (1 point per millimeter), the rounded
        number of pixels to average is 75 / 25.4 = 3.

        Parameters
        ----------
        reference ::class:`~Dosepy.image.ArrayImage`
            The reference image that has the target resolution.

        Raises
        ------
        AttributeError
            If the physical dimensions of the images are not the same.

        Examples
        --------
        Create two images with different resolution, same physical dimensions, and reduce the resolution of one of them::
        
        >>> from Dosepy.image import load
        >>> import numpy as np

        >>> # Generate the arrays, eval and ref.
        >>> eval = np.random.rand(100, 100)
        >>> ref = np.random.rand(10, 10)

        >>> # Create the dose distributions.
        >>> D_eval = load(eval, dpi=10)
        >>> D_ref = load(ref, dpi=1)

        >>> # Reduce the resolution of the image D_eval to match the resolution of D_ref.
        >>> D_eval.reduce_resolution_as(D_ref)

        >>> # Print the new shape of the D_eval array.
        >>> print(D_eval.shape)  # (10, 10)
        """

        # Check that reference has a smaller resolution
        if reference.dpi > self.dpi:
            raise AttributeError(
                "The reference image must have a smaller resolution than the image to be reduced."
                )
        elif reference.dpi == self.dpi:
            print("The spatial resolution of both images are the same.")
            return

        ## Check if the physical dimensions are the same within a tolerance
        if not math.isclose(self.physical_shape[0], reference.physical_shape[0], abs_tol = 1./reference.dpmm/2):
            raise AttributeError(
                f"The physical dimensions of the images are not the same. The shapes are {self.physical_shape[0]} and {reference.physical_shape[0]}"
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


def equate_array_size(
        image_list: list,
        axis: tuple[str, ...] = ("height", "width"),
        ) -> list:
    """
    Equate TIFF files to have the same array size with respect to the smallest one.
    Pixels are cropped equally from both sides.

    Parameters
    ----------
    image_list : list
        List with images (TiffImage, ArrayImage instance).

    axis : str
        Axis to equate: height, width, or both.

    Returns
    -------
    list
        A list with the new images.
    """
    
    cropped_images = copy.deepcopy(image_list)
    idx_min_height, idx_min_width = _find_smallest_image(image_list)

    if "height" in axis:
        for count, img in enumerate(image_list):
            if count == idx_min_height: continue
            cropped_images[count] = _equate_height(image_list[idx_min_height], img)
            
    image_list = cropped_images
    if "width" in axis:

        for count, img in enumerate(image_list):
            if count == idx_min_width: continue
            cropped_images[count] = _equate_width(image_list[idx_min_width], img)

    return cropped_images


def average_tiff_images(images: list[TiffImage | ArrayImage]) -> list:
    """
    Average images with the same file name, ignoring the last 7 characters.

    Parameters
    ----------
    paths : list
        List of strings with the TIFF file paths.

    images : list
        List of TiffImage or ArrayImage objects.

    Returns
    -------
    averaged_images : list
        List of TiffImage or ArrayImage objects.

    Note
    ----
    Since the last 7 characters of the file name are ignored, the function
    averages the following files:

    - my_file_name_001.tif
    - my_file_name_002.tif
    - my_file_name_003.tif
    """

    # Get base_name for identification
    paths_as_str = [img.base_path for img in images]

    # Create a list with no duplicate names
    unique_names = list(set([file[:-7] for file in paths_as_str]))
    
    averaged_images = []
    for unique in unique_names:
        to_merge =[]
        buff = copy.deepcopy(images[0])

        # Catch files with same name
        for file_name, image in zip(paths_as_str, images):
            if file_name[:-7] == unique:
                to_merge.append(image)
        
        new_array = np.stack(tuple(img.array for img in to_merge), axis=-1)
        buff.array = np.mean(new_array, axis=3)
        averaged_images.append(buff)
    
    return averaged_images


def stack_images(img_list: list, axis=0, padding=0):
    """
    Takes in a list of images and concatenates them side by side.
    Useful for film calibration when more than one image is needed
    to scan several films.
    
    Adapted from OMG_Dosimetry (https://omg-dosimetry.readthedocs.io/en/latest/)

    Parameters
    ----------
    img_list : list
        The images to be stacked. List of TiffImage or ArrayImage.

    axis : int, default: 0
        The axis along which the arrays will be joined. 0 if vertical or 1 if horizontal.

    padding : float, default: 0
        Add padding in millimeters to simulate an empty space between films.

    Returns
    -------
    ::class:`~Dosepy.image.TiffImage`
        Instance of a TiffImage class.

    Example
    -------

        >>> img1 = load(np.ones((5, 5, 3)), dpi=1)
        >>> img2 = load(np.ones((5, 5, 3)), dpi=1)

        >>> img = stack_images([img1, img2])

        >>> img.shape  # (10, 5, 3)
    """

    first_img = copy.deepcopy(img_list[0])

    # Check that all images are the same width
    for img in img_list:
        
        if axis == 0:
            if img.shape[1] != first_img.shape[1]:
                raise ValueError("Images were not the same width")
        if axis == 1:
            if img.shape[0] != first_img.shape[0]:
                raise ValueError("Images were not the same height")

    #height = first_img.shape[0]
    width = first_img.shape[1]

    padding_pixels = int(padding * img_list[0].dpmm)

    new_img_list = []
    
    for img in img_list:

        height = img.shape[0]

        background = np.zeros(
            (2*padding_pixels + height, 2*padding_pixels + width, 3)
            ) + int(2**16 - 1)

        background[
            padding_pixels: padding_pixels + height,
            padding_pixels: padding_pixels + width,
            :
            ] = img.array
        new_img = copy.deepcopy(img)
        new_img.array = background
        new_img_list.append(new_img)
    
    new_array = np.concatenate(tuple(img.array for img in new_img_list), axis)
    first_img.array = new_array.astype(np.uint16)

    return first_img

def _find_smallest_image(images: list[ImageLike]):

    min_higth = images[0].shape[0]
    min_width = images[0].shape[1]

    index_min_height = 0
    index_min_width = 0

    for count, img in enumerate(images[1:], start=1):

        if img.shape[0] < min_higth:
            min_higth = img.shape[0]
            index_min_height = count

        if img.shape[1] < min_width:
            min_width = img.shape[1]
            index_min_width = count

    logging.debug(f"Smallest height: {min_higth} at index {index_min_height}")
    logging.debug(f"Smallest width: {min_width} at index {index_min_width}")
    return index_min_height, index_min_width


def _equate_height(small_image, image):
    """
    Crop the image to have the same height as the small_image. 
    If the difference is odd, the extra pixel is cropped from the top.
    Otherwise, the extra pixels are cropped equally from both sides.
    """
    logging.debug(f"Image height before cropping: {image.shape}")
    height_diff = abs(int(image.shape[0] - small_image.shape[0]))
    logging.debug(f"Height difference: {height_diff}")

    if height_diff > 0:
                
        if height_diff == 1:
            image.crop(height_diff, edges="bottom")

        elif not(height_diff%2):
            image.crop(int(height_diff/2), edges=('bottom', 'top'))

        else:
            image.crop(int(math.floor(height_diff/2)), edges="top")
            image.crop(int(math.floor(height_diff/2) + 1), edges="bottom")


    logging.debug(f"Image height after cropping: {image.shape}")
    return image


def _equate_width(small_image, image):
    """
    Crop the image to have the same width as the small_image.
    If the difference is odd, the extra pixel is cropped from the left.
    Otherwise, the extra pixels are cropped equally from both sides.
    """

    width_diff = abs(int(image.shape[1] - small_image.shape[1]))

    if width_diff > 0:

        if width_diff==1:
            image.crop(width_diff, edges="right")

        elif not(width_diff%2):
            image.crop(int(width_diff/2), edges=("left", "right"))

        else:
            image.crop(int(math.floor(width_diff/2)), edges="left")
            image.crop(int(math.floor(width_diff/2) + 1), edges="right")

    return image