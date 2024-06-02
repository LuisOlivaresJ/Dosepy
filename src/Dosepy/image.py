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
import os.path as osp
from typing import Any, Union
import imageio.v3 as iio
import copy
from scipy import ndimage
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import square, erosion
from skimage.measure import label, regionprops
from skimage.filters.rank import mean
import math

from .calibration import polynomial_g3, rational_func, Calibration
from .tools.resol import equate_resolution
from .i_o import retrieve_dicom_file, is_dicom_image

MM_PER_INCH = 25.4

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
                f"File `{path}` does not exist. Verify the file path name.")
        else:
            self.path = path
            self.base_path = osp.basename(path)

    
    @property
    def physical_shape(self) -> tuple[float, float]:
        """The physical size of the image in mm."""
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
        dpi : int, float
            The dots-per-inch of the image, defined at isocenter.

            .. note:: If a X and Y Resolution tag is found in the image, that
            value will override the parameter, otherwise this one will be used.
        sid : int, float
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

        self.label_image = np.array([])
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

        if not self.label_image.any():
            self.set_labeled_img(threshold = 0.90)

        if show:
            fig, axes = plt.subplots(ncols=1)
            #ax = axes.ravel()
            axes = plt.subplot(1, 1, 1)
            #axes.imshow(gray_scale, cmap="gray")
            axes.imshow(self.array/np.max(self.array))

        #print(f"Number of images detected: {num}")

        # Films
        if ch in ["R", "Red", "r", "red"]:
            films = regionprops(self.label_image, intensity_image=self.array[:, :, 0])
        elif ch in ["G", "Green", "g", "green"]:
            films = regionprops(self.label_image, intensity_image=self.array[:, :, 1])
        elif ch in ["B", "Blue", "b", "blue"]:
            films = regionprops(self.label_image, intensity_image=self.array[:, :, 2])
        elif ch in ["M", "Mean", "m", "mean"]:
            films = regionprops(self.label_image,
                                intensity_image=np.mean(self.array, axis=2)
                                )
        else:
            print("Channel not founded")

        # Find the unexposed film.
        #mean_pixel = []
        #for film in films:
        #    mean_pixel.append(film.intensity_mean)
        #index_ref = mean_pixel.index(max(mean_pixel))
        #print(f"Index reference: {index_ref}")
        #end Find the unexposed film.

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
    
    def set_labeled_img(self, threshold=None):
        
        erosion_pix = int(6*self.dpmm)  # Number of pixels used for erosion.
        #print(f"Number of pixels to remove borders: {erosion_pix}")

        gray_scale = rgb2gray(self.array)
        if not threshold:
            thresh = threshold_otsu(gray_scale)  # Used for films identification.
        else:
            thresh = threshold * np.amax(gray_scale)
        binary = erosion(gray_scale < thresh, square(erosion_pix))
        self.label_image, self.number_of_films = label(binary, return_num=True)
        

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
        ​​for each point coincide with each other, that is, the images are registered.

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

        >>> # We generate the arrays, A and B, with the values ​​96 and 100 in all their elements.
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


def load_multiples(image_file_list, for_calib=False):
    """
    Combine multiple image files into one superimposed image.

    Parameters
    ----------
    image_file_list : list
        A list of paths to the files to be superimposed.

    Returns
    -------
    ::class:`~Dosepy.image.TiffImage`
        Instance of a TiffImage class.

    From omg_dosimetry.imageRGB load_multiples and pylinac.image
    """
    # load images
    img_list = [load(path, for_calib=for_calib) for path in image_file_list]
    first_img = img_list[0]
    
    if len(img_list) > 1:
        # check that all images are the same size
        for img in img_list:
            if img.array.shape != first_img.array.shape:
                raise ValueError("Images were not the same shape")
    
        # stack and combine arrays
        new_array = np.stack(tuple(img.array for img in img_list), axis=-1)
        combined_arr = np.mean(new_array, axis=3)
    
        # replace array of first object and return
        first_img.array = combined_arr

    return first_img

def stack_images(img_list, axis=0, padding=0):
    """
    Takes in an image list and concatenate them side by side.
    Useful for film calibration, when more than one image is needed
    to scan all gafchromic bands.
    
    Adapted from OMG_Dosimetry (https://omg-dosimetry.readthedocs.io/en/latest/)

    Parameters
    ----------
    img_list : list
        The images to be stacked. List of TiffImage objects.

    axis : int, default: 0
        The axis along which the arrays will be joined.

    padding : float, default: 0
        Add padding as a percentage (0-1) of the array size to simulate an empty space betwen films.

    Returns
    -------
    ::class:`~Dosepy.image.TiffImage`
        Instance of a TiffImage class.

    """

    first_img = copy.deepcopy(img_list[0])

    # check that all images are the same size
    for img in img_list:
        
        if img.shape[1] != first_img.shape[1]:
            raise ValueError("Images were not the same width")

    height = first_img.shape[0]
    width = first_img.shape[1]

    padding_pixels = int(height * padding)

    new_img_list = []
    
    for img in img_list:

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
    first_img.array = new_array

    return first_img

def load_folder(path):
    files = os.listdir(path)
    tif_files = []
    for file in files:
        if file.endswith(".tif") or file.endswith(".tiff"):
            tif_files.append(os.path.join(path,file))
    
    film_list = list(set([x[:-7] for x in tif_files]))
    img_list = []
    for film in film_list:
        file_list =[]
        for file in tif_files:
            if file[:-7] == film:
                file_list.append(file)
        img = load_multiples(file_list)
        img_list.append(img)
    return img_list

def equate_images(image1: ImageLike, image2: ImageLike) -> tuple[ArrayImage, ArrayImage]:
    """Crop the biggest of the two images and resize image2 to make them:
        * The same pixel dimensions
        * The same DPI

    The usefulness of the function comes when trying to compare images from different sources.
    The best example is calculating gamma. The physical
    and pixel dimensions must be normalized, the SID normalized

    Parameters
    ----------
    image1 : {:class:`~Dosepy.image.ArrayImage`, :class:`~Dosepy.image.TiffImage`}
        Must have DPI and SID.
    image2 : {:class:`~Dosepy.image.ArrayImage`, :class:`~Dosepy.image.TiffImage`}
        Must have DPI and SID.

    Returns
    -------
    image1 : :class:`~Dosepy.image.ArrayImage`
        The first image croped.
    image2 : :class:`~Dosepy.image.ArrayImage`
        The second image equated.
    """
    image1 = copy.deepcopy(image1)
    image2 = copy.deepcopy(image2)
    # crop images to be the same physical size
    # ...crop height
    physical_height_diff = image1.physical_shape[0] - image2.physical_shape[0]
    if physical_height_diff < 0:  # image2 is bigger
        img = image2  # img is a view of image2 (not a copy)
    else:
        img = image1
    pixel_height_diff = abs(int(math.floor(-physical_height_diff * img.dpmm / 2)))
    if pixel_height_diff > 0:
        print(f"Cropping {pixel_height_diff} pixels height (top and bottom) to {img.shape}")
        img.crop(pixel_height_diff, edges=("top", "bottom"))

    # ...crop width
    physical_width_diff = image1.physical_shape[1] - image2.physical_shape[1]
    if physical_width_diff > 0:
        img = image1
    else:
        img = image2
    pixel_width_diff = abs(int(math.floor(physical_width_diff * img.dpmm / 2)))
    if pixel_width_diff > 0:
        img.crop(pixel_width_diff, edges=("left", "right"))

    # make sure we have exactly the same shape
    # ...crop height
    height_diff = image1.shape[0] - image2.shape[0]
    if height_diff < 0:  # image2 is bigger
        img = image2
    else:
        img = image1
    if abs(height_diff) > 0:
        img.crop(abs(height_diff), edges='bottom')

    # ...crop width
    width_diff = image1.shape[1] - image2.shape[1]
    if width_diff > 0:
        img = image1
    else:
        img = image2
    if abs(width_diff) > 0:
        img.crop(abs(width_diff), edges='right')

    # resize images to be of the same shape
    if image2.dpmm > image1.dpmm:
        image2_array = equate_resolution(
            array=image2.array,
            array_resolution=1./image2.dpmm,
            target_resolution=1./image1.dpmm)
        image2 = load(image2_array, dpi=image1.dpi)

    elif image1.dpmm > image2.dpmm:
        zoom_factor = image1.shape[1] / image2.shape[1]
        image2_array = ndimage.interpolation.zoom(image2.as_type(float), zoom_factor)
        image2 = load(image2_array, dpi=image2.dpi * zoom_factor)

    return image1, image2