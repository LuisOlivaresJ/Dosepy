"""
NAME
    Image module

DESCRIPTION
    This module holds functionalities for tif image loading and manipulation.
    The main function is load. Some common methods are get_stat, to_dose and
    get_calibration. The content is heavily based from pylinac
    (https://pylinac.readthedocs.io/en/latest/_modules/pylinac/core/image.html),
    and omg_dosimetry
    https://omg-dosimetry.readthedocs.io/en/latest/

"""

from pathlib import Path
import numpy as np
import os.path as osp
from typing import Any, Union
import imageio.v3 as iio

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import square, erosion
from skimage.measure import label, regionprops
from skimage.filters.rank import mean

#from Dosepy.tools.calibration import polynomial_g3, rational_func, Calibration
from calibration import polynomial_g3, rational_func, Calibration

MM_PER_INCH = 25.4

ImageLike = Union["ArrayImage", "TiffImage", "CalibImage"]


def load(path: str | Path | np.ndarray,
         for_calib: bool = False,
         filter: int | None = None,
         **kwargs) -> "ImageLike":
    r"""Load a TIFF image or numpy 2D array.

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
        See :class:`~dosepy.image.ArrayImage`, :class:`~dosepy.image.TiffImage`,
        or :class:`~dosepy.image.CalibImage` for keyword arguments.

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
        array_image = ArrayImage(path, **kwargs)
        if isinstance(filter, int):
            array_image.array = mean(array_image.array, footprint=square(filter))
        return array_image

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

        self.label_img = None
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
            threshold=None
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
        >>> mean, std = cal_image.get_stat(
                ch = 'G',
                roi=(5,5),
                show = True
                )
        >>> list(zip(mean, std))

        """

        if not self.label_img:
            self.set_labeled_img(threshold = threshold)

        if show:
            fig, axes = plt.subplots(ncols=1)
            #ax = axes.ravel()
            axes = plt.subplot(1, 1, 1)
            #axes.imshow(gray_scale, cmap="gray")
            axes.imshow(self.label_img, cmap="gray")

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

    def to_dose(self, cal) -> ImageLike:
        """Convert the tiff image to a dose distribution. The tiff file image
        has to contain an unirradiated film used as a reference for zero Gray.

        Parameters
        ----------
        cal : Dosepy.calibration.Calibration
            Instance of a Calibration class

        Returns
        -------
        ImageLike : ArrayImage
            Dose distribution.
        """
        mean_pixel, _ = self.get_stat(ch=cal.channel, roi=(5, 5), show=False)
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

        return load(dose_image, dpi=self.dpi)

    def doses_in_central_rois(self, cal, roi, show):
        """Dose in central film rois.

        Parameters
        ----------
        cal : dosepy.calibration.Calibration
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
            thresh = threshold
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
        threshold=None
    ):
        r"""Computes calibration curve. Use non-linear least squares to
        fit a function, func, to data. For more information see
        scipy.optimize.curve_fit.

        Parameter
        ---------
        doses : list
            The doses values that were used to expose films for calibration.
        func : string
            "P3": Polynomial function of degree 3,
            "RF" or "Rational": Rational function.
        channel : str
            Color channel. "R": Red, "G": Green and "B": Blue, "M": mean.
        roi : tuple
            Width and height region of interest (roi) in millimeters, at the
            center of the film.

        Returns
        -------
        ::class:`~dosepy.calibration.Calibration`
            Instance of a Calibration class.

        Examples
        --------
        Load an image from a file and compute a calibration curve using green
        channel::

        >>> from dosepy.image import load
        >>> path_to_image = r"C:\QA\image.tif"
        >>> cal_image = load(path_to_image, for_calib = True)
        >>> cal = cal_image.get_calibration(
            doses = [0, 0.5, 1, 2, 4, 6, 8, 10],
            channel = "G")
        >>> # Plot the calibration curve
        >>> cal.plot(color = "green")
        """

        doses = sorted(doses)
        mean_pixel, _ = self.get_stat(ch=channel, roi=roi, threshold=threshold)
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
        """The Dots-per-mm of the image, defined at isocenter. E.g. if an EPID
        image is taken at 150cm SID, the dpmm will scale back to 100cm."""
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


    def save_as_tif(self, file_name):
        """Used to save a dose distribution (in Gy) as a tif file (in cGy).
        
        Parameters
        ----------
        file_name : str
            File name as a string

        """
        data = np.int64(self.array*100) # Gy to cGy
        np_tif = data.astype(np.uint16)
        tif_encoded = iio.imwrite(
            "<bytes>",
            np_tif,
            extension=".tif",
            resolution = (self.dpi, self.dpi),
            )
        with open(file_name, 'wb') as f:
            f.write(tif_encoded)

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
