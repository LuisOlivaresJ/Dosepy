"""
NAME
    Image module

DESCRIPTION
    This module holds functionalities for tif image loading and manipulation.
    The content is heavily based from pylinac
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

from pylinac.core.io import is_dicom_image

from .calibration import polynomial_g3, rational_func, Calibration
from .i_o import retrieve_dicom_file

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
        See :class:`~dosepy.image.ArrayImage`, :class:`~dosepy.image.TiffImage`,
        or :class:`~dosepy.image.CalibImage` for keyword arguments.

    Returns
    -------
    ::class:`~dosepy.image.BaseImage`

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
        >>> mean, std = cal_image.get_stat(ch = 'G', roi=(5,5), show = True)
        >>> list(zip(mean, std))
        """

        if not self.label_image.any():
            self.set_labeled_img(threshold = threshold)

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


    def gamma2D(self,
                reference,
                dose_ta=3,
                dist_ta=3,
                *,
                dose_threshold=10,
                dose_ta_Gy=False,
                local_norm=False,
                mask_radius=10,
                max_as_percentile=True
                ):
        '''
        Cálculo del índice gamma contra una distribución de referencia.
        Se obtiene una matriz que representa los índices gamma en cada posición de la distribución de dosis,
        así como el índice de aprobación definido como el porcentaje de valores gamma que son menor o igual a 1.
        Se asume el registro de las distribuciones de dosis, es decir, que la coordenada espacial de un punto en la distribución de
        referencia es igual a la coordenada del mismo punto en la distribución a evaluar.

        Parameters
        ----------

        reference : Dosepy.image.ArrayImage
            Distribución de dosis de referencia contra la cual se realizará la comparación.
            El número de filas y columnas debe de ser igual a la distribución a evaluar (self.array).
            Lo anterior implica que las dimesiones espaciales de las distribuciones deben de ser iguales.

        dose_ta : float, default = 3
            Dose-to-agreement.
            Este valor puede interpretarse de 3 formas diferentes según los parámetros dose_ta_Gy,
            local_norm y max_as_percentil, los cuales se describen más adelante.

        dist_ta : float, default = 3
            Distance-to-agreement in mm.

        dose_threshold : float, default = 10
            Umbral de dosis, en porcentaje (0 a 100) con respecto a la dosis máxima de la 
            distribución de referencia (o al percentil 99 si max_as_percentile = TRUE). 
            Todo punto en la distribución de dosis con un valor menor al umbral
            de dosis, es excluido del análisis.
            
        dose_ta_Gy : bool, default: False
            Si el argumento es True, entonces "dose_ta" (la dosis de tolerancia) se interpreta como un valor fijo y absoluto en Gray [Gy].
            Si el argumento es False (default), "dose_ta" se interpreta como un porcentaje.

        local_norm : bool, default: False
            Si el argumento es True (normalización local), el porcentaje de dosis de tolerancia "dose_ta" se interpreta con respecto a la dosis local
            en cada punto de la distribución de referencia.
            Si el argumento es False (normalización global), el porcentaje de dosis de tolerancia "dose_ta" se interpreta con respecto al
            máximo de la distribución a evaluar.
            Notas:
            * Los argumentos dose_ta_Gy y local_norm NO deben ser seleccionados como True de forma simultánea.
            * Si se desea utilizar directamente el máximo de la distirbución, utilizar el parámetro max_as_percentile = False (ver explicación mas adelante).

        mask_radius : float, default: 10
            Distancia física en milímetros que se utiliza para acotar el cálculo con posiciones que estén dentro de una vecindad dada por mask_radius.

            Para lo anterior, se genera un área de busqueda cuadrada o "máscara" aldrededor de cada punto o posición en la distribución de referencia.
            El uso de esta máscara permite reducir el tiempo de cálculo debido al siguiente proceso:
            
                Por cada punto en la distribución de referencia, el cálculo de la función Gamma se realiza solamente
                con aquellos puntos o posiciones de la distribución a evaluar que se encuentren a una distancia relativa
                menor o igual a mask_radius, es decir, con los puntos que están dentro de la vecindad dada por mask_radius.
                La longitud de uno de los lados de la máscara cuadrada es de 2*mask_radius + 1.

            Por otro lado, si se prefiere comparar con todos los puntos de la distribución a evaluar, es suficiente con ingresar
            una distancia mayor a las dimensiones de la distribución de dosis (por ejemplo mask_radius = 1000).

        max_as_percentile : bool, default: True
            Si el argumento es True, se utiliza el percentil 99 como una aproximación del valor máximo de la
            distribución de dosis. Lo anterior permite excluir artefactos o errores en posiciones puntuales
            (de utilidad por ejemplo cuando se utiliza película radiocrómica o etiquetas puntuales en la distribución).
            Si el argumento es False, se utiliza directamente el valor máximo de la distribución a evaluar.

        Returns
        -------

        ndarray :
            Array, o matriz bidimensional con la distribución de índices gamma.

        float :
            Índice de aprobación. Se calcula como el porcentaje de valores gamma <= 1, sin incluir las posiciones
            en donde la dosis es menor al umbral de dosis.

        Notes
        -----

        Es posible utilizar el percentil 99 de la distribución de dosis como una aproximación del valor máximo.
        Esto permite evitar la posible inclusión de artefactos o errores en posiciones puntuales de la distribución
        (de utilidad por ejemplo cuando se utiliza película radiocrómica o etiquetas puntuales en la distribución).

        Se asume que ambas distribuciones a evaluar representan exactamente las mismas dimensiones físicas, y las posiciones
        espaciales para cada punto conciden entre ellas, es decir, las imagenes de cada distribución están registradas.

        No se realiza interpolación entre puntos.

        **Referencias**
        
        Para mayor información sobre los mecanismos de operación, efectividad y exactitud de la herramienta gamma consultar:

        [1] M. Miften, A. Olch, et. al. "Tolerance Limits and Methodologies for IMRT Measurement-Based
        Verification QA: Recommendations of AAPM Task Group No. 218" Medical Physics, vol. 45, nº 4, pp. e53-e83, 2018.

        [2] D. Low, W. Harms, S. Mutic y J. Purdy, «A technique for the quantitative evaluation of dose distributions,»
        Medical Physics, vol. 25, nº 5, pp. 656-661, 1998.

        [3] L. A. Olivares-Jimenez, "Distribución de dosis en radioterapia de intensidad modulada usando películas de tinte
        radiocrómico : irradiación de cerebro completo con protección a hipocampo y columna con protección a médula"
        (Tesis de Maestría) Posgrado en Ciencias Físicas, IF-UNAM, México, 2019

        Examples
        --------

        >>> # Importamos los paquetes Dosepy, así como numpy para crear matrices de ejemplo que representen dos distribuciones de dosis.
        >>> from Dosepy.image import load
        >>> import numpy as np

        >>> # Generamos las matrices, A y B, con los valores 96 y 100 en todos sus elementos.
        >>> A = np.zeros((30, 30)) + 96
        >>> B = np.zeros((30, 30)) + 100

        >>> # Generamos las distribuciones de dosis
        >>> D_ref = load(A, dpi = 25.4)
        >>> D_eval = load(B, dpi = 25.4)

        >>> # Sobre la variable D_eval, aplicamos el método gamma2D proporcionando como argumentos la distribución de referencia, D_ref, y el criterio (3 %, 1 mm).
        >>> gamma_distribution, pass_rate = D_eval.gamma2D( D_ref, 3, 1) 
        >>> print(f"El porcentaje de aporbación es: {pass_rate:.1f} %")

        Archivos en formato CSV (comma separated values)::

        >>> from Dosepy.image import load

        >>> # Cargamos los archivos "D_TPS.csv" y "D_FILM.csv"
        >>> # Los archivos de ejemplo .csv se encuentran dentro del paquete Dosepy, en la carpeta src/Dosepy/data
        >>> np_film = np.genfromtxt('../D_FILM.csv', delimiter = ",", comments = "#")
        >>> np_tps = np.genfromtxt('../D_TPS.csv', delimiter = ",", comments = "#")
        >>> d_film = load(np_film, dpi=25.4)
        >>> d_tps = load(np_tps, dpi=25.4)

        >>> # Llamamos al método gamma2D, con criterio 3 %, 2 mm.
        >>> g, pass_rate = d_tps.gamma2D(d_film, 3, 2)

        >>> # Imprimimos el resultado
        >>> print(f'El índice de aprobación es: {pass_rate:.1f} %')
        >>> plt.imshow(g, vmax = 1.4)
        >>> plt.show()
        >>> # El índice de aprobación es: 98.9 %
        '''

        #%%

        #   Verificar la ocurrencia de excepciones
        if reference.array.shape != self.array.shape:
            raise Exception("No es posible el cálculo con matrices de diferente tamaño.")

        if local_norm and dose_ta_Gy:
            raise Exception("No es posible la selección simultánea de dose_ta_Gy y local_norm.")

        if not self.dpi:
            raise Exception("La distribución no tiene asociada una resolución espacial.")

        if reference.dpi != self.dpi:
            raise Exception("No es posible el cálculo con resoluciones diferentes para cada distribución.")

        #%%

        D_ref = reference.array
        D_eval = self.array

        if max_as_percentile:
            maximum_dose = np.percentile(D_eval, 99)
        else:
            maximum_dose = np.amax(D_eval)
        print(f'Dosis máxima: {maximum_dose:.1f}')
        #  Umbral de dosis
        Dose_threshold = (dose_threshold/100)*maximum_dose
        print(f'Umbral de dosis: {Dose_threshold:.1f}')

        #   Dosis de tolerancia absoluta o relativa
        if dose_ta_Gy:
            pass
        elif local_norm:
            pass
        else:
            dose_ta = (dose_ta/100) * maximum_dose

        #   Número de pixeles que se usarán para definir una vecindad sobre la que se calculará el índice gamma.
        neighborhood = round(mask_radius*self.dpmm)

        #   Matriz que guardará el resultado del índice gamma.
        gamma = np.zeros( (self.array.shape[0], self.array.shape[1]) )




        #%%
        for i in np.arange( D_ref.shape[0] ):
            #   Código que permite incluir puntos cerca de la frontera de la distribución de dosis
            mi = -(neighborhood - max(0, neighborhood - i))
            mf = neighborhood - max(0, neighborhood - (D_eval.shape[0] - (i+1))) + 1

            for j in np.arange( D_ref.shape[1] ):
                ni = -(neighborhood - max(0, neighborhood - j))
                nf = neighborhood - max(0, neighborhood - (D_eval.shape[1] - (j+1))) + 1

                #   Para almacenar temporalmente los valores de la función Gamma por cada punto en la distribución de referencia
                Gamma = []

                for m in np.arange(mi , mf):
                    for n in np.arange(ni, nf):

                        # Distancia entre dos posiciones (en milímetros), por fila
                        dm = m*(1./self.dpmm)
                        # Distancia entre dos posiciones (en milímetros), por columna
                        dn = n*(1./self.dpmm)
                        
                        # Distancia total entre dos puntos
                        distance = np.sqrt(dm**2 + dn**2)

                        # Diferencia en dosis
                        dose_dif = D_eval[i + m, j + n] - D_ref[i,j]


                        if local_norm:
                            # La dosis de tolerancia se actualiza al porcentaje con respecto al valor
                            # de dosis local en la distribución de referencia.
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

                # Para la posición en cuestión, si la dosis es menor al umbral de dosis,
                # entonces dicho punto no se toma en cuenta en el porcentaje de aprobación.
                if D_eval[i,j] < Dose_threshold:
                    gamma[i,j] = np.nan

        # Arroja las coordenadas en donde los valores gamma son menor o igual a 1
        less_than_1_coordinate = np.where(gamma <= 1)
        # Cuenta el número de coordenadas en donde se cumple que gamma <= 1
        less_than_1 = np.shape(less_than_1_coordinate)[1]
        # Número de valores gamma diferentes de np.nan
        total_points = np.shape(gamma)[0]*np.shape(gamma)[1] - np.shape(np.where(np.isnan(gamma)))[1]

        #   Índice de aprobación
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
