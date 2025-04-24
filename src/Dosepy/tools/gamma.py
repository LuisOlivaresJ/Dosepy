import copy

import numpy as np
from scipy.ndimage import sobel, zoom

from Dosepy.image import ArrayImage, MM_PER_INCH


def is_close(val: float, target: float, delta: float):
    """Return whether the value is neat the target."""
    if target - delta < val < target + delta:
        return True
    return False


class GammaManager:
    
    def gamma(
        method,
        reference: ArrayImage,
        evaluated: ArrayImage,
        params: dict,
     ):
        """
        Parameters
        ----------

        method : str
            ExhaustiveSearch: Compute gamma for every pixel.
            Chi: Computes comparision based on `Bakai et al <http://iopscience.iop.org/0031-9155/48/21/006/>`_ eq.6,
        params: {dose_ta: value, dist_ta: value, }
        """
        # TODO
        if method == "ExhaustiveSearch":
            return evaluated.gamma2D(
                 dose_ta=3,
                 dist_ta=3
            )[0]
        
        if method == "FastGamma_Chi":
             return chi(
                  reference_image=reference,
                  comparison_image=evaluated,
             )
        

def chi(
        reference_image: ArrayImage,
        comparison_image: ArrayImage,
        doseTA: float = 3,
        distTA: float = 3,
        threshold: float = 10,
        interpolate: bool = True,
    ) -> np.ndarray:
        """Calculate the gamma between the current image (reference) and a comparison image.
        Adapted from pylinac.

        The gamma calculation is based on `Bakai et al
        <http://iopscience.iop.org/0031-9155/48/21/006/>`_ eq.6,
        which is a quicker alternative to the standard Low gamma equation.

        Parameters
        ----------
        reference_image : {:class:`~Dosepy.image.ArrayImage`}
            The reference image.
        comparison_image : {:class:`~Dosepy.image.ArrayImage`}
            The comparison image. The image must have the same DPI/DPMM to be comparable.
            The size of the images must also be the same.
        doseTA : int, float
            Dose-to-agreement in percent; e.g. 2 is 2%.
        distTA : int, float
            Distance-to-agreement in mm.
        threshold : float
            The dose threshold percentage of the maximum dose, below which is not analyzed.
            Must be between 0 and 100.
        interpolate : bool
            If True the arrays are zoomed using spline interpolation to have a spatial resolution of dpmm = 1.

        Returns
        -------
        gamma_map : numpy.ndarray
            The calculated gamma map.

        """
        # error checking
        if not is_close(reference_image.dpmm, comparison_image.dpmm, delta=1):
            raise AttributeError(
                f"The image resolution do not match: {reference_image.dpi:.2f} vs. {comparison_image.dpi:.2f}"
            )
        same_x = is_close(reference_image.shape[1], comparison_image.shape[1], delta=1.1)
        same_y = is_close(reference_image.shape[0], comparison_image.shape[0], delta=1.1)
        if not (same_x and same_y):
            raise AttributeError(
                f"The images are not the same size: {reference_image.shape} vs. {comparison_image.shape}"
            )

        # interpolate reference and comparison images to have a 
        # spatial resolution of dpmm = 2 (dpi = 50.8)
        new_dpmm = 2
        if interpolate:
            ref_img = ArrayImage(
                zoom(reference_image.array, zoom = new_dpmm/reference_image.dpmm),
                dpi=new_dpmm*MM_PER_INCH,
            )

            comp_img = ArrayImage(
                zoom(comparison_image.array, zoom = new_dpmm/comparison_image.dpmm),
                dpi=new_dpmm*MM_PER_INCH,
            )

        # convert distance value from mm to pixels
        distTA_pixels = reference_image.dpmm * distTA

        # Calculate doseTa
        doseTA_Gray = doseTA/100 * np.amax(ref_img.array)

        # construct image gradient using sobel filter
        img_x = np.gradient(ref_img.array, axis=1)
        img_y = np.gradient(ref_img.array, axis=0)
        grad_img = np.hypot(img_x, img_y)

        # equation: (measurement - reference) / sqrt ( doseTA^2 + distTA^2 * image_gradient^2 )
        subtracted_img = np.abs(comp_img.array - ref_img.array)
        denominator = np.sqrt(
            ((doseTA_Gray) ** 2) + ((distTA_pixels**2) * (grad_img**2))
        )

        chi_map = subtracted_img / denominator

        # invalidate dose values below threshold
        chi_map[ref_img.array < threshold/100 * np.max(ref_img.array)] = np.nan

        # Number of values that are not np.nan
        total_points = np.sum(~np.isnan(chi_map))

        # Pass rate
        chi_less_than_1 = np.sum(chi_map <= 1)
        chi_rate = chi_less_than_1/total_points*100

        return chi_map, chi_rate