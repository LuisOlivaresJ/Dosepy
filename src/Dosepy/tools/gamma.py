import copy

import numpy as np
from scipy.ndimage import sobel, zoom

from Dosepy.image import ArrayImage, MM_PER_INCH


def is_close(val: float, target: float, delta: float):
    """Return whether the value is neat the target."""
    if target - delta < val < target + delta:
        return True
    return False
    

def chi(
        reference_image: ArrayImage,
        comparison_image: ArrayImage,
        dose_ta: float = 3,
        dist_ta: float = 3,
        threshold: float = 10,
        interpolate: bool = True,
    ) -> np.ndarray:
        """Calculate the chi (analogous to gamma) between the current image (reference) and a comparison image.
        Adapted from pylinac.core.image.

        The chi calculation is based on `Bakai et al
        <http://iopscience.iop.org/0031-9155/48/21/006/>`_ eq.6,
        which is a quicker alternative to the standard Low gamma equation.

        Parameters
        ----------
        reference_image : {:class:`~Dosepy.image.ArrayImage`}
            The reference image.
        comparison_image : {:class:`~Dosepy.image.ArrayImage`}
            The comparison image. The image must have the same DPI/DPMM to be comparable.
            The size of the images must also be the same.
        dose_ta : int, float
            Dose-to-agreement in percent; e.g. 2 is 2%.
        dist_ta : int, float
            Distance-to-agreement in mm.
        threshold : float
            The dose threshold percentage of the maximum dose, below which is not analyzed.
            Must be between 0 and 100.
        interpolate : bool
            True to perfom interpolation (recommended)

        Returns
        -------
        gamma_map, chi_rate : numpy.ndarray, float
            The calculated chi map and passing rate.

        """
        # Error checking
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

        # Interpolate reference and comparison images to dpmm = 2 (dpi = 50.8).
        # This is 1/4 of 2 mm (a commonly used tolerance limit) accordingly to AAPM TG218 recomendations,
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

        # Convert distance value from mm to pixels
        dist_ta_pixels = reference_image.dpmm * dist_ta

        # Calculate dose_ta
        dose_ta_Gray = dose_ta/100 * np.amax(ref_img.array)

        # Construct image gradient
        img_x = np.gradient(ref_img.array, axis=1)
        img_y = np.gradient(ref_img.array, axis=0)
        grad_img = np.hypot(img_x, img_y)

        # Equation: (measurement - reference) / sqrt ( dose_ta^2 + dist_ta^2 * image_gradient^2 )
        subtracted_img = np.abs(comp_img.array - ref_img.array)
        denominator = np.sqrt(
            ((dose_ta_Gray) ** 2) + ((dist_ta_pixels**2) * (grad_img**2))
        )

        chi_map = subtracted_img / denominator

        # Invalidate dose values below threshold
        chi_map[ref_img.array < threshold/100 * np.max(ref_img.array)] = np.nan

        # Number of values that are not np.nan
        total_points = np.sum(~np.isnan(chi_map))

        # Pass rate
        chi_less_than_1 = np.sum(chi_map <= 1)
        chi_rate = chi_less_than_1/total_points*100

        return chi_map, chi_rate