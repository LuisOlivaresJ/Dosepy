import copy

import numpy as np
from scipy.ndimage import sobel

from Dosepy.image import ArrayImage

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
            Chi: .
        """
        if method == "ExhaustiveSearch":
            return evaluated.gamma2D()[0]
        

    def chi(
            self,
            reference_image: ArrayImage,
            comparison_image: ArrayImage,
            doseTA: float = 1,
            distTA: float = 1,
            threshold: float = 0.1,
            normalize: bool = True,
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
            normalize : bool
                Whether to normalize the images. This sets the max value of each image to the same value.

            Returns
            -------
            gamma_map : numpy.ndarray
                The calculated gamma map.

            See Also
            --------
            :func:`~pylinac.core.image.equate_images`
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

            # set up reference and comparison images
            ref_img = ArrayImage(copy.copy(reference_image.array))

            if normalize:
                ref_img.normalize()
            comp_img = ArrayImage(copy.copy(comparison_image.array))

            if normalize:
                comp_img.normalize()

            # invalidate dose values below threshold so gamma doesn't calculate over it
            ref_img.array[ref_img < threshold/100 * np.max(ref_img)] = np.nan

            # convert distance value from mm to pixels
            distTA_pixels = reference_image.dpmm * distTA

            # construct image gradient using sobel filter
            img_x = sobel(ref_img.as_type(np.float32), 1)
            img_y = sobel(ref_img.as_type(np.float32), 0)
            grad_img = np.hypot(img_x, img_y)

            # equation: (measurement - reference) / sqrt ( doseTA^2 + distTA^2 * image_gradient^2 )
            subtracted_img = np.abs(comp_img - ref_img)
            denominator = np.sqrt(
                ((doseTA / 100.0) ** 2) + ((distTA_pixels**2) * (grad_img**2))
            )
            gamma_map = subtracted_img / denominator

            return gamma_map