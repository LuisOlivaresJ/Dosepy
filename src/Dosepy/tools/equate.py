from Dosepy.image import ArrayImage
from math import floor

def crop_using_ref_position(
        img_film: ArrayImage,
        img_dicom: ArrayImage,
        row_film: int,
        column_film: int,
        row_dicom: int,
        column_dicom: int,
    ) -> tuple[ArrayImage, ArrayImage]:
    """
    Function to crop two images with differing physical sizes and spatial resolutions,
    allowing for their physical sizes to be matched.

    The function uses a reference point on both images to compute and eliminate
    any excess boundary points.

    Parameters
    ----------
    img_film : ArrayImage
        The film image to crop.
    img_dicom : ArrayImage
        The DICOM image to crop.
    row_film : int
        The row of the reference point in the film image.
    column_film : int
        The column of the reference point in the film image.
    row_dicom : int
        The row of the reference point in the DICOM image.
    column_dicom : int
        The column of the reference point in the DICOM image.
    
    Returns
    -------
    tuple[ArrayImage, ArrayImage]
        A tuple containing the cropped film and DICOM images, respectively.
    """

    # Validate the reference points
    if row_film < 0 or row_film >= img_film.shape[0]:
        raise ValueError("The reference point on the film image is out of bounds.")
    if column_film < 0 or column_film >= img_film.shape[1]:
        raise ValueError("The reference point on the film image is out of bounds.")
    if row_dicom < 0 or row_dicom >= img_dicom.shape[0]:
        raise ValueError("The reference point on the DICOM image is out of bounds.")
    if column_dicom < 0 or column_dicom >= img_dicom.shape[1]:
        raise ValueError("The reference point on the DICOM image is out of bounds.")
    

    def calculate_points_to_remove(distance_film, distance_dicom, dpi_film, dpi_dicom):
        """Specifically, if the DICOM image is larger, the
        function only crops the lowest integer values. Given that the film has a higher
        resolution, removing points from the film provides a more accurate approximation
        of its image size compared to removing points from the DICOM image.
        
        """
        difference_inch = distance_film - distance_dicom

        if difference_inch > 0:
            return round(difference_inch * dpi_film), 0
        else:
            return 0, floor(abs(difference_inch) * dpi_dicom)

    # Calculate points to remove for each side
    # The distance is computed from one border to the center of the reference point (row, column) +- 0.5
    points_to_remove_on_film_up, points_to_remove_on_dicom_up = calculate_points_to_remove(
        (row_film + 0.5) / img_film.dpi,
        (row_dicom + 0.5) / img_dicom.dpi,
        img_film.dpi, img_dicom.dpi
    )
    points_to_remove_on_film_down, points_to_remove_on_dicom_down = calculate_points_to_remove(
        (img_film.shape[0] - row_film - 0.5) / img_film.dpi,
        (img_dicom.shape[0] - row_dicom - 0.5) / img_dicom.dpi,
        img_film.dpi, img_dicom.dpi
    )
    points_to_remove_on_film_left, points_to_remove_on_dicom_left = calculate_points_to_remove(
        (column_film + 0.5)/ img_film.dpi,
        (column_dicom + 0.5) / img_dicom.dpi,
        img_film.dpi, img_dicom.dpi
    )
    points_to_remove_on_film_right, points_to_remove_on_dicom_right = calculate_points_to_remove(
        (img_film.shape[1] - column_film - 0.5) / img_film.dpi,
        (img_dicom.shape[1] - column_dicom - 0.5) / img_dicom.dpi,
        img_film.dpi, img_dicom.dpi
    )

    # Crop the images
    img_film_result = ArrayImage(img_film.array[
        points_to_remove_on_film_up : img_film.shape[0] - points_to_remove_on_film_down,
        points_to_remove_on_film_left : img_film.shape[1] - points_to_remove_on_film_right
    ], dpi=img_film.dpi)
    img_dicom_result = ArrayImage(img_dicom.array[
        points_to_remove_on_dicom_up : img_dicom.shape[0] - points_to_remove_on_dicom_down,
        points_to_remove_on_dicom_left : img_dicom.shape[1] - points_to_remove_on_dicom_right
    ], dpi=img_dicom.dpi)

    return img_film_result, img_dicom_result