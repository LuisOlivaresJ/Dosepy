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
    Function used to crop two images with different physical size and spatial resolution.
    The algorithm uses a reference point on both images to crop the largest. 
    If dicom is larger, crop only lowest integer values.
    """

    def calculate_points_to_remove(distance_film, distance_dicom, dpi_film, dpi_dicom):
        difference_inch = distance_film - distance_dicom

        if difference_inch > 0:
            return round(difference_inch * dpi_film), 0
        else:
            return 0, floor(abs(difference_inch) * dpi_dicom)

    # Calculate points to remove for each side
    # The distance is computed from one border to the center of the point (row, column)
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