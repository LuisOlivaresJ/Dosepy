import pytest
import numpy as np

from Dosepy.tools.equate import crop_using_ref_position
from Dosepy.image import ArrayImage

# Testing equete module
# Tesgin crop_using_ref_position
def test_tps_largest_without_rounding_points():
    film = ArrayImage(np.ones((9, 9)), dpi = 9)
    dicom = ArrayImage(np.zeros((10, 10)), dpi = 3)
    dicom.array[4:7, 4:7] = 1

    film_result, dicom_result = crop_using_ref_position(
        img_film = film,
        img_dicom = dicom,
        row_film = 4,
        column_film = 4,
        row_dicom = 5,
        column_dicom = 5,
    )

    assert dicom_result.shape == (3, 3)
    assert np.mean(dicom_result.array) == 1


def test_dicom_largest_with_rounding_points():
    """
    One extra point in dicom needed to cover one small point in film 
    """
    film = ArrayImage(np.ones((10, 10)), dpi=9)
    dicom = ArrayImage(np.zeros((10, 10)), dpi=3)
    dicom.array[4:8, 4:8] = 1

    film_result, dicom_result = crop_using_ref_position(
        img_film = film,
        img_dicom = dicom,
        row_film=4,
        column_film=4,
        row_dicom=5,
        column_dicom=5,
    )

    assert dicom_result.shape == (4, 4)
    assert np.mean(dicom_result.array) == 1


def test_dicom_largest_with_rounding_points_2():
    """
    Several points in film to almost fill one point in dicom
    """
    film = ArrayImage(np.ones((8, 8)), dpi=9)
    dicom = ArrayImage(np.zeros((10, 10)), dpi=3)
    dicom.array[4:8, 4:8] = 1

    film_result, dicom_result = crop_using_ref_position(
        img_film = film,
        img_dicom = dicom,
        row_film=4,
        column_film=4,
        row_dicom=5,
        column_dicom=5,
    )

    assert dicom_result.shape == (3, 3)
    assert np.mean(dicom_result.array) == 1


def test_mixing_large():
    
    film = ArrayImage(np.ones((11, 6)), dpi = 6)
    dicom = ArrayImage(np.zeros((6, 5)), dpi = 4)
    dicom.array[:, 0: 4] = 1

    film_result, dicom_result = crop_using_ref_position(
        img_film=film,
        img_dicom=dicom,
        row_film = 5,
        column_film = 3,
        row_dicom = 3,
        column_dicom = 1,
    )

    assert film_result.shape == (9, 5)
    assert dicom_result.shape == (6, 4)
    assert np.mean(dicom_result.array) == 1
    

def test_mixing():

    film = ArrayImage(np.ones((9, 4)), dpi=6)
    dicom = ArrayImage(np.ones((5, 3)), dpi=4)
    #dicom.array[]

    film_result, dicom_result = crop_using_ref_position(
        img_film = film,
        img_dicom = dicom,
        row_film = 5,
        column_film = 2,
        row_dicom = 3,
        column_dicom = 1,
    )

    assert film_result.shape == (8, 4)
    assert dicom_result.shape == (5, 3)
    assert np.mean(dicom_result.array) == 1