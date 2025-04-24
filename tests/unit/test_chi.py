import os

import pytest
import numpy as np
from pathlib import Path

from Dosepy import image
from Dosepy.tools.gamma import chi


# Test if dose_TA is below dose difference, chi pass rate throws 0
def test_small_dose_ta():
    
    # All points differ by 4 %
    a = np.zeros((10, 10)) + 96  
    b = np.zeros((10, 10)) + 100
    D_ref = image.load(a, dpi = 25.4)
    D_eval = image.load(b, dpi = 25.4)

    # Execute the code being tested
    chi_analysis = chi(
        reference_image = D_ref,
        comparison_image = D_eval,
        doseTA = 3,
        distTA = 1,
    )

    # With a dose to agreement 3%, pass rate = 0.0
    assert chi_analysis[1] == pytest.approx(0, 0.1)

# Test if dose_TA is above dose_difference, chi pass rate throws 100
def test_big_dose_ta():
    
    # All points differ by 2 %
    a = np.zeros((10, 10)) + 98  
    b = np.zeros((10, 10)) + 100
    D_ref = image.load(a, dpi = 25.4)
    D_eval = image.load(b, dpi = 25.4)

    # Execute the code being tested
    chi_analysis = chi(
        reference_image = D_ref,
        comparison_image = D_eval,
        doseTA = 3,
        distTA = 1,
    )

    # With a dose to agreement 3%, pass rate = 100.0
    assert chi_analysis[1] == pytest.approx(100, 0.1)


# Test using spine (logo) dose distribution
def test_using_spine_dose_distribution():

    dir_path = Path(os.getcwd())
    film_path = dir_path / "tests" / "unit" / "fixtures" / "GAMMA" / "film_dose_map_Dosepy.csv"
    dcm_path = dir_path / "tests"/ "unit" / "fixtures" / "GAMMA" / "dcm_dose_map_Dosepy.csv"

    film = image.ArrayImage(
        np.genfromtxt(film_path, delimiter=","),
        dpi=25.4
        )
    dcm = image.ArrayImage(
        np.genfromtxt(dcm_path, delimiter=","),
        dpi=25.4
        )
    
    _, chi_rate = chi(
        reference_image=dcm,
        comparison_image=film,
        doseTA=3,
        distTA=2,
        threshold=10,
    )

    assert chi_rate == pytest.approx(expected=98.5, abs=1)


# Test using lattice dose distribution
def test_using_lattice():

    dir_path = Path(os.getcwd())
    film_path = dir_path / "tests" / "unit" / "fixtures" / "GAMMA" / "film_dose_map_isle.csv"
    dcm_path = dir_path / "tests"/ "unit" / "fixtures" / "GAMMA" / "dcm_dose_map_isle.csv"

    film = image.ArrayImage(
        np.genfromtxt(film_path, delimiter=","),
        dpi=75
        )
    dcm = image.ArrayImage(
        np.genfromtxt(dcm_path, delimiter=","),
        dpi=25.4
        )
    
    film.reduce_resolution_as(dcm)

    _, chi_rate = chi(
        reference_image=dcm,
        comparison_image=film,
        doseTA=3,
        distTA=2,
        threshold=10,
    )

    assert chi_rate == pytest.approx(expected=88.5, abs=1)


# Test threshold parameter
def test_threshold():
    dir_path = Path(os.getcwd())
    film_path = dir_path / "tests" / "unit" / "fixtures" / "GAMMA" / "film_dose_map_isle.csv"
    dcm_path = dir_path / "tests"/ "unit" / "fixtures" / "GAMMA" / "dcm_dose_map_isle.csv"

    film = image.ArrayImage(
        np.genfromtxt(film_path, delimiter=","),
        dpi=75
        )
    dcm = image.ArrayImage(
        np.genfromtxt(dcm_path, delimiter=","),
        dpi=25.4
        )
    
    film.reduce_resolution_as(dcm)

    _, chi_rate = chi(
        reference_image=dcm,
        comparison_image=film,
        doseTA=3,
        distTA=2,
        threshold=30,
    )

    assert chi_rate == pytest.approx(expected=93.0, abs=1)