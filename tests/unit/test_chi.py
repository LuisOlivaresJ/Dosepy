import pytest
import numpy as np

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