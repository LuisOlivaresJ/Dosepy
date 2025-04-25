# Used to test the tiff2does.py module

import pytest
from pathlib import Path
import numpy as np
from numpy import ndarray
#from Dosepy.tiff2dose import Tiff2Dose
from Dosepy.calibration import LUT
from Dosepy.image import TiffImage, DoseImage, load
from Dosepy.tiff2dose import RedPolynomialDoseConverter

cwd = Path(__file__).parent

@pytest.fixture
def calibration_lateral_lut_without_filters():
    file_path = cwd / "fixtures/CAL/film20240620_002.tif"
    beam_profile_path = cwd / "fixtures/CAL/BeamProfile.csv"

    cal_img = load(file_path)
    #profile = np.genfromtxt(beam_profile_path, delimiter=",")

    cal = LUT(cal_img)
    cal.set_central_rois((180, 8), show=False)
    cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
    cal.set_beam_profile(str(beam_profile_path))
    cal.compute_lateral_lut()

    return cal


@pytest.fixture
def calibration_central_lut_without_filters():
    file_path = cwd / "fixtures/CAL/film20240620_002.tif"
    cal_img = load(file_path)
    cal = LUT(cal_img)
    cal.set_central_rois((8, 8))
    cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
    cal.compute_central_lut()

    return cal

@pytest.fixture
def calibration_lut_with_filters():
    file_path = cwd / "fixtures/CAL20241106_001.tif"
    beam_profile_path = cwd / "fixtures/CAL/BeamProfile.csv"

    cal_img = load(file_path)

    cal = LUT(cal_img)
    cal.set_central_rois((180, 8), show=False)
    cal.set_doses([0, 0.5, 1, 2, 4, 6, 8, 10])
    cal.set_beam_profile(str(beam_profile_path))
    cal.compute_lateral_lut()

    return cal


@pytest.fixture
def qa_tiff_image():
    file_path = cwd / "fixtures/TO_DOSE/Film_a_20240711_002.tif"
    return load(file_path)

@pytest.fixture
def verif_img():
    file_path = cwd / "fixtures/CAL/film20240620_002.tif"
    return load(file_path)

@pytest.fixture
def verif_img_with_filters():
    file_path = cwd / "fixtures/Ver_050dpi20241106_001.tif"
    img = load(file_path)
    img.set_labeled_films_and_filters()
    return img


"""
# Test instance of the Tiff2Dose class
def test_tiff2dose_instance(verif_img, calibration_lut):
    t2d = Tiff2Dose(verif_img, calibration_lut)
    assert isinstance(t2d, Tiff2Dose)


# Test the red method of the Tiff2Dose class
def test_red_retuns_a_DoseImage_instance(verif_img, calibration_lut):
    t2d = Tiff2Dose(verif_img, calibration_lut)
    dred = t2d.red("rational")

    assert isinstance(dred, ndarray)
"""

# Test the Tiff2DoseFactory class

# Test the register and get_dose_converter methods
def test_register_method():
    from Dosepy.tiff2dose import Tiff2DoseFactory
    from Dosepy.tiff2dose import RedPolynomialDoseConverter

    t2df = Tiff2DoseFactory()
    t2df.register_method("RP", RedPolynomialDoseConverter)

    assert isinstance(t2df.get_dose_converter("RP"), RedPolynomialDoseConverter)


# Test DoseConverter using RedPolynomialDoseConverter inhereted class

# Test _set_lateral_positions method
def test_set_lateral_positions(verif_img_with_filters):

    img = verif_img_with_filters

    rpd = RedPolynomialDoseConverter()
    rpd._set_lateral_positions(img)

    assert rpd.pixel_positions_mm[0] < -150
    assert rpd.pixel_positions_mm[-1] > 150

# Test _get_zero_dose_intensity method
def test_get_zero_dose_intensity(verif_img_with_filters):

    rpd = RedPolynomialDoseConverter()

    intensity, _ = rpd._get_zero_dose_intensity(
        verif_img_with_filters,
        "red",
        at_zero_position = True
        )

    assert intensity == pytest.approx(40050, abs = 100)

# Test _get_lateral_intensities_for_zero_dose method
def test_get_lateral_intensities_for_zero_dose(
        verif_img_with_filters,
        calibration_lut_with_filters
        ):

    rpd = RedPolynomialDoseConverter()

    lateral_intensities, positions = rpd._get_lateral_intensities_for_zero_dose(
        verif_img_with_filters,
        calibration_lut_with_filters,
        "red"
        )

    assert len(lateral_intensities) == 177
    assert len(positions) == 177

# Test convert2dose method
def test_convert2dose_using_RedPolynomialDoseConverter(
        verif_img_with_filters,
        calibration_lut_with_filters
    ):
    rpdc = RedPolynomialDoseConverter()
    dose = rpdc.convert2dose(
        verif_img_with_filters,
        calibration_lut_with_filters
    )
    # Dose tolerance of 5%
    assert np.mean(dose.array[100:125, 400:500]) == pytest.approx(5, abs=0.25)


# Test convert2dose method without lateral correction
def test_convert2dose_without_lateral_correction(
    verif_img,
    calibration_central_lut_without_filters
    ):
    rpdc = RedPolynomialDoseConverter()
    dose = rpdc.convert2dose(
        verif_img,
        calibration_central_lut_without_filters
    )

    assert np.mean(dose.array[370:380, 430:460]) == pytest.approx(4, rel=0.05)