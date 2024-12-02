# Used to test the tiff2does.py module

import pytest
from pathlib import Path
import numpy as np
from Dosepy.tiff2dose import Tiff2Dose
from Dosepy.calibration import LUT
from Dosepy.image import TiffImage, DoseImage, load

cwd = Path(__file__).parent

@pytest.fixture
def calibration_lut():
    file_path = cwd / "fixtures/CAL/film20240620_002.tif"
    beam_profile_path = cwd / "fixtures/CAL/BeamProfile.csv"

    cal_img = load(file_path)
    profile = np.genfromtxt(beam_profile_path, delimiter=",")

    cal = LUT(cal_img)
    cal.set_central_rois((180, 8), show=False)
    cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
    cal.compute_central_lut()

    return cal

@pytest.fixture
def qa_tiff_image():
    file_path = cwd / "fixtures/TO_DOSE/Film_a_20240711_002.tif"
    return load(file_path)

@pytest.fixture
def verif_img():
    file_path = cwd / "fixtures/CAL/film20240620_002.tif"
    return load(file_path)

# Test instance of the Tiff2Dose class
def test_tiff2dose_instance(verif_img, calibration_lut):
    t2d = Tiff2Dose(verif_img, calibration_lut)
    assert isinstance(t2d, Tiff2Dose)

# Test the red method of the Tiff2Dose class
def test_red_retuns_a_DoseImage_instance(verif_img, calibration_lut):
    t2d = Tiff2Dose(verif_img, calibration_lut)
    dred = t2d.red("rational")

    assert isinstance(dred, DoseImage)