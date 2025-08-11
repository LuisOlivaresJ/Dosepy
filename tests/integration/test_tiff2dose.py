import pytest
import numpy as np
import os
import gdown

from Dosepy.calibration import LUT
from Dosepy.image import load, load_multiples
from Dosepy.tiff2dose import Tiff2DoseM


# Calibration files with optical filters.
# Doses [0, 0.5, 1, 2, 4, 6, 8, 10]
# The files are downloaded from Google Drive

TIFF_FILES = {
    "tests/unit/fixtures/tiff_files/CAL20241106_001.tif": "104lJjZ4KWi4Y7KyIWb8CUaIbun2BNBBN",
    "tests/unit/fixtures/tiff_files/CAL20241106_002.tif": "19vHQk_Bf3V8VKQJuZfU6Y5hMv_U8PZRI",
    "tests/unit/fixtures/tiff_files/CAL20241106_003.tif": "1-zsLpqKF8j3a7YjgqUdxErE2Bq-MncYW"
}

os.makedirs("tests/unit/fixtures/tiff_files/", exist_ok=True)
for filename, file_id in TIFF_FILES.items():
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} aldready exists.")

files = [file for file in TIFF_FILES.keys()]

VER_FILES = {
    "tests/unit/fixtures/tiff_files/Ver_050dpi20241106_001.tif": "1U9RVokrM7U0swCATbKo9GiQjLyXJLh1o",
    "tests/unit/fixtures/tiff_files/Ver_050dpi20241106_002.tif": "1dRC4DTc__HuspOrTL2vyHxv59f23j-R3",
    "tests/unit/fixtures/tiff_files/Ver_050dpi20241106_003.tif": "1TsGLavGMBAMBfB_vgoY-shj_ewpCEr2s",
}

os.makedirs("tests/unit/fixtures/tiff_files/", exist_ok=True)
for filename, file_id in VER_FILES.items():
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} aldready exists.")

ver_files = [file for file in VER_FILES.keys()]

@pytest.fixture
def verif_img():

    return load_multiples(ver_files)


@pytest.fixture
def cal_center():
    cal_image = load_multiples(files)
    cal = LUT(cal_image)
    cal.set_central_rois(size=(16, 8))
    cal.set_doses([0, 0.5, 1, 2, 4, 6, 8, 10])
    #cal.plot_rois()
    cal.compute_central_lut(filter = 5)

    return cal

@pytest.fixture
def cal_lateral_without_profile():
    cal_image = load_multiples(files)
    cal = LUT(cal_image)
    cal.set_central_rois(size=(16, 8))
    cal.set_doses([0, 0.5, 1, 2, 4, 6, 8, 10])
    #cal.plot_rois()
    cal.compute_lateral_lut(filter = 5)

    return cal


beam_file = "tests/unit/fixtures/BeamProfile.csv"
beam_file_id = "1kRXQwvN50_8b9UVAAPmpqQep0_g8ML7M"

os.makedirs("tests/unit/fixtures/", exist_ok=True)
if not os.path.exists(beam_file):
    url = f"https://drive.google.com/uc?id={beam_file_id}"
    print(f"Downloading {beam_file}...")
    gdown.download(url, beam_file, quiet=False)
else:
    print(f"{beam_file} aldready exists.")


@pytest.fixture
def cal_lateral_with_profile():
    cal_image = load_multiples(files)
    cal = LUT(cal_image)
    cal.set_central_rois(size = (180, 8))
    cal.set_doses([0, 0.5, 1, 2, 4, 6, 8, 10])
    #cal.plot_rois()
    cal.set_beam_profile(beam_file)
    cal.compute_lateral_lut(filter = 5)

    return cal


##################################################################
##################################################################
# POLYNOMIAL


# Test red polynomial calibration with rois at center
def test_RP_cal_rois_center(verif_img, cal_center):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="RP",
        lut=cal_center,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test green polynomial, calibration with rois at center
def test_GP_cal_rois_center(verif_img, cal_center):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="GP",
        lut=cal_center,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test blue polynomial, calibration with rois at center
def test_BP_cal_rois_center(verif_img, cal_center):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="BP",
        lut=cal_center,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.1)

# Test red polynomial, lateral calibration without relative dose profile
def test_RP_cal_lateral_without_profile(
        verif_img,
        cal_lateral_without_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="RP",
        lut=cal_lateral_without_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test green polynomial, lateral calibration without relative dose profile
def test_GP_cal_lateral_without_profile(
        verif_img,
        cal_lateral_without_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="GP",
        lut=cal_lateral_without_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test green polynomial, lateral calibration without relative dose profile
def test_BP_cal_lateral_without_profile(
        verif_img,
        cal_lateral_without_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="BP",
        lut=cal_lateral_without_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.1)


# Test red polynomial, lateral calibration with relative dose profile
def test_RP_cal_lateral_without_profile(
        verif_img,
        cal_lateral_with_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="RP",
        lut=cal_lateral_with_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test green polynomial, lateral calibration with relative dose profile
def test_RP_cal_lateral_without_profile(
        verif_img,
        cal_lateral_with_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="GP",
        lut=cal_lateral_with_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test blue polynomial, lateral calibration with relative dose profile
def test_RP_cal_lateral_without_profile(
        verif_img,
        cal_lateral_with_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="BP",
        lut=cal_lateral_with_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.1)


##################################################################
##################################################################
# RATIONAL

# Test red rational calibration with rois at center
def test_RR_cal_rois_center(verif_img, cal_center):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="RR",
        lut=cal_center,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test green rational, calibration with rois at center
def test_GR_cal_rois_center(verif_img, cal_center):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="GR",
        lut=cal_center,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test blue rational, calibration with rois at center
def test_BR_cal_rois_center(verif_img, cal_center):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="BR",
        lut=cal_center,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.1)

# Test red rational, lateral calibration without relative dose profile
def test_RR_cal_lateral_without_profile(
        verif_img,
        cal_lateral_without_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="RR",
        lut=cal_lateral_without_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test green rational, lateral calibration without relative dose profile
def test_GR_cal_lateral_without_profile(
        verif_img,
        cal_lateral_without_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="GR",
        lut=cal_lateral_without_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test green rational, lateral calibration without relative dose profile
def test_BR_cal_lateral_without_profile(
        verif_img,
        cal_lateral_without_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="BR",
        lut=cal_lateral_without_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.1)


# Test red rational, lateral calibration with relative dose profile
def test_RR_cal_lateral_without_profile(
        verif_img,
        cal_lateral_with_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="RR",
        lut=cal_lateral_with_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test green rational, lateral calibration with relative dose profile
def test_GR_cal_lateral_without_profile(
        verif_img,
        cal_lateral_with_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="GR",
        lut=cal_lateral_with_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.05)

# Test blue rational, lateral calibration with relative dose profile
def test_BR_cal_lateral_without_profile(
        verif_img,
        cal_lateral_with_profile,
        ):
    t2d = Tiff2DoseM()
    dose = t2d.get_dose(
        img=verif_img,
        format="BR",
        lut=cal_lateral_with_profile,
    )
    dose_at_center = dose.array[110:140, 450:480]
    dose_mean = np.mean(dose_at_center)

    assert dose_mean == pytest.approx(5, rel=0.1)