"""This module contains tools to calculate Biological Equivalent Dose"""

from pathlib import Path
import SimpleITK as sitk


def load_dose(path_to_file: str | Path):
    """
    Load a dose distribution from a DICOM file.

    Parameters
    ----------
    path_to_file : str or Path
        The path to the DICOM file containing the dose distribution.

    Returns
    -------
    sitk.Image
        A SimpleITK image representing the dose distribution.
        
    """

    # Check if the input is a string or Path object
    if not isinstance(path_to_file, (str, Path)):
        raise TypeError("path_to_file must be a string or a Path object.")

    # Convert str to Path if necessary
    if isinstance(path_to_file, str):
        path_to_file = Path(path_to_file)

    # Check if the file exists
    if not path_to_file.is_file():
        raise FileNotFoundError(f"The file {path_to_file} does not exist.")

    # Check if the file is a DICOM file
    with open(path_to_file, "rb") as my_file:
        my_file.read(128)  # Skip first 128 bytes

        if my_file.read(4) != b'DICM':
            print(f"{path_to_file} is not a valid dcm file.")
            raise ValueError(f"{path_to_file} is not a valid dcm file.")
        
        
    # Load the DICOM file using SimpleITK
    img = sitk.ReadImage(str(path_to_file), outputPixelType=sitk.sitkFloat64)

    # Check if the tag '3004|000e' (DoseGridScaling) exists in the metadata
    if not img.HasMetaDataKey('3004|000e'):
        raise ValueError(f"The DICOM file {path_to_file} does not contain the required metadata tag DoseGridScaling (3004|000e).")

    # Convert image to a dose distribution
    dose = img * float(img.GetMetaData('3004|000e'))

    return dose


def eqd2(dose: sitk.Image, alpha_beta: float, number_fractions: int) -> sitk.Image:
    """
    Equivalent dose in 2 Gy per fraction calculation for every boxel.

    Parameters
    ----------
    dose : SimpleITK.Image
        Dose distribution
    alpha_beta : float
        Alpha / beta ratio
    number_fractions : int
        Number of fractions

    Returns
    -------
    SimpleITK.Image
        Dose distribution in EQD2Gy

    Note
    ----
    The equivalent dose is calculated using the formula:

    EQD2 = D * (d + alpha_beta) / (2 + alpha_beta), 
    where D is the total dose, d is the dose per fraction (D/number_fractions) 
    and alpha_beta is the alpha/beta ratio.

    EQD2 represents an equivalent radiation dose (EQD2) that would have the same biological 
    effect as a standard fractionation schedule of 2 Gy per fraction.
    """

    # Check if dose is a valid sitk.Image object
    if not isinstance(dose, sitk.Image):
        raise ValueError(f"Invalid parameter for dose. It has to be a SimpleITK.Image")
    # Check if alpha_beta is a number
    if not isinstance(alpha_beta, (int, float)):
        raise ValueError(f"Invalid parameter for alpha_beta. It has to be a number")
    # Check if alpha_beta is a valid number
    if not 1 <= alpha_beta <= 15:
        raise ValueError(f"Invalid aplha_beta parameter. It has to be between 1 and 15")
    # Check if number_fractions is a valid number
    if not isinstance(number_fractions, (int, float)):
        raise ValueError(f"Invalid parameter for number_fractions. It has to be a number")
    # Check if number_fractions is a valid integer
    if number_fractions % 1 != 0:
        raise ValueError(f"Invalid parameter for number_fractions. It has to be an integer")

    # Image as numpy array
    dose_array = sitk.GetArrayFromImage(dose)

    # EQD2 calculation
    dose_eqd2_array = dose_array * (dose_array/number_fractions + alpha_beta) / (2 + alpha_beta)
    
    # Back to SimpleITK
    dose_eqd2 = sitk.GetImageFromArray(dose_eqd2_array)
    dose_eqd2.CopyInformation(dose)

    return dose_eqd2


