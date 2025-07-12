"""This module contains tools to calculate Biological Equivalent Dose"""

from pathlib import Path
import SimpleITK as sitk
from matplotlib.pylab import f
import pydicom


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


def get_structures(path_to_file: str | Path) -> list[str]:
    """
    Get the structure names from a DICOM RTSTRUCT file.

    Parameters
    ----------
    path_to_file : str or Path
        The path to the DICOM RTSTRUCT file.

    Returns
    -------
    dict[str, int]
        A dictionary with structure names as keys and their corresponding index.
    """
    
    # Check if the input is a string or Path object
    if not isinstance(path_to_file, (str, Path)):
        raise TypeError("path_to_file must be a string or a Path object.")

    # Check if the given path is a file and exists
    if isinstance(path_to_file, str):
        path_to_file = Path(path_to_file)
    if not path_to_file.is_file():
        raise FileNotFoundError(f"The file {path_to_file} does not exist.")

    # Check if the file is a DICOM file
    with open(path_to_file, "rb") as my_file:
        my_file.read(128) # Skip first 128 bytes
        if my_file.read(4) != b'DICM':
            raise ValueError(f"{path_to_file} is not a valid DICOM file.")
        
    # Read the DICOM file
    ds = pydicom.dcmread(path_to_file)

    # Check if the DICOM file represents structures, using SOP class name RT Structure Set Storage
    # https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_I.4.html
    if not ds.get("SOPClassUID") == '1.2.840.10008.5.1.4.1.1.481.3':
        raise ValueError(f"{path_to_file} is not a valid DICOM RTSTRUCT file.")
    
    # Get the structure names
    structures = [s.ROIName for s in ds.StructureSetROISequence]
    # Get the structure numbers
    structure_numbers = [int(s.ROINumber) for s in ds.StructureSetROISequence]
    # Check if there are any structures
    if not structures:
        raise ValueError(f"No structures found in the DICOM RTSTRUCT file {path_to_file}.")
    
    # Create a dictionary with structure names and their corresponding index
    structures = {name: number for name, number in zip(structures, structure_numbers)}
    
    return structures