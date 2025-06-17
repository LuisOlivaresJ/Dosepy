"""This module contains tools to calculate Biological Equivalent Dose"""

from pathlib import Path
import SimpleITK as sitk


def load(path_to_file: str | Path):
    """
    Load a dose distribution from a DICOM file.

    Parameters
    ----------
    path_to_file : str or Path
        The path to the DICOM file containing the dose distribution.

    Returns
    -------
    sitk.Image
        A SimpleITK image object representing the dose distribution.
        
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

    # Check if the file is DICOM file
    with open(path_to_file, "rb") as my_file:
        my_file.read(128)  # Skip first bytes

        if my_file.read(4) != b'DICM':
            print(f"{path_to_file} is not a valid dcm file.")
            raise ValueError(f"{path_to_file} is not a valid dcm file.")
        
        
    # Load the DICOM file using SimpleITK
    img = sitk.ReadImage(str(path_to_file), outputPixelType=sitk.sitkFloat64)

    # Check if the tag '3004|000e' exists in the metadata
    if not img.HasMetaDataKey('3004|000e'):
        raise ValueError(f"The DICOM file {path_to_file} does not contain the required metadata tag DoseGridScaling (3004|000e).")

    # Convert image to a dose distribution
    dose = img * float(img.GetMetaData('3004|000e'))

    return dose