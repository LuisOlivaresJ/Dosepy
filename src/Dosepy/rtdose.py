"""This module contains tools to calculate Biological Equivalent Dose"""

from pathlib import Path
from turtle import title

import numpy as np
import SimpleITK as sitk
import pydicom
from skimage.draw import polygon
import plotly.graph_objects as go


def load_dose(path_to_file: str | Path) -> sitk.Image:
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

    _is_dicom(path_to_file)
        
    # Load the DICOM file using SimpleITK
    img = sitk.ReadImage(str(path_to_file), outputPixelType=sitk.sitkFloat64)

    # Check if the tag '3004|000e' (DoseGridScaling) exists in the metadata
    if not img.HasMetaDataKey('3004|000e'):
        raise ValueError(f"The DICOM file {path_to_file} does not contain the required metadata tag DoseGridScaling (3004|000e).")

    # Convert image to a dose distribution
    dose = img * float(img.GetMetaData('3004|000e'))

    return dose


def eqd2(
    dose: sitk.Image,
    alpha_beta: float,
    number_fractions: int) -> sitk.Image:
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


def get_structures_names(path_to_file: str | Path) -> dict:
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


def get_structure_coordinates(
    structure: str,
    path_to_file: str) -> list[np.ndarray]:
    """
    Get the coordinates of a structure from a DICOM RTSTRUCT file.
    Parameters
    ----------
    structure : str
        The name of the structure to get the coordinates for.
    path_to_file : str
        The path to the DICOM RTSTRUCT file.

    Returns
    -------
    list[np.ndarray]
        A list of numpy arrays of shape (N, 3). Each array represents a slice, 
        and contains the coordinates of the structure as (x, y, z = constant).
    
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
        my_file.read(128)  # Skip first 128 bytes
        if my_file.read(4) != b'DICM':
            raise ValueError(f"{path_to_file} is not a valid DICOM file.")
    # Check if the parameter structure is a string
    if not isinstance(structure, str):
        raise TypeError("structure must be a string.")
    
    # Read the DICOM file
    ds = pydicom.dcmread(path_to_file)

    # Check if the DICOM file represents structures, using SOP class name RT Structure Set Storage
    if not ds.get("SOPClassUID") == '1.2.840.10008.5.1.4.1.1.481.3':
        raise ValueError(f"{path_to_file} is not a valid DICOM RTSTRUCT file.")

    # Create a dictionary with structure names and their corresponding index
    structures_names = [s.ROIName for s in ds.StructureSetROISequence]
    structure_numbers = [int(s.ROINumber) for s in ds.StructureSetROISequence]

    
    structures = {name: number for name, number in zip(structures_names, structure_numbers)}

    # Check if the structure is in the list of structures
    if structure not in structures:
        raise ValueError(f"The structure {structure} is not in the DICOM RTSTRUCT file {path_to_file}.")

    # Get index of the structure
    structure_index = structures.get(structure) - 1

    # Get the structure coordinates
    coordinates = []
    for slice in ds.ROIContourSequence[structure_index].ContourSequence:
        points = slice.NumberOfContourPoints
        slice_coordinates = np.array(slice.ContourData).reshape(points, 3)
        coordinates.append(slice_coordinates)
    
    return coordinates


def get_dose_plane_by_coordinate(
    dose: sitk.Image,
    z_coordinate: float) -> sitk.Image:
    """
    Get a 2D dose plane at a specific z-coordinate from a 3D dose distribution.

    Parameters
    ----------
    dose : sitk.Image
        The 3D dose distribution.
    z_coordinate : float
        The z-coordinate at which to extract the dose plane.

    Returns
    -------
    sitk.Image
        A 2D dose plane at the specified z-coordinate.

    Raises
    ------
    ValueError
        If the z_coordinate is not a number or is not within the range of the dose distribution.

    Notes
    -----
    The z-coordinate should be within the range of the dose distribution's z-coordinates.
    The algorithm uses linear interpolation to extract the dose plane.
    """

    # Check if z_coordinate is a number
    if not isinstance(z_coordinate, (int, float)):
        raise ValueError(f"Invalid parameter for z_coordinate. It has to be a number")
    
    # Check if z_coordinate is between coordinates of the dose distribution
    z_min = dose.GetOrigin()[2]
    z_max = dose.GetOrigin()[2] + dose.GetDepth() * dose.GetSpacing()[2]

    if not z_min <= z_coordinate <= z_max:
        raise ValueError(
            f"The z_coordinate {z_coordinate} is not between the coordinates of the dose distribution "
            f"({z_min}, {z_max})."
        )

    # Reference image with the required location
    reference_image = sitk.Image(
        (dose.GetSize()[0], dose.GetSize()[1], 1),
        dose.GetPixelIDValue()
    )

    reference_image.SetOrigin((dose.GetOrigin()[0], dose.GetOrigin()[1], z_coordinate))
    reference_image.SetDirection(dose.GetDirection())
    reference_image.SetSpacing(dose.GetSpacing())

    interpolated_dose = sitk.Resample(
        dose,
        reference_image,  
        sitk.Transform(3, sitk.sitkIdentity),  # Do not apply any transformation
        sitk.sitkLinear,  # Uses linear interpolation
    )
    
    return interpolated_dose[:, :, 0]


def get_2D_mask_by_coordinates_and_image_shape(
    coordinates: np.ndarray,
    image: sitk.Image) -> np.ndarray:
    """
    Get a mask from the coordinates of a structure.
    Parameters
    ----------
    coordinates : np.ndarray
        The coordinates of the structure, with shape (N, 3),
        where N is the number of points and each point is represented by (x, y, z).
    image : sitk.Image
        An image used as reference to create the mask shape. Only the first
        two dimensions are used (x, y).
    Returns
    -------
    np.ndarray
        A 2D mask of the structure. The shape is the same as the given image.

    """

    # Check that coordinates is a numpy array
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("coordinates must be a numpy array.")
    
    # Check that image is a SimpleITK image
    if not isinstance(image, sitk.Image):
        raise TypeError("image must be a SimpleITK.Image object.")

    # Check that the array of coordinates has the right shape
    if not coordinates.shape[1] == 3:
        raise ValueError("coordinates must be a 2D array with shape (N, 3), where N is the number of points.")

    # Row and columns of the polygon's vertices
    r = coordinates[:, 1]  # (x, y, z) Left-Rigth, Up-Down, Feet-Head
    c = coordinates[:, 0]

    # Vertices as index
    r_index = (r - image.GetOrigin()[1]) / image.GetSpacing()[1]
    c_index = (c - image.GetOrigin()[0]) / image.GetSpacing()[0]

    # Size of the mask
    shape = (image.GetSize()[1], image.GetSize()[0])

    # Get the indices of the pixels inside the polygon
    rr, cc = polygon(r_index, c_index, shape=shape)

    # Create a boolean mask with the same shape as the dose distribution
    mask = np.zeros(shape, dtype=np.bool)
    mask[rr, cc] = 1

    return mask


def _get_dose_in_structure_by_plane(dose_2D: sitk.Image, coordinates: np.ndarray) -> list[float]:
    # Check coordinates is a numpy array
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("coordinates must be a numpy array.")
    # Check that the array of coordinates has the right shape
    if not coordinates.shape[1] == 3:
        raise ValueError("coordinates must be a 2D array with shape (N, 3), where N is the number of points.")
    # Check that dose_2D is a SimpleITK image
    if not isinstance(dose_2D, sitk.Image):
        raise TypeError("dose_2D must be a SimpleITK.Image object.")
    
    mask = get_2D_mask_by_coordinates_and_image_shape(coordinates, dose_2D)
    dvh = sitk.GetArrayFromImage(dose_2D)[mask]
    
    return dvh


def get_dose_in_structure(
    path_to_dose_file: str,
    path_to_structures_file: str,
    structure: str) -> np.ndarray:
    """
    Get the dose volume histogram (DVH) for a specific structure from a DICOM RTSTRUCT file.

    Parameters
    ----------
    path_to_dose_file : str
        The path to the DICOM file containing the dose distribution.
    path_to_structures_file : str
        The path to the DICOM RTSTRUCT file containing the structure information.
    structure : str
        The name of the structure for which to get the DVH.
    
    Returns
    -------
    np.ndarray
        A numpy array containing the DVH for the specified structure.
    
    Raises
    ------
    ValueError
        If the structure is not found in the DICOM RTSTRUCT file.
    TypeError
        If the path_to_dose_file or path_to_structures_file is not a string or Path object.
    """
    _is_dicom(path_to_dose_file)
    _is_dicom(path_to_structures_file)
    # Check if structure is a string
    if not isinstance(structure, str):
        raise TypeError("structure must be a string.")

    structures = get_structures_names(path_to_structures_file)
    if not structure in structures:
        raise ValueError(f"The structure {structure} is not in the DICOM RTSTRUCT file.")
    
    structure_coordinates = get_structure_coordinates(structure, path_to_structures_file)

    dose_distribution = load_dose(path_to_dose_file)

    dose_in_structure = []
    for slice_coordinates in structure_coordinates:
        # Check that each slice_coordinates has a constant z coordinate
        if not len(set(slice_coordinates[:, 2])) == 1:
            raise ValueError("Each slice_coordinates must have a constant z coordinate.")
        
        z_coordinate = slice_coordinates[0, 2]

        dose_2D = get_dose_plane_by_coordinate(dose_distribution, z_coordinate)
        dose_in_structure_in_slice = _get_dose_in_structure_by_plane(dose_2D, slice_coordinates)
        dose_in_structure.extend(dose_in_structure_in_slice)

    return dose_in_structure


def plot_dvh(dose_in_structure: np.ndarray):
    """
    Plot the dose volume histogram (DVH) for a specific structure.

    Parameters
    ----------
    dose_in_structure : np.ndarray
        A numpy array containing the dose values for the specified structure.
    
    Returns
    -------
    None
        Displays the DVH plot.
    
    Notes
    -----
    The DVH is plotted as a histogram with cumulative density.
    """

    fig = go.Figure(
        data = [
            go.Histogram(
                x=dose_in_structure,
                histnorm="percent",
                cumulative_enabled=True,
                cumulative={
                    "direction": "decreasing",
                },
                xbins = dict( # bins used for histogram
                    start = 0,
                    end = np.amax(dose_in_structure),
                    size = 0.05
                ),
                opacity=0.75
            )
        ]
    )
    fig.update_layout(
        title_text='DVH', # title of plot
        xaxis_title_text='Dose [Gy]', # xaxis label
        yaxis_title_text='Volume [%]', # yaxis label
    )

    fig.show()


def _is_dicom(path_to_file: str | Path):

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