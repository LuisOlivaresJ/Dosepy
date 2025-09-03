"""This module contains tools to calculate dose volume histograms (DVH)."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import FileDataset
from skimage.draw import polygon
import plotly.graph_objects as go


class StructureSet:

    def __init__(self, ds : FileDataset):
        self.ds = ds
    
    def get_names(self):
        """
        Get the structure names in a DataSet.

        Returns
        -------
        dict[str, int]
            A dictionary with structure names as keys and their corresponding index.
        """
        
        # Get the structure names
        structures = [s.ROIName for s in self.ds.StructureSetROISequence]
        # Get the structure numbers
        structure_numbers = [int(s.ROINumber) for s in self.ds.StructureSetROISequence]
        # Check if there are any structures
        if not structures:
            raise ValueError(f"No structures found in the DataSet.")
        
        # Create a dictionary with structure names and their corresponding index
        structures = {name: number for name, number in zip(structures, structure_numbers)}
        
        return structures
    
    def get_coordinates(self, structure: str) -> list[np.ndarray]:
        """
        Get the coordinates of a structure.
        Parameters
        ----------
        structure : str
            The name of the structure to get the coordinates for.
        
        Returns
        -------
        list[np.ndarray]
            A list of numpy arrays of shape (N, 3). Each array represents a slice, 
            and contains the coordinates of the structure as (x, y, z = constant).
        
        """
        # Check if structure is a string
        if not isinstance(structure, str):
            raise TypeError("structure must be a string.")

        # Create a dictionary with structure names and their corresponding index
        structures_names = [s.ROIName for s in self.ds.StructureSetROISequence]
        structure_numbers = [int(s.ROINumber) for s in self.ds.StructureSetROISequence]

        structures = {name: number for name, number in zip(structures_names, structure_numbers)}

        # Check if the structure is in the list of structures
        if structure not in structures:
            raise ValueError(f"The structure {structure} is not in the dataset.")

        # Get index of the structure
        structure_index = structures.get(structure) - 1

        # Get the structure coordinates
        coordinates = []
        for slice in self.ds.ROIContourSequence[structure_index].ContourSequence:
            points = slice.NumberOfContourPoints
            slice_coordinates = np.array(slice.ContourData).reshape(points, 3)
            coordinates.append(slice_coordinates)
        
        return coordinates
    
    def get_physical_size(self, structure: str) -> float:
        pass


def pydicom_to_simpleitk(ds: pydicom.FileDataset) -> sitk.Image:
    """
    Convert a pydicom FileDataset to a SimpleITK Image.

    Parameters
    ----------
    ds : pydicom.FileDataset
        The pydicom FileDataset to convert.

    Returns
    -------
    sitk.Image
        The converted SimpleITK Image.
    
    Notes
    -----
    Dose distribution is represented by a SimpleITK.Image.
    """

    # Check if the tag '3004|000E' (DoseGridScaling) exists in the metadata
    if not 'DoseGridScaling' in ds:
        raise ValueError("The DICOM file does not contain the required metadata tag DoseGridScaling (3004|000e).")

    # Convert pydicom FileDataset to SimpleITK Image.

    array = pydicom.pixels.pixel_array(ds).astype(np.float64)
    img = sitk.GetImageFromArray(array)

    # Set image origin, spacing, and direction from DICOM metadata
    img.SetOrigin((
        float(ds.ImagePositionPatient[0]),
        float(ds.ImagePositionPatient[1]),
        float(ds.ImagePositionPatient[2])
    ))
    img.SetSpacing((
        float(ds.PixelSpacing[0]),
        float(ds.PixelSpacing[1]),
        float(_get_z_spacing_from_dose_as_frames(ds))
    ))
    img.SetDirection((
        float(ds.ImageOrientationPatient[0]), float(ds.ImageOrientationPatient[1]), float(ds.ImageOrientationPatient[2]),
        float(ds.ImageOrientationPatient[3]), float(ds.ImageOrientationPatient[4]), float(ds.ImageOrientationPatient[5]),
        0.0, 0.0, 1.0
    ))

    # Convert image to a dose distribution
    dose = img * float(ds.DoseGridScaling)

    return dose


def _get_z_spacing_from_dose_as_frames(ds: pydicom.FileDataset):
    # Check if elements in a list have unifrom spacing in z direction
    if ds["NumberOfFrames"].value < 2:
        # Trivial case: 0 or 1 element has uniform spacing
        raise ValueError("The DICOM file must contain at least 2 frames to calculate the z spacing.")
    # Create a list of offsets
    offsets = [s for s in ds["GridFrameOffsetVector"]]
    # Calculate the initial expwcted spacing in z direction
    z_spacing = offsets[1] - offsets[0]
    # Iterate through the rest of the list and compare differences
    for i in range(2, len(offsets)):
        difference = offsets[i] - offsets[i-1]
        if difference != z_spacing:
            raise ValueError("The DICOM file does no have a uniform spacing in z direction")
        
    return z_spacing


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
    
    Notes
    -----
    Dose distribution is represented by a SimpleITK.Image. Therefore we
    can use the SimpleITK library to manipulate and analyze the dose distribution.
    For more information about SimpleITK, see:
    https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html
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


def load_structures(path_to_file: str) -> StructureSet:
    """ Wraps the pydicom.dcmread function to load a DICOM RTSTRUCT file."""

    _is_dicom(path_to_file=path_to_file)

    # Read the DICOM file
    ds = pydicom.dcmread(path_to_file)

    # Check if the DICOM file represents structures, using SOP class name RT Structure Set Storage
    if not ds.get("SOPClassUID") == '1.2.840.10008.5.1.4.1.1.481.3':
        raise ValueError(f"{path_to_file} is not a valid DICOM RTSTRUCT file.")
    
    return StructureSet(ds=ds)


def get_axial_dose_plane(
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


def get_sagital_dose_plane(
    dose: sitk.Image,
    coordinate: float) -> sitk.Image:
    """Get a 2D dose sagital plane at a specific x-coordinate from a 3D dose distribution.
    Parameters
    ----------
    dose : sitk.Image
        The 3D dose distribution.
    coordinate : float
        The x-coordinate at which to extract the dose plane.
    Returns
    -------
    sitk.Image
        A 2D dose plane at the specified x-coordinate.
    """

    # Check if coordinate is a number
    if not isinstance(coordinate, (int, float)):
        raise ValueError(f"Invalid parameter for coordinate. It has to be a number")
    # Check if coordinate is between coordinates of the dose distribution
    x_min = dose.GetOrigin()[0]
    x_max = dose.GetOrigin()[0] + dose.GetWidth() * dose.GetSpacing()[0]
    if not x_min <= coordinate <= x_max:
        raise ValueError(
            f"The coordinate {coordinate} is not between the coordinates of the dose distribution "
            f"({x_min}, {x_max})."
        )

    # Reference image
    reference_image = sitk.Image(
        (1, dose.GetHeight(), dose.GetDepth()),
        dose.GetPixelIDValue()
    )
    reference_image.SetOrigin((coordinate, dose.GetOrigin()[1], dose.GetOrigin()[2]))
    reference_image.SetDirection(dose.GetDirection())
    reference_image.SetSpacing(dose.GetSpacing())

    interpolated_dose = sitk.Resample(
        dose,
        reference_image,
        sitk.Transform(3, sitk.sitkIdentity),  # Do not apply any transformation
        sitk.sitkLinear,  # Uses linear interpolation
    )
    
    return interpolated_dose[0, :, :]


def get_coronal_dose_plane(
    dose: sitk.Image,
    coordinate: float) -> sitk.Image:
    """Get a 2D dose coronal plane at a specific y-coordinate from a 3D dose distribution.
    Parameters
    ----------
    dose : sitk.Image
        The 3D dose distribution.
    coordinate : float
        The y-coordinate at which to extract the dose plane.
    Returns
    -------
    sitk.Image
        A 2D dose plane at the specified y-coordinate.
    """
    # Check if coordinate is a number
    if not isinstance(coordinate, (int, float)):
        raise ValueError(f"Invalid parameter for coordinate. It has to be a number")
    # Check if coordinate is between coordinates of the dose distribution
    y_min = dose.GetOrigin()[1]
    y_max = dose.GetOrigin()[1] + dose.GetHeight() * dose.GetSpacing()[1]
    if not y_min <= coordinate <= y_max:
        raise ValueError(
            f"The coordinate {coordinate} is not between the coordinates of the dose distribution "
            f"({y_min}, {y_max})."
        )
    # Reference image
    reference_image = sitk.Image(
        (dose.GetWidth(), 1, dose.GetDepth()),
        dose.GetPixelIDValue()
    )
    reference_image.SetOrigin((dose.GetOrigin()[0], coordinate, dose.GetOrigin()[2]))
    reference_image.SetDirection(dose.GetDirection())
    reference_image.SetSpacing(dose.GetSpacing())

    interpolated_dose = sitk.Resample(
        dose,
        reference_image,
        sitk.Transform(3, sitk.sitkIdentity),  # Do not apply any transformation
        sitk.sitkLinear,  # Uses linear interpolation
    )

    return interpolated_dose[:, 0, :]


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


def get_dose_in_structure_by_plane(
        dose_2D: sitk.Image,
        coordinates: np.ndarray) -> np.ndarray:
    
    # Check that dose_2D is a SimpleITK image
    if not isinstance(dose_2D, sitk.Image):
        raise TypeError("dose_2D must be a SimpleITK.Image object.")
    # Check dose_2D dimension
    if dose_2D.GetDimension() != 2:
        raise ValueError("dose_2D must be a 2D SimpleITK.Image object.")

    # Check coordinates is a numpy array
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("coordinates must be a numpy array.")
    # Check that the array of coordinates has the right shape
    if not coordinates.shape[1] == 3:
        raise ValueError("coordinates must be a 2D array with shape (N, 3), where N is the number of points.")

    
    mask = get_2D_mask_by_coordinates_and_image_shape(coordinates, dose_2D)
    dose_values = sitk.GetArrayFromImage(dose_2D)[mask]
    
    return dose_values


def get_dose_in_structure(
    dose_distribution: sitk.Image,
    coordinates: list[np.ndarray]) -> np.ndarray:
    """
    Get the dose points in a structure from a 3D dose distribution.

    Parameters
    ----------
    dose_distribution : sitk.Image
        The 3D dose distribution.
    coordinates : list[np.ndarray]
        A list of numpy arrays, where each array contains the coordinates of a slice of the structure
        in the format (x, y, z). Each slice must have a constant z coordinate
    
    Returns
    -------
    list[float]
        A list of dose values in the structure.
    
    Raises
    ------
    TypeError
        If dose_distribution is not a SimpleITK.Image object or coordinates is not a list of numpy arrays.
    ValueError
        If any of the slice coordinates do not have a constant z coordinate.
    """
    # Check that dose_distribution is a SimpleITK image
    if not isinstance(dose_distribution, sitk.Image):
        raise TypeError("dose_distribution must be a SimpleITK.Image object.")
    # Check that coordinates is a list of numpy arrays
    if not isinstance(coordinates, list):
        raise TypeError("coordinates must be a list.")

    dose_in_structure = []
    for slice_coordinates in coordinates:
        # Check that each slice_coordinates has a constant z coordinate
        if not len(set(slice_coordinates[:, 2])) == 1:
            raise ValueError("Each slice_coordinates must have a constant z coordinate.")
        
        z_coordinate = slice_coordinates[0, 2]

        dose_2D = get_axial_dose_plane(dose_distribution, z_coordinate)
        dose_in_structure_in_slice = get_dose_in_structure_by_plane(dose_2D, slice_coordinates)
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
        #title_text='DVH', # title of plot
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