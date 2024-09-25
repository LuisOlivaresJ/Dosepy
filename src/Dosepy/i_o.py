"""I/O helper functions"""

from pathlib import Path
import os
#from pylinac.core.io import get_url
import pydicom
import struct
from pydicom.errors import InvalidDicomError

from tqdm import tqdm
from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError, URLError


def get_url(
    url: str, destination: str | Path | None = None, progress_bar: bool = True
) -> str:
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    This fuction is copied from pylinac https://pylinac.readthedocs.io/en/latest/overview.html
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    try:
        if progress_bar:
            with tqdm(
                unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
            ) as t:
                filename, _ = urlretrieve(
                    url, filename=destination, reporthook=my_hook(t)
                )
        else:
            filename, _ = urlretrieve(url, filename=destination)
    except (HTTPError, URLError, ValueError) as e:
        raise e
    return filename


def retrieve_demo_file(name: str, force: bool = False) -> Path:
    """Retrieve the demo file either by getting it from file or from a URL.

    If the file is already on disk it returns the file name. If the file isn't
    on disk, get the file from the URL and put it at the expected demo file location
    on disk for lazy loading next time.

    Parameters
    ----------
    name : str
        File name.
    """

    urls = {
        "QA_Post.tif": r"https://raw.githubusercontent.com/LuisOlivaresJ/Dosepy/main/docs/Jupyter/",
        "QA_Pre.tif": r"https://raw.githubusercontent.com/LuisOlivaresJ/Dosepy/main/docs/Jupyter/",
        "cal_48h.tif": r"https://raw.githubusercontent.com/LuisOlivaresJ/Dosepy/main/docs/Jupyter/",
        "verif.tif": r"https://raw.githubusercontent.com/LuisOlivaresJ/Dosepy/main/docs/Jupyter/",
        "RD_20x20cm2_256x256pix.dcm": r"https://raw.githubusercontent.com/LuisOlivaresJ/Dosepy/main/docs/Jupyter/",
        "D_FILM.csv": r"https://raw.githubusercontent.com/LuisOlivaresJ/Dosepy/main/docs/Jupyter/",
        "D_TPS.csv": r"https://raw.githubusercontent.com/LuisOlivaresJ/Dosepy/main/docs/Jupyter/",
        }
    
    url = urls[name] + name
    demo_path = Path(__file__).parent / "demo_files" / name
    
    demo_dir = demo_path.parent
    if not demo_dir.exists():
        os.makedirs(demo_dir)
    if force or not demo_path.exists():
        get_url(url, destination = demo_path)
    return demo_path

def retrieve_dicom_file(file: str | Path) -> pydicom.FileDataset:
    """Read and return the DICOM dataset.

    Parameters
    ----------
    file : str
        The path to the file.
    """
    ds = pydicom.dcmread(file, force=True)
    return ds

def is_dicom_image(file: str | Path) -> bool:
    """Boolean specifying if file is a proper DICOM file with a image

    Parameters
    ----------
    file : str
        The path to the file.

    See Also
    --------
    pydicom.filereader.read_preamble
    pydicom.filereader.read_partial
    """
    result = False
    try:
        pydicom.dcmread(file)
        result = True
    except InvalidDicomError as error:
        #print(f"A {type(error).__name__} has occurred.")
        pass

    return result