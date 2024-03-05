"""I/O helper functions"""

from pathlib import Path
import os
from pylinac.core.io import get_url

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
        }
    
    url = urls[name] + name
    demo_path = Path(__file__).parent / "demo_files" / name
    
    demo_dir = demo_path.parent
    if not demo_dir.exists():
        os.makedirs(demo_dir)
    if force or not demo_path.exists():
        get_url(url, destination = demo_path)
    return demo_path