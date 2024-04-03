from PySide6.QtWidgets import (
      QFileDialog,
      QDialog,
      QDialogButtonBox,
      QVBoxLayout,
      QLabel,
)
from PySide6.QtCore import QDir

from pathlib import Path

def open_files_dialog(filter, dir = "home") -> list:
    """
    Use of QFileDialog to get a list of strings (paths) to the files selected.

    Paramters
    ---------
    filter : str
        Filters used by QFileDialog.
    dir : str
        Sets the file dialog's current directory.
        "home", "calib"

    Returns
    -------
    list : list 
        List of strings containing the path to the files. 
        If not files selected, returns None

    Examples
    --------
    filter = "Images (*.tif *.tiff)"
    files = open_files_dialog(filter=filter)

    """
    dialog = QFileDialog()

    if dir == "home":
        dialog.setDirectory(QDir.home())

    elif dir == "calib":
        cali_path = Path(__file__).parent.parent / "user" / "calibration"
        dialog.setDirectory(str(cali_path))

    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setNameFilter(filter)

    if dialog.exec():
        filenames = dialog.selectedFiles()
        list_files = [str(Path(f_name)) for f_name in filenames]
        return list_files
    
    else:
        return None
    

def save_lut_file_dialog(root_directory: str):
    file_path, _ = QFileDialog.getSaveFileName(
        None,
        "Save as",
        root_directory,
        )
    return file_path

class Error_Dialog(QDialog):
    """
    Basic QDialog to show an error message.
    
    Parameters
    ----------
    msg : str
        The message to show in the dialog window.
    """
    def __init__(self, msg):
        super().__init__()

        self.setWindowTitle("Error")

        buttons = (QDialogButtonBox.StandardButton.Ok)
        
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        message = QLabel(msg)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)