from PySide6.QtWidgets import (
      QFileDialog,
      QDialog,
      QDialogButtonBox,
      QVBoxLayout,
      QLabel,
)
from PySide6.QtCore import QDir

from pathlib import Path

def open_files_dialog(filter):
    """
    Use of QFileDialog to get a list of paths to files.

    Paramters
    ---------
    filter : str
        Filters used by QFileDialog.

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
    dialog.setDirectory(QDir.home())
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    dialog.setNameFilter(filter)

    if dialog.exec():
        filenames = dialog.selectedFiles()
        list_files = [str(Path(f_name)) for f_name in filenames]
        return list_files
    
    else:
        return None
    

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