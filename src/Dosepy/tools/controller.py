"""Class used as a controller in a MVC pattern."""

from PySide6.QtWidgets import (
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)
from pathlib import Path

class DosepyController():
    def __init__(self, model, view):
        self._model = model
        self._view = view
        self._connectSignalsAndSlots()


    def _open_file_button(self):
        print("Hola boton open")
        dialog = QFileDialog()
        dialog.setDirectory(r'C:\images')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.tif *.tiff)")
        if dialog.exec():
            filenames = dialog.selectedFiles()
            # Absolute paths to files
            list_files = [str(Path(filename)) for filename in filenames]
            if filenames:
                if self._model.valid_tif_files(list_files):
                    self._view.calibration_widget.set_files_list(list_files)
                else:
                    msg = "Invalid tiff file. Is it a RGB file?"
                    print(msg)
                    Error_Dialog(msg).exec()


    def _connectSignalsAndSlots(self):
        # Calibration Widget
        self._view.calibration_widget.open_button.clicked.connect(self._open_file_button)
            

class Error_Dialog(QDialog):
    """
    Basic QDialog to show an error message.
    
    Parameters
    ----------
    msg : str
        The message to show.
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
