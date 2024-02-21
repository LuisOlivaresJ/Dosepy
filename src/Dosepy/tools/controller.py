"""Class used as a controller in a MVC pattern."""

from PySide6.QtWidgets import QFileDialog
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
            

    def _connectSignalsAndSlots(self):
        self._view.calibration_widget.open_button.clicked.connect(self._open_file_button)
