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
        #print("Hola boton open")
        dialog = QFileDialog()
        dialog.setDirectory(r'C:\images')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.tif *.tiff)")

        if dialog.exec():
            filenames = dialog.selectedFiles()
            # Absolute paths to files
            list_files = \
                [str(Path(filename)) for filename in filenames] + \
                self._view.calibration_widget.get_files_list()
            
            if filenames:
                if self._model.are_valid_tif_files(list_files):
                    if self._model.are_files_equal_shape(list_files):

                        self._view.calibration_widget.set_files_list(list_files)

                        # load the files
                        img = self._model.load_files(list_files)
                        self._view.calibration_widget.plot(img)
                    
                    else:
                        msg = "The tiff files must have the same shape."
                        print(msg)
                        Error_Dialog(msg).exec()

                else:
                    msg = "Invalid file. Is it a tiff RGB file?"
                    print(msg)
                    Error_Dialog(msg).exec()

    def _clear_file_button(self):
        #TODO_
        self._view.calibration_widget.files_list.clear()


    def _connectSignalsAndSlots(self):
        # Calibration Widget
        self._view.calibration_widget.open_button.clicked.connect(self._open_file_button)
        #self._view.calibration_widget.clear_button.clicked.connect(self._clear_file_button) #TODO_
            

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
