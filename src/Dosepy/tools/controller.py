"""Class used as a controller in a MVC pattern."""

from PySide6.QtWidgets import (
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QHeaderView,
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
                self._view.cal_widget.get_files_list()
            
            if filenames:
                if self._model.are_valid_tif_files(list_files):
                    if self._model.are_files_equal_shape(list_files):

                        # Display path to files
                        self._view.cal_widget.set_files_list(list_files)

                        # load the files
                        self._model.calibration_img = self._model.load_files(
                            list_files,
                            for_calib=True
                            )
                        self._view.cal_widget.plot(self._model.calibration_img)

                        # Find how many film we have and show a table for user input dose values
                        self._model.calibration_img.set_labeled_img()
                        num = self._model.calibration_img.number_of_films
                        print(f"Number of detected films: {num}")
                        self._view.cal_widget.set_table_rows(rows = num)
                        header = self._view.cal_widget.dose_table.horizontalHeader()
                        self._view.cal_widget.dose_table.cellChanged.connect(self._is_a_valid_dose)
                        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

                        self._view.cal_widget.apply_button.setEnabled(True)
                    
                    else:
                        msg = "The tiff files must have the same shape."
                        print(msg)
                        Error_Dialog(msg).exec()

                else:
                    msg = "Invalid file. Is it a tiff RGB file?"
                    print(msg)
                    Error_Dialog(msg).exec()

    def _apply_calib_button(self):
        print("Apply button pressed.")
        # Is dose table completed?
        num = self._model.calibration_img.number_of_films
        if self._view.cal_widget.is_dose_table_complete(num):
        #if all([self._view.cal_widget.dose_table.item(row, 0) for row in range(num)]):
            print("Doses OK")
            dose_list = [self._view.cal_widget.dose_table.item(row, 0).text() for row in range(num)]
            doses = sorted([float(dose) for dose in dose_list])
            print(doses)
        else:
            msg = "Invalid dose values"
            print(msg)
            Error_Dialog(msg).exec()

    def _clear_file_button(self):
        #TODO_
        self._view.cal_widget.files_list.clear()

    # Secondary functions
    # -------------------
    # TODO_ 
    # Is here, in controller, a good place to have this funtion?
    # Is there another approach to resolve this?
    # Negative values are not valid doses.
    def _is_a_valid_dose(self, row):
        print(f"The row is: {row + 1}")
        data = self._view.cal_widget.dose_table.item(row, 0).text()
        try:
            float(data)
            #self._view.cal_widget.dose_table.item(row, 0).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            print("Data OK")
        except ValueError:
            print("Bad dose input. Changing to 0")
            self._view.cal_widget.dose_table.item(row, 0).setText("0")

    def _connectSignalsAndSlots(self):
        # Calibration Widget
        self._view.cal_widget.open_button.clicked.connect(self._open_file_button)
        self._view.cal_widget.apply_button.clicked.connect(self._apply_calib_button)
        #self._view.cal_widget.clear_button.clicked.connect(self._clear_file_button) #TODO_
            

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
