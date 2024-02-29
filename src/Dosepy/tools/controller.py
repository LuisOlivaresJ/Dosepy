"""Class used as a controller in a ModelViewControll (MVC) pattern."""

from PySide6.QtWidgets import (
    QFileDialog,
    QHeaderView,
)
from pathlib import Path
import os
import numpy as np

from gui_widgets.file_dialog import open_files_dialog, Error_Dialog

class DosepyController():
    def __init__(self, model, view):
        self._model = model
        self._view = view
        self._lut = None
        self._connectSignalsAndSlots()
    
    ########################
    #-----------------------
    # Related to Calibration
        
    def _open_file_button(self):
        new_files = open_files_dialog("Images (*.tif *.tiff)")

        if new_files:
    
            if self._model.are_valid_tif_files(new_files):
                current_files = self._view.cal_widget.get_files_list()
                list_files = current_files + new_files
                if self._model.are_files_equal_shape(list_files):

                    # Display path to files
                    self._view.cal_widget.set_files_list(list_files)

                    # load the files
                    self._model.calibration_img = self._model.load_files(
                        list_files,
                        for_calib=True
                        )
                    self._view.cal_widget.plot_image(self._model.calibration_img)

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
            print("Doses OK")
            doses = self._view.cal_widget.get_doses()
            print(doses)
            if self._model.calibration_img:
                cal = self._model.calibration_img.get_calibration(
                    doses = doses,
                    channel = self._view.cal_widget.channel_combo_box.currentText(),
                    roi = (16, 8),
                    func = self._view.cal_widget.fit_combo_box.currentText()
                    )
                self._view.cal_widget.plot_cal_curve(cal)
                self._view.cal_widget.save_cal_button.setEnabled(True)

                # Dosepy implementation.
                self._lut = self._model.create_dosepy_lut(doses, roi=(16, 8))
                # OMG_dosimetry implementation.

            else:
                msg = "Something wrong with the image."
                print(msg)
                Error_Dialog(msg).exec()
        else:
            msg = "Invalid dose values."
            print(msg)
            Error_Dialog(msg).exec()


    def _save_calib_button(self):
        root_calibration_path = Path(__file__).parent.parent / "user" / "calibration"
        if not root_calibration_path.exists():
            os.makedirs(root_calibration_path)

        file_path, _ = QFileDialog.getSaveFileName(
            self._view,
            "Save as",
            str(root_calibration_path),
            "Array (*.npy)"
            )
        print(file_path)
        if file_path != "":
            np.save(file_path, self._lut)


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
            print("Data OK")
        except ValueError:
            print("Bad dose input. Changing to 0")
            self._view.cal_widget.dose_table.item(row, 0).setText("0")

    # end related to calibration
    # --------------------------
    ############################

    def _connectSignalsAndSlots(self):
        # Calibration Widget
        self._view.cal_widget.open_button.clicked.connect(self._open_file_button)
        self._view.cal_widget.apply_button.clicked.connect(self._apply_calib_button)
        self._view.cal_widget.save_cal_button.clicked.connect(self._save_calib_button)
        #self._view.cal_widget.clear_button.clicked.connect(self._clear_file_button) #TODO_

        #self._view.
            
