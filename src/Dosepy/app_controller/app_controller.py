"""Classes used as controllers in a ModelViewControll (MVC) pattern."""

from PySide6.QtWidgets import (
    QFileDialog,
    QHeaderView,
    QMessageBox,
)
from pathlib import Path
from abc import ABC, abstractmethod
import os
import numpy as np

from Dosepy.app_components.file_dialog import (
    open_files_dialog,
    save_lut_file_dialog,
    Error_Dialog,
)
from Dosepy.image import load

class BaseController(ABC):
    """Abstract class."""
    def __init__(self, model, view):

        self._model = model
        self._view = view

    @abstractmethod
    def _connectSignalsAndSlots(self):
        pass


class CalibrationController(BaseController):
    """Related to Calibration."""
    def __init__(self, model, view):

        super().__init__(model, view)

        self._connectSignalsAndSlots()
    
        
    def _open_file_button(self):
        new_files = open_files_dialog("Images (*.tif *.tiff)")

        if new_files:
    
            if self._model.are_valid_tif_files(new_files):
                current_files = self._view.cal_widget.get_files_list()
                list_files = current_files + new_files

                if self._model.are_files_equal_shape(list_files):

                    # Display path to files
                    self._view.cal_widget.set_files_list(list_files)

                    # load files
                    self._model.calibration_img = self._model.load_calib_files(
                        list_files,
                        for_calib=True
                        )
                    self._prepare_for_calibration()
                
                else:
                    
                    msg = "Do you want to cut the images to have the same size?"
                    print(msg)
                    dlg = QMessageBox()
                    dlg.setWindowTitle("Equate images")
                    dlg.setText(msg)
                    dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    dlg.setIcon(QMessageBox.Icon.Question)
                    button = dlg.exec()

                    if button == QMessageBox.Yes:
                        # Display path to files
                        self._view.cal_widget.set_files_list(list_files)

                        # load files
                        self._model.calibration_img = self._model.load_calib_files(
                        list_files,
                        )
                        self._prepare_for_calibration()

                    else:

                        msg = "The tiff files must have the same shape."
                        print(msg)
                        Error_Dialog(msg).exec()

            else:
                msg = "Invalid file. Is it a tiff RGB file?"
                print(msg)
                Error_Dialog(msg).exec()


    def _prepare_for_calibration(self):

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


    def _apply_calib_button(self):
        print("Apply button pressed.")
        # Is dose table completed?
        num = self._model.calibration_img.number_of_films
        if self._view.cal_widget.is_dose_table_complete(num):
            #print("Doses OK")
            doses = self._view.cal_widget.get_doses()
            #print(doses)
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
                self._model.lut = cal
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

        lut_file_name = save_lut_file_dialog(
                root_directory = str(root_calibration_path)
            )

        if lut_file_name:
            print(lut_file_name)
        
            self._model.save_lut(str(lut_file_name))
            #self._view.dose_widget.cali_label.setText(
            #    f"Calibration file: " + str(lut_file_name)
            #    )


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


    def _connectSignalsAndSlots(self):
        # Calibration Widget
        self._view.cal_widget.open_button.clicked.connect(self._open_file_button)
        self._view.cal_widget.apply_button.clicked.connect(self._apply_calib_button)
        self._view.cal_widget.save_cal_button.clicked.connect(self._save_calib_button)
        #self._view.cal_widget.clear_button.clicked.connect(self._clear_file_button) #TODO_
            

class Tiff2DoseController(BaseController):
    """Related to Tiff to Dose."""

    def __init__(self, model, view):
        super().__init__(model, view)

        self._connectSignalsAndSlots()

    
    def _open_tif2dose_button(self):
        """
        Uses a QFileDialog window to ask for tif files.
        If files are ok, uses a calibration (ask if not exists) and calculate the dose distribution.
        Show the dose distribution.
        """
        
        new_files = open_files_dialog("Images (*.tif *.tiff)")

        if new_files:
    
            if self._model.are_valid_tif_files(new_files):

                current_files = self._view.dose_widget.get_files_list()
                list_files = current_files + new_files
                
                if self._model.are_files_equal_shape(list_files):

                    # Display path to files
                    self._view.dose_widget.set_files_list(list_files)

                    # load the files
                    self._model.tif_img = self._model.load_files(
                        list_files,
                        )
                    
                    if self._model.lut:
                        self._model.ref_dose_img = self._model.tif_img.to_dose(self._model.lut) # An ArrayImage

                    else:
                        "Open lut"
                        lut_file_path = open_files_dialog(
                            filter = "Calibration. (*.cal)",
                            dir = "calib"
                            )
                        
                        if lut_file_path:
                            if len(lut_file_path) == 1:
                                self._model.lut = self._model.load_lut(lut_file_path[0])
                                self._model.ref_dose_img = self._model.tif_img.to_dose(
                                    self._model.lut
                                    ) # An ArrayImage
                            else:
                                msg = "Chose one calibration file."
                                print(msg)
                                Error_Dialog(msg).exec()
                        else:
                            self._view.dose_widget.files_list.clear()

                    self._view.dose_widget.plot_dose(self._model.ref_dose_img)

                else:
                    
                    msg = "Do you want to equalize the files?"
                    print(msg)
                    dlg = QMessageBox()
                    dlg.setWindowTitle("Equate images")
                    dlg.setText(msg)
                    dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    dlg.setIcon(QMessageBox.Icon.Question)
                    button = dlg.exec()

                    if button == QMessageBox.Yes:
                        # Display path to files
                        #self._view.cal_widget.set_files_list(list_files)
                        self._view.dose_widget.set_files_list(list_files)
                        file_path = list_files[0]
                        self._model.tif_img = load(file_path)  # Placeholder
                        
                        equal_images = equate(list_files, axis="width")

                        merged_images = merge(list_files, equal_images)

                        img = stack_images(merged_images, padding=6)
                        # load the files
                        self._model.tif_img.array = img.array 

                        if self._model.lut:
                            self._model.ref_dose_img = self._model.tif_img.to_dose(self._model.lut) # An ArrayImage

                        else:
                            "Open lut"
                            lut_file_path = open_files_dialog(
                                filter = "Calibration. (*.cal)",
                                dir = "calib"
                                )
                            
                            if lut_file_path:
                                if len(lut_file_path) == 1:
                                    self._model.lut = self._model.load_lut(lut_file_path[0])
                                    self._model.ref_dose_img = self._model.tif_img.to_dose(
                                        self._model.lut
                                        ) # An ArrayImage
                                else:
                                    msg = "Chose one calibration file."
                                    print(msg)
                                    Error_Dialog(msg).exec()
                            else:
                                self._view.dose_widget.files_list.clear()

                        self._view.dose_widget.plot_dose(self._model.ref_dose_img)
                        

                    else:

                        msg = "The tiff files must have the same shape."
                        print(msg)
                        Error_Dialog(msg).exec()
            
            else:
                msg = "Invalid file. Is it a tiff RGB file?"
                print(msg)
                Error_Dialog(msg).exec()


    def _save_tif2dose_button(self):

        print("Hola save as tif button")
        root_dose_path = Path(__file__).parent / "user" / "dose distr"
        if not root_dose_path.exists():
            os.makedirs(root_dose_path)

        dose_file_name = save_lut_file_dialog(
            root_directory = str(root_dose_path)
        )

        if dose_file_name:
            print(dose_file_name)
        
            self._model.save_dose_as_tif(str(dose_file_name))

    # end related to tiff2dose
    # --------------------------
    ############################
    
    def _connectSignalsAndSlots(self):

        self._view.dose_widget.open_button.clicked.connect(self._open_tif2dose_button)
        self._view.dose_widget.save_button.clicked.connect(self._save_tif2dose_button)

