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
import pydicom
from os.path import isfile, join

from Dosepy.app_components.file_dialog import (
    open_files_dialog,
    save_lut_file_dialog,
    Error_Dialog,
)
from Dosepy.i_o import is_dicom_image
from Dosepy.calibration import LUT
from Dosepy.tiff2dose import Tiff2DoseM, T2D_METHOD_MAP
from Dosepy.image import ArrayImage


class BaseController(ABC):
    """Abstract class."""
    def __init__(self, model, view):

        self._model = model
        self._view = view

    @abstractmethod
    def _connectSignalsAndSlots(self):
        pass


# Class that controls the main toolbar widget
class ToolbarController(BaseController):
    """Related to the toolbar."""
    def __init__(self, model, view):
        super().__init__(model, view)

        self._connectSignalsAndSlots()


    def _open_calibration_settings(self):

        if self._view.conf_window.isVisible():
            self._view.conf_window.hide()
        else:
            self._view.conf_window.show()

    
    def _event_handler_ct_viewer(self):
        print("CT Viewer")
        if self._view.ct_viewer.isVisible():
            self._view.ct_viewer.hide()
        else:
            self._view.ct_viewer.show()


    def _save_settings(self):
        print("Saving settings")
        roi_size_h = self._view.conf_window.roi_size_h.text()
        roi_size_v = self._view.conf_window.roi_size_v.text()
        # TODO Performe validation

        channel = self._view.conf_window.channel.currentText()
        fit_function = self._view.conf_window.fit_function.currentText()

        settings = self._model.config

        settings.set_calib_roi_size((float(roi_size_h), float(roi_size_v)))

        self._view.conf_window.roi_size_h_label.setText(
            f"ROI size horizontal (mm): {settings.get_calib_roi_size()[0]}")
        self._view.conf_window.roi_size_v_label.setText(
            f"ROI size vertical (mm): {settings.get_calib_roi_size()[1]}")
        
        settings.set_channel(channel)
        self._view.conf_window.channel_label.setText(
            f"Channel: {settings.get_channel()}"
            )
        settings.set_fit_function(fit_function)
        self._view.conf_window.fit_label.setText(
            f"Fit function: {settings.get_fit_function()}"
            )


    def _event_handler_open_ct_button(self):

        slices = self._open_ct_slices()
        if slices:
            # Save the ct image in the model
            self._model.ct_array_img = self._create_array_from_dicom(slices)

            # Compute aspect ratio and initial index
            self._compute_aspect_ratio(slices)
            self._compute_initial_index(slices)

            # Update sliders and labels
            self._setup_sliders()
            self._update_labels()

            # Update the plot
            self._update_ct_axial_plot()
            self._update_ct_coronal_plot()
            self._update_ct_sagittal_plot()

            # Update crosshair
            self._update_ct_axial_crosshair()
            self._update_ct_coronal_crosshair()
            self._update_ct_sagittal_crosshair()


    def _open_ct_slices(self) -> list[pydicom.dataset.FileDataset]:
        files = open_files_dialog(filter="DICOM (*.dcm)")
        
        if files:

            #TODO check if files are valid dicom files

            ct_files = [pydicom.dcmread(f) for f in files if isfile(f) and is_dicom_image(f)]

            # skip files with no SliceLocation (eg scout views)
            slices = []
            skipcount = 0
            for f in ct_files:
                if hasattr(f, "SliceLocation"):
                    slices.append(f)
                else:
                    skipcount = skipcount + 1

            print(f"skipped, no SliceLocation: {skipcount}")
    
            return slices
        
        else:
            return None
    

    def _create_array_from_dicom(self, slices: list) -> np.ndarray:
            # ensure they are in the correct order
            slices = sorted(slices, key=lambda s: s.SliceLocation)

            # create 3D array
            img_shape = list(slices[0].pixel_array.shape)
            img_shape.append(len(slices))
            img3d = np.zeros(img_shape)

            # fill 3D array with the images from the files
            for i, s in enumerate(slices):
                img2d = s.pixel_array
                img3d[:, :, i] = img2d

            return img3d
    

    def _compute_aspect_ratio(self, slices: list):
        # Assume all slices are the same
        ps = slices[0].PixelSpacing
        ss = slices[0].SliceThickness
        ax_aspect = ps[1] / ps[0]
        sag_aspect = ps[1] / ss
        cor_aspect = ss / ps[0]

        self._model.ct_aspect = {
            "axial": ax_aspect,
            "sagittal": sag_aspect,
            "coronal": cor_aspect,
        }


    def _compute_initial_index(self, slices: list):
        # Assume all slices are the same
        self._model.ct_index = [
            slices[0].Rows // 2,
            slices[0].Columns // 2,
            len(slices) // 2,
        ]
        print("Inside _compute_initial_index method")
        print(f"Initial index: {self._model.ct_index}")


    def _setup_sliders(self):
        # Set sliders
        self._view.ct_viewer.axial_slider.setMinimum(0)
        self._view.ct_viewer.coronal_slider.setMinimum(0)
        self._view.ct_viewer.sagittal_slider.setMinimum(0)

        self._view.ct_viewer.axial_slider.setMaximum(self._model.ct_array_img.shape[2] - 1)
        self._view.ct_viewer.coronal_slider.setMaximum(self._model.ct_array_img.shape[0] - 1)
        self._view.ct_viewer.sagittal_slider.setMaximum(self._model.ct_array_img.shape[1] - 1)

        self._view.ct_viewer.axial_slider.setValue(self._model.ct_index[2])
        self._view.ct_viewer.coronal_slider.setValue(self._model.ct_index[0])
        self._view.ct_viewer.sagittal_slider.setValue(self._model.ct_index[1])


    def _update_labels(self):
        # Update labels
        self._view.ct_viewer.axial_label.setText(f"Axial: {self._model.ct_index[2]}")
        self._view.ct_viewer.coronal_label.setText(f"Coronal: {self._model.ct_index[0]}")
        self._view.ct_viewer.sagittal_label.setText(f"Sagittal: {self._model.ct_index[1]}")


    def _update_ct_axial_plot(self):
        axial = self._view.ct_viewer.ct_axial_widget
        index = self._model.ct_index[2]
        axial._show_img(
            img = self._model.ct_array_img[:, :, index],
            aspect = self._model.ct_aspect["axial"],
            )
        

    def _update_ct_coronal_plot(self):
        coronal = self._view.ct_viewer.ct_coronal_widget
        index = self._model.ct_index[0]
        coronal._show_img(
            img = self._model.ct_array_img[index, :, :].T,
            aspect = self._model.ct_aspect["coronal"],
            origin='lower',
            )


    def _update_ct_sagittal_plot(self):
        sagittal = self._view.ct_viewer.ct_sagittal_widget
        index = self._model.ct_index[1]
        sagittal._show_img(
            img = self._model.ct_array_img[:, index, :],
            aspect = self._model.ct_aspect["sagittal"],
            )


    def _event_handler_axial_slider(self):
        #print(f"Inside axial slider event handler")
        self._model.ct_index[2] = self._view.ct_viewer.axial_slider.value()
        self._update_labels()
        self._update_ct_axial_plot()
        self._update_ct_coronal_crosshair()
        self._update_ct_sagittal_crosshair()

    
    def _event_handler_coronal_slider(self):
        #print(f"Inside coronal slider event handler")
        self._model.ct_index[0] = self._view.ct_viewer.coronal_slider.value()
        self._update_labels()
        self._update_ct_coronal_plot()
        self._update_ct_axial_crosshair()
        self._update_ct_sagittal_crosshair()


    def _event_handler_sagittal_slider(self):
        #print(f"Inside sagittal slider event handler")
        self._model.ct_index[1] = self._view.ct_viewer.sagittal_slider.value()
        self._update_labels()
        self._update_ct_sagittal_plot()
        self._update_ct_axial_crosshair()
        self._update_ct_coronal_crosshair()

    
    def _update_ct_axial_crosshair(self):
        self._view.ct_viewer.ct_axial_widget._show_crosshair(
            row = self._model.ct_index[0],
            column = self._model.ct_index[1],
            )


    def _update_ct_coronal_crosshair(self):
        self._view.ct_viewer.ct_coronal_widget._show_crosshair(
            row = self._model.ct_index[2],
            column = self._model.ct_index[1],
            )
    

    def _update_ct_sagittal_crosshair(self):
        self._view.ct_viewer.ct_sagittal_widget._show_crosshair(
            row = self._model.ct_index[0],
            column = self._model.ct_index[2],
            )


    def _connectSignalsAndSlots(self):
        self._view.calib_setings_action.triggered.connect(self._open_calibration_settings)
        self._view.ct_viewer_action.triggered.connect(self._event_handler_ct_viewer)
        self._view.conf_window.save_button.clicked.connect(self._save_settings)
        
        # CT Viewer
        self._view.ct_viewer.load_button.clicked.connect(self._event_handler_open_ct_button)
        self._view.ct_viewer.axial_slider.sliderReleased.connect(self._event_handler_axial_slider)
        self._view.ct_viewer.coronal_slider.sliderReleased.connect(self._event_handler_coronal_slider)
        self._view.ct_viewer.sagittal_slider.sliderReleased.connect(self._event_handler_sagittal_slider)
        self._view.ct_viewer.accept_button.clicked.connect(self._event_handler_ct_viewer)


class CalibrationController(BaseController):
    """Related to Calibration."""
    def __init__(self, model, view):

        super().__init__(model, view)

        self._connectSignalsAndSlots()
    
        
    def _open_file_button(self):
        new_files = open_files_dialog("Images (*.tif *.tiff)")

        if new_files:
            
            # Validate the files
            if self._model.are_valid_tif_files(new_files):
                
                current_files = self._view.cal_widget.get_files_list()
                list_files = current_files + new_files

                if self._model.are_files_equal_shape(list_files):

                    # Display path to files
                    self._view.cal_widget.set_files_list(list_files)

                    # Load files
                    self._model.calibration_img = self._model.load_files(list_files)
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

                        # Load files
                        self._model.calibration_img = self._model.load_files(
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

        # Show the images in the GUI
        self._view.cal_widget.plot_image(self._model.calibration_img)

        # Find how many film we have and show a table for user input dose values
        self._model.calibration_img.set_labeled_films_and_filters()
        num = self._model.calibration_img.number_of_films
        print(f"Number of detected films: {num}")
        # Update the number of rows in the table
        self._view.cal_widget.set_table_rows(rows = num)
        header = self._view.cal_widget.dose_table.horizontalHeader()
        # Validate dose as the user inputs the values
        self._view.cal_widget.dose_table.cellChanged.connect(self._is_a_valid_dose)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        # Enable the apply calibration button
        self._view.cal_widget.apply_button.setEnabled(True)


    def _apply_calib_button(self):
        print("Apply calibration button pressed.")
        # Is dose table completed?
        num = self._model.calibration_img.number_of_films
        if self._view.cal_widget.is_dose_table_complete(num):
            
            # Get user configured values
            doses = self._view.cal_widget.get_doses()
            roi_size = self._model.config.get_calib_roi_size()
            print(f"{roi_size=}")
            channel = self._model.config.get_channel()
            print(f"{channel=}")
            fit = self._model.config.get_fit_function()
            print(f"{fit=}")
            
            # Create LUT
            #breakpoint()
            #print(doses)
            if self._model.calibration_img:
                """ cal = self._model.calibration_img.get_calibration(
                    doses = doses,
                    channel = channel,
                    roi = roi_size,
                    func = fit,
                    ) 
                """
                cal = LUT(self._model.calibration_img)
                cal.set_central_rois(roi_size)
                cal.set_doses(doses)
                cal.compute_central_lut()
                #breakpoint()

                # Update Plot of the fit function
                self._view.cal_widget.plot_cal_curve(
                    cal,
                    channel=channel.lower(),
                    fit_function = fit.lower(),
                    )
                
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
            self._model.save_lut(str(lut_file_name))


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
            
            # Validate files
            if self._model.are_valid_tif_files(new_files):

                current_files = self._view.dose_widget.get_files_list()
                list_files = current_files + new_files
                
                if self._model.are_files_equal_shape(list_files):

                    # Display path to files
                    self._view.dose_widget.set_files_list(list_files)

                    # load the files
                    self._model.tif_img = self._model.load_files(list_files)
                    self._prepare_for_tif_to_dose()

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
                        self._view.dose_widget.set_files_list(list_files)

                        self._model.tif_img = self._model.load_files(
                            list_files,
                            )
                        
                        self._prepare_for_tif_to_dose()                        

                    else:

                        msg = "The tiff files must have the same shape."
                        print(msg)
                        Error_Dialog(msg).exec()
            
            else:
                msg = "Invalid file. Is it a tiff RGB file?"
                print(msg)
                Error_Dialog(msg).exec()


    def _prepare_for_tif_to_dose(self):
        # Check for LUT
        if not self._model.lut:
            print("Open the lut file")
            lut_file_path = open_files_dialog(
                filter = "Calibration. (*.yaml)",
                dir = "calib"
                )
            
            if lut_file_path:
                if len(lut_file_path) == 1:
                    self._model.lut = self._model.load_lut(lut_file_path[0])
                else:
                    msg = "Choose one calibration file."
                    print(msg)
                    Error_Dialog(msg).exec()
            else:
                self._view.dose_widget.files_list.clear()
        
        # Get user settings
        channel = self._model.config.get_channel()
        fit_function = self._model.config.get_fit_function()

        # Define tiff to dose method
        t2d_method = T2D_METHOD_MAP.get((channel.lower(), fit_function.lower()))

        # Compute dose from tiff
        t2d_manager = Tiff2DoseM()
        # TODO support for channel and fit function
        #t2d_format = "RP"
        self._model.dose_img_from_film = t2d_manager.get_dose(
            self._model.tif_img,
            t2d_method,
            self._model.lut
            )

        # Plot the dose distribution
        self._view.dose_widget.plot_dose(self._model.dose_img_from_film)


    def _save_tif2dose_button(self):

        root_dose_path = Path(__file__).parent / "user" / "dose distr"
        if not root_dose_path.exists():
            os.makedirs(root_dose_path)

        dose_file_name = save_lut_file_dialog(
            root_directory = str(root_dose_path)
        )

        if dose_file_name:
            print(dose_file_name)
        
            self._model.save_dose_as_tif(str(dose_file_name))


    # Related to tiff2dose tool buttons in plot
    def _flip_h_button(self):
        """Flip the the dose distribution in the left/right direction."""
        if self._model.dose_img_from_film is not None:
            self._model.dose_img_from_film.fliplr()
            self._view.dose_widget.plot_dose(self._model.dose_img_from_film)

    def _flip_v_button(self):
        """Flip the the dose distribution in the up/down direction."""
        if self._model.dose_img_from_film is not None:
            self._model.dose_img_from_film.flipud()
            self._view.dose_widget.plot_dose(self._model.dose_img_from_film)

    def _rotate_cw_button(self):
        """Rotate the the dose distribution clockwise."""
        if self._model.dose_img_from_film is not None:
            self._model.dose_img_from_film.rotate(angle = 1)
            self._view.dose_widget.plot_dose(self._model.dose_img_from_film)

    def _rotate_ccw_button(self):
        """Rotate the the dose distribution counter clockwise."""
        if self._model.dose_img_from_film is not None:
            self._model.dose_img_from_film.rotate(angle = -1)
            self._view.dose_widget.plot_dose(self._model.dose_img_from_film)

    def _selection_button(self):
        """Select a region of interest in the dose distribution."""
        if self._view.dose_widget.selection_button.isChecked():
            self._view.dose_widget.rs.set_active(True)

        else:
            self._view.dose_widget.rs.set_active(False)
            #self._view.dose_widget.rs.set_visible(False)


    def _grid_button(self):
        """Show or hide the grid in the plot."""
        self._view.dose_widget.grid()


    def _on_move_plot(self, event):
        """Show the dose value in the plot view label."""
        if event.inaxes == self._view.dose_widget.axe_image and self._model.dose_img_from_film is not None:
            column = int(event.xdata)
            row = int(event.ydata)
            dose = self._model.dose_img_from_film.array[row, column]
            self._view.dose_widget.show_dose_value(column, row, dose)

    def _cut_button(self):
        """Cut the dose distribution based on rectangle selection."""
        xmin, xmax, ymin, ymax = self._view.dose_widget.rs.extents
        self._model.dose_img_from_film.array = self._model.dose_img_from_film.array[int(ymin): int(ymax), int(xmin): int(xmax)]
        self._view.dose_widget.rs.set_visible(False)
        self._view.dose_widget.selection_button.setChecked(False)
        self._view.dose_widget.cut_button.setEnabled(False)
        self._view.dose_widget.plot_dose(self._model.dose_img_from_film)
        self._view.dose_widget._create_rectangle_selector()


    # end related to tiff2dose
    # --------------------------
    ############################
    
    def _connectSignalsAndSlots(self):

        self._view.dose_widget.open_button.clicked.connect(self._open_tif2dose_button)
        self._view.dose_widget.save_button.clicked.connect(self._save_tif2dose_button)
        self._view.dose_widget.flip_button_h.clicked.connect(self._flip_h_button)
        self._view.dose_widget.flip_button_v.clicked.connect(self._flip_v_button)
        self._view.dose_widget.rotate_cw.clicked.connect(self._rotate_cw_button)
        self._view.dose_widget.rotate_ccw.clicked.connect(self._rotate_ccw_button)
        self._view.dose_widget.grid_button.clicked.connect(self._grid_button)
        self._view.dose_widget.selection_button.clicked.connect(self._selection_button)
        self._view.dose_widget.cut_button.clicked.connect(self._cut_button)

        self._view.dose_widget.canvas_widg.figure.canvas.mpl_connect('motion_notify_event', self._on_move_plot)

