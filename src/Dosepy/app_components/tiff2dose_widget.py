"""
A widget to manage tiff2dose.
"""

from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QLabel,
)

from PySide6.QtCore import Qt, QSize
from PySide6 import QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar

import numpy as np

from Dosepy.app_components.styles.styles import Size
import pathlib

class Tiff2DoseWidget(QWidget):
    def __init__(self):
        super().__init__()

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        buttons_widget = self._setup_right_widget()
        plot_widget = self._setup_left_widget()

        main_layout.addWidget(plot_widget, 1)
        main_layout.addWidget(buttons_widget)


    def _setup_right_widget(self) -> QWidget:
        # Paramters and buttons Widget
        parameters_widget = QWidget()
        parameters_layout = QVBoxLayout()
        parameters_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        parameters_widget.setLayout(parameters_layout)

        self.open_button = QPushButton("Browse")
        self.open_button.setMinimumSize(Size.MAIN_BUTTON.value)
        #self.clear_button = QPushButton("Clear")
        self.files_list = QListWidget()
        self.files_list.setMaximumSize(QSize(300, 100))

        #self.cali_label = QLabel("Calibration file: ")
        #self.cali_button = QPushButton("Open")
        self.save_button = QPushButton("Save as tif [in cGy]")

        parameters_layout.addWidget(self.open_button)
        parameters_layout.addWidget(self.files_list)
        #parameters_layout.addWidget(self.cali_label)
        #parameters_layout.addWidget(self.cali_button)
        parameters_layout.addWidget(self.save_button)
        self.save_button.setMinimumSize(Size.MAIN_BUTTON.value)
        parameters_layout.addStretch()

        return parameters_widget
    

    def _setup_left_widget(self) -> QWidget:
        # Plots Widget
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        fig = Figure(
            #figsize=(3, 5),
            layout="constrained"
            )
        self.canvas_widg = FigureCanvas(fig)
        self.canvas_widg.setMinimumSize(QSize(450, 50))

        self.axe_image = fig.add_subplot(1, 1, 1)
        self.axe_image.set_xticks([])
        self.axe_image.set_yticks([])
        
        #plot_layout.addWidget(NavigationToolbar(self.canvas_widg, self))
        plot_buttons_w = self._setup_plot_buttons()
        plot_layout.addWidget(plot_buttons_w)
        plot_layout.addWidget(self.canvas_widg)

        return plot_widget
    

    def _setup_plot_buttons(self) -> QWidget:
        self.flip_button_h = QPushButton()
        self.flip_button_v = QPushButton()
        self.rotate_cw = QPushButton()
        self.rotate_ccw = QPushButton()
        self.selection_button = QPushButton()
        self.selection_button.setCheckable(True)
        self.cut_button = QPushButton()
        self.cut_button.setEnabled(False)

        # Flip horizontal icon
        flip_h_icon_path = pathlib.Path(__file__).parent.parent.joinpath(
            "Icon", "reflect-horizontal-regular-60.png")
        flip_h_icon = QtGui.QIcon(str(flip_h_icon_path))
        self.flip_button_h.setIcon(flip_h_icon)
        #Flip vertical icon
        flip_v_icon_path = pathlib.Path(__file__).parent.parent.joinpath(
            "Icon", "reflect-vertical-regular-60.png")
        flip_v_icon = QtGui.QIcon(str(flip_v_icon_path))
        self.flip_button_v.setIcon(flip_v_icon)
        #Flip rotate right icon
        rotate_r_icon_path = pathlib.Path(__file__).parent.parent.joinpath(
            "Icon", "rotate-right-regular-60.png")
        rotate_r_icon = QtGui.QIcon(str(rotate_r_icon_path))
        self.rotate_cw.setIcon(rotate_r_icon)
        #Flip rotate left icon
        rotate_l_icon_path = pathlib.Path(__file__).parent.parent.joinpath(
            "Icon", "rotate-left-regular-60.png")
        rotate_l_icon = QtGui.QIcon(str(rotate_l_icon_path))
        self.rotate_ccw.setIcon(rotate_l_icon)
        # Selection icon
        selection_icon_path = pathlib.Path(__file__).parent.parent.joinpath(
            "Icon", "selection-regular-60.png")
        selection_icon = QtGui.QIcon(str(selection_icon_path))
        self.selection_button.setIcon(selection_icon)
        # Cut icon
        cut_icon_path = pathlib.Path(__file__).parent.parent.joinpath(
            "Icon", "cut-regular-60.png")
        cut_icon = QtGui.QIcon(str(cut_icon_path))
        self.cut_button.setIcon(cut_icon)

        layout = QHBoxLayout()
        layout.addWidget(self.flip_button_h)
        layout.addWidget(self.flip_button_v)
        layout.addWidget(self.rotate_cw)
        layout.addWidget(self.rotate_ccw)
        layout.addWidget(self.selection_button)
        layout.addWidget(self.cut_button)

        widget = QWidget()
        widget.setLayout(layout)

        return widget

    def plot_dose(self, img):
        """
        Show the transformed tif to dose distribution.

        Parameters
        ----------
        img : Dosepy.tools.image.ArrayImage
        """
        max_dose = np.percentile(img.array, [99.9])[0]
        print(f"Maximum dose: {max_dose}")
        pos = self.axe_image.imshow(img.array, cmap='nipy_spectral')
        pos.set_clim(-0.05, max_dose)
        self.canvas_widg.figure.colorbar(pos, ax=self.axe_image)
        self.canvas_widg.draw()


    def set_files_list(self, files: list):
        """
        Set the files list.
        
        Parameters
        ----------
        files : list
            List of strings containing the absolute 
            paths of the selected files.
        """
        self.files_list.clear()
        self.files_list.addItems(files)


    def get_files_list(self) -> list:
        """
        Get the current files in the view.
        """
        files_list = []
        
        for index in range(self.files_list.count()):
            files_list.append(str(self.files_list.item(index).text()))
        
        return files_list

# Used for development
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    # Create the application
    app = QApplication(sys.argv)
    # Create the main window (view)
    root_window = Tiff2DoseWidget()

    root_window.show()

    sys.exit(app.exec())