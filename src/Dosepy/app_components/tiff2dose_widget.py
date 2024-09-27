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
    QToolButton,
)

from PySide6.QtCore import Qt, QSize
from PySide6 import QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar

import numpy as np

from Dosepy.app_components.styles.styles import Size, SizeButton
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
        self.flip_button_h = QToolButton()
        self.flip_button_v = QToolButton()
        self.rotate_cw = QToolButton()
        self.rotate_ccw = QToolButton()
        self.selection_button = QToolButton()
        self.selection_button.setCheckable(True)
        self.cut_button = QToolButton()
        self.cut_button.setEnabled(False)

        layout = QHBoxLayout()
        layout.addWidget(self.flip_button_h)
        layout.addWidget(self.flip_button_v)
        layout.addWidget(self.rotate_cw)
        layout.addWidget(self.rotate_ccw)
        layout.addWidget(self.selection_button)
        layout.addWidget(self.cut_button)

        widget = QWidget()
        widget.setLayout(layout)

        # Setup button icons and sizes
        icons_path = pathlib.Path(__file__).parent.parent.joinpath("Icon")
        icon_file_name = [
            "reflect-horizontal-regular-60.png",
            "reflect-vertical-regular-60.png",
            "rotate-right-regular-60.png",
            "rotate-left-regular-60.png",
            "selection-regular-60.png",
            "cut-regular-60.png"
        ]
        counter = 0
        for w in widget.children():         
            if isinstance(w, QToolButton):
                w.setIconSize(SizeButton.TOOL.value)
                icon_path = icons_path.joinpath(icon_file_name[counter])
                icon = QtGui.QIcon(str(icon_path))
                w.setIcon(icon)
                counter += 1

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
        try:
            self.canvas_widg.figure.axes[1]
            axes_cbar = self.canvas_widg.figure.axes[1]
        except:
            axes_cbar = None
        self.canvas_widg.figure.colorbar(
            pos,
            cax = axes_cbar,
            ax = self.axe_image,
            )
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