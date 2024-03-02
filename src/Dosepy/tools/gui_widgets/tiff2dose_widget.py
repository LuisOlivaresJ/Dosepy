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
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar

import numpy as np

from .styles.styles import Size

class Tiff2DoseWidget(QWidget):
    def __init__(self):
        super().__init__()

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Paramters Widget
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
        
        plot_layout.addWidget(NavigationToolbar(self.canvas_widg, self))
        plot_layout.addWidget(self.canvas_widg)

        main_layout.addWidget(plot_widget, 1)
        main_layout.addWidget(parameters_widget)

    
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