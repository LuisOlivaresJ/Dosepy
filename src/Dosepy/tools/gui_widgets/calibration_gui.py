"""
A widget to manage film calibration.
"""

from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QComboBox,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QSizePolicy,
    QDialog,
    QTableWidget,
)
from PySide6.QtCore import Qt, QSize

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar

import numpy as np

class CalibrationWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout used for two widgets: FigureCanvas and parameters_widget.
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        

        # Paramters Widget
        parameters_widget = QWidget()
        parameters_layout = QVBoxLayout()
        parameters_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        parameters_widget.setLayout(parameters_layout)

        self.open_button = QPushButton("Browse")
        self.open_button.setMinimumSize(QSize(150, 50))
        #self.clear_button = QPushButton("Clear")
        self.files_list = QListWidget()
        self.files_list.setMaximumSize(QSize(320, 100))

        self.dose_table = QTableWidget()

        self.channel_combo_box = QComboBox()
        self.channel_combo_box.addItems(["Red", "Green", "Blue"])
        self.fit_combo_box = QComboBox()
        self.fit_combo_box.addItems(["Rational", "Polynomial"])

        parameters_layout.addWidget(self.open_button)
        #parameters_layout.addWidget(self.clear_button)
        parameters_layout.addWidget(self.files_list)
        parameters_layout.addWidget(self.dose_table, 1)
        parameters_layout.addSpacing(30)
        parameters_layout.addWidget(QLabel("Channel:"))
        parameters_layout.addWidget(self.channel_combo_box)
        parameters_layout.addWidget(QLabel("Fit function:"))
        parameters_layout.addWidget(self.fit_combo_box)
        parameters_layout.addStretch()

        main_layout.addWidget(parameters_widget)
        #print(self.children())


        # Plots Widget
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        fig = Figure(
            #figsize=(3, 5),
            layout="constrained"
            )
        self.canvas = FigureCanvas(fig)

        self.axe_image = fig.add_subplot(2, 1, 1)
        self.axe_curve = fig.add_subplot(2, 1, 2)
        
        #main_layout.addWidget(FigureCanvas(self.fig), 1) 
        plot_layout.addWidget(NavigationToolbar(self.canvas, self))
        plot_layout.addWidget(self.canvas)
        """The second argument (1) is used as a strech factor. 
        Widgets with higher stretch factors grow more on 
        window resizing.
        """


        main_layout.addWidget(plot_widget)

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
    
    def set_table_rows(self, rows: int):
        self.dose_table.setRowCount(rows)
        self.dose_table.setColumnCount(1)
        self.dose_table.setHorizontalHeaderLabels(["Dose [cGy]"]) 

    def plot(self, img):
        """
        Show an array.

        Parameters
        ----------
        img : Dosepy.tools.image.TiffImage
            
        """
        img.plot(ax = self.axe_image, show=False)
        self.canvas.draw()
