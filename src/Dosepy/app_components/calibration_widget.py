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
    QTableWidget,
    QTableWidgetItem,
)
from PySide6.QtCore import Qt, QSize

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar

from .styles.styles import Size
from enum import Enum


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
        self.open_button.setMinimumSize(Size.MAIN_BUTTON.value)
        #self.clear_button = QPushButton("Clear")
        self.files_list = QListWidget()
        self.files_list.setMaximumSize(QSize(300, 100))

        self.dose_table = QTableWidget()

        self.apply_button = QPushButton("Apply")
        self.apply_button.setMinimumSize(Size.MAIN_BUTTON.value)
        self.apply_button.setEnabled(False)

        self.save_cal_button = QPushButton("Save")
        self.save_cal_button.setMinimumSize(Size.MAIN_BUTTON.value)
        self.save_cal_button.setEnabled(False)

        parameters_layout.addWidget(self.open_button)
        parameters_layout.addWidget(self.files_list)
        parameters_layout.addWidget(self.dose_table, 1)
        parameters_layout.addWidget(self.apply_button)
        parameters_layout.addWidget(self.save_cal_button)
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

        self.axe_image = fig.add_subplot(1, 2, 1)
        self.axe_image.set_xticks([])
        self.axe_image.set_yticks([])
        self.axe_curve = fig.add_subplot(1, 2, 2)
        
        plot_layout.addWidget(NavigationToolbar(self.canvas_widg, self))
        plot_layout.addWidget(self.canvas_widg)
        """The second argument (1) is used as a strech factor. 
        Widgets with higher stretch factors grow more on 
        window resizing.
        """


        main_layout.addWidget(plot_widget, 1)
        main_layout.addWidget(parameters_widget)


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
    
    def get_doses(self) -> list:
        """
        Get the input doses as a list of float numbers.
        """
        num = self.dose_table.rowCount()
        dose_list = [self.dose_table.item(row, 0).text() for row in range(num)]
        doses = sorted([float(dose) for dose in dose_list])
        return doses

    def set_table_rows(self, rows: int):
        self.dose_table.setRowCount(rows)
        self.dose_table.setColumnCount(1)
        self.dose_table.setHorizontalHeaderLabels(["Dose [Gy]"])
        for row in range(rows):
            self.dose_table.setItem(row, 0, QTableWidgetItem(""))
            self.dose_table.item(row, 0).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def is_dose_table_complete(self, rows):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
            
        if all(
            [is_number(self.dose_table.item(row, 0).text()) for row in range(rows)]
            ):
            return True
        
        else: 
            return False


    def plot_image(self, img):
        """
        Show an array image.

        Parameters
        ----------
        img : Dosepy.tools.image.TiffImage
            
        """
        img.plot(ax = self.axe_image, show=False)
        self.canvas_widg.draw()

    def plot_cal_curve(self, cal):
        self.axe_curve.clear()
        cal.plot(self.axe_curve, show=False)
        self.canvas_widg.draw()
