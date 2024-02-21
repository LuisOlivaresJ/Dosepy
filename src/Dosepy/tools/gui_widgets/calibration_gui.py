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
)
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas

class CalibrationWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout used for two widgets: FigureCanvas and parameters_widget.
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        parameters_widget = QWidget()
        parameters_layout = QVBoxLayout()
        parameters_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        parameters_widget.setLayout(parameters_layout)

        self.open_button = QPushButton("Browse")
        self.files_list = QListWidget()
        self.channel_combo_box = QComboBox()
        self.channel_combo_box.addItems(["Red", "Green", "Blue"])
        self.fit_combo_box = QComboBox()
        self.fit_combo_box.addItems(["Rational", "Polynomial"])
        parameters_layout.addWidget(self.open_button)
        parameters_layout.addWidget(self.files_list)
        parameters_layout.addSpacing(30)
        parameters_layout.addWidget(QLabel("Channel:"))
        parameters_layout.addWidget(self.channel_combo_box)
        parameters_layout.addWidget(QLabel("Fit function:"))
        parameters_layout.addWidget(self.fit_combo_box)
        parameters_layout.addStretch()

        main_layout.addWidget(parameters_widget)
        print(self.children())

        fig = Figure(
            #figsize=(3, 5),
            layout="constrained"
            )
        self.axe_image = fig.add_subplot(2, 1, 1)
        self.axe_curve = fig.add_subplot(2, 1, 2)
        
        main_layout.addWidget(FigureCanvas(fig), 1) 
        """The second argument (1) is used as a strech factor. 
        Widgets with higher stretch factors grow more on 
        window resizing.
        """

    def set_files_list(self, files: list):
        """
        Set the files list.
        
        Parameters
        ----------
        files : list
            List of strings containing the absolute 
            paths of the selected files.
        """
        self.files_list.addItems(files)