# Qtwidget to display CT images

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtWidgets import QGridLayout, QPushButton, QSlider
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from numpy import ndarray


class CTViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.setWindowTitle("CT Viewer")

        self.main_layout = QVBoxLayout()

        self._create_body()

        self.setLayout(self.main_layout)


    def _create_body(self):
        # Create a gird layout to display the CT images
        grid_layout = QGridLayout()

        # Create three matplotlib figure widgets to display the CT images
        self.ct_axial_widget = CTFigureWidget(title = "Axial")
        self.ct_coronal_widget = CTFigureWidget(title="Coronal")
        self.ct_sagittal_widget = CTFigureWidget(title = "Sagittal")

        grid_layout.addWidget(self.ct_axial_widget, 0, 0)
        grid_layout.addWidget(self.ct_coronal_widget, 1, 0)
        grid_layout.addWidget(self.ct_sagittal_widget, 1, 1)
        self.main_layout.addLayout(grid_layout)

        # Create a vertical layout to display the instructions and the slice number
        v_layout_instructions = QVBoxLayout()
        
        # Button to open a file dialog to load the CT images
        self.load_button = QPushButton("Load CT")

        # Create a text box to display instructions for the user
        self.instructions = QLabel()
        self.instructions.setText(
            "Navigate through the slices to set a reference position\n"
            )
        
        # Sliders to navigate through the slices
        self.axial_slider = QSlider()
        self.axial_label = QLabel(f"Axial: {self.axial_slider.value()}")
        self.axial_slider.setOrientation(Qt.Orientation.Horizontal)

        self.coronal_slider = QSlider()
        self.coronal_label = QLabel(f"Coronal: {self.coronal_slider.value()}")
        self.coronal_slider.setOrientation(Qt.Orientation.Horizontal)

        self.sagittal_slider = QSlider()
        self.sagittal_label = QLabel(f"Sagittal: {self.sagittal_slider.value()}")
        self.sagittal_slider.setOrientation(Qt.Orientation.Horizontal)


        v_layout_instructions.addWidget(self.load_button)
        v_layout_instructions.addWidget(self.instructions)
        v_layout_instructions.addWidget(self.axial_label)
        v_layout_instructions.addWidget(self.axial_slider)
        v_layout_instructions.addWidget(self.coronal_label)
        v_layout_instructions.addWidget(self.coronal_slider)
        v_layout_instructions.addWidget(self.sagittal_label)
        v_layout_instructions.addWidget(self.sagittal_slider)
        
        grid_layout.addLayout(v_layout_instructions, 0, 1)

        # Create an accept button and align it the lower right corner of the main layout
        self.accept_button = QPushButton("Accept")
        self.main_layout.addWidget(self.accept_button, alignment=Qt.AlignmentFlag.AlignRight)


class CTFigureWidget(QWidget):
    """Matplotlib widget to display CT images"""
    def __init__(self, title, parent=None):
        super().__init__()

        self.main_layout = QVBoxLayout()

        self._create_body(title)

        self.setLayout(self.main_layout)


    def _create_body(self, title):
        self.view = FigureCanvas(Figure())
        self.ax = self.view.figure.add_subplot(111)
        self.ax.set_title(title)
        self.main_layout.addWidget(self.view)


    def _show_img(self, img: ndarray, aspect: float, **kwargs):

        self.ax.imshow(img, cmap="gray", **kwargs)
        self.ax.set_aspect(aspect)
        self.view.draw()


    def _show_crosshair(self, row: int, column: int):

        self.hline = self.ax.axhline(row, color="red", lw=1)
        self.vline = self.ax.axvline(column, color="red", lw=1)
        self.view.draw()


