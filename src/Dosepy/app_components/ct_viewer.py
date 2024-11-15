# Qtwidget to display CT images

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtWidgets import QGridLayout, QPushButton
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT


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

        # Create three matplotlib widgets to display the CT images
        self.ct_axial_widget = CTWidget(title = "Axial")
        self.ct_coronal_widget = CTWidget(title="Coronal")
        self.ct_sagittal_widget = CTWidget(title = "Sagittal")

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
            "Use scroll wheel to navigate through the slices\n"
            )
        
        # Create a text widget to display the current slice number
        self.slice_number = QLabel()
        self.slice_number.setText("X: 0, Y: 0, Z: 0")

        v_layout_instructions.addWidget(self.load_button)
        v_layout_instructions.addWidget(self.instructions)
        v_layout_instructions.addWidget(self.slice_number)
        
        grid_layout.addLayout(v_layout_instructions, 0, 1)

        # Create an accept button and align it the lower right corner of the main layout
        self.accept_button = QPushButton("Accept")
        self.main_layout.addWidget(self.accept_button, alignment=Qt.AlignmentFlag.AlignRight)


class CTWidget(QWidget):
    """Matplotlib widget to display CT images"""
    def __init__(self, title, parent=None):
        super().__init__()

        self.main_layout = QVBoxLayout()

        self._create_body(title)

        self.setLayout(self.main_layout)


    def _create_body(self, title):
        self.view = FigureCanvas(Figure())
        self.axes = self.view.figure.add_subplot(111)
        self.axes.set_title(title)
        #self.toolbar = NavigationToolbar2QT(self.view, self)
        #self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.view)

