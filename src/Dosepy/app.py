import sys

from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QLabel,
    QTabWidget,
    QMainWindow,
)

# Import app views
from Dosepy.app_components.calibration_widget import CalibrationWidget
from Dosepy.app_components.tiff2dose_widget import Tiff2DoseWidget
from Dosepy.app_components.tif_widget import TifWidget

# Import app controllers
from Dosepy.app_controller.app_controller import CalibrationController, Tiff2DoseController
# Import app model
from Dosepy.app_model.app_model import Model

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dosepy")

        self.main_widget = QWidget()
        self.tab_layout = QVBoxLayout()

        tabs = QTabWidget()
        self.cal_widget = CalibrationWidget()
        tabs.addTab(self.cal_widget, "Calibration")
        self.dose_widget = Tiff2DoseWidget()
        tabs.addTab(self.dose_widget, "Film2Dose")
        #self.tif_widget = TifWidget()
        #tabs.addTab(self.tif_widget, "Tif Image")
        self.tab_layout.addWidget(tabs)

        self.main_widget.setLayout(self.tab_layout)
        self.setCentralWidget(self.main_widget)

# Create the application
app = QApplication(sys.argv)
# Create the main window (view)
root_window = MainWindow()
# Create the model
dosepy_model = Model()

# Create the controllers
dosepy_calibration_controller = CalibrationController(model=dosepy_model, view=root_window)
dosepy_tiff2dose_controller = Tiff2DoseController(model=dosepy_model, view=root_window)

root_window.show()

sys.exit(app.exec())