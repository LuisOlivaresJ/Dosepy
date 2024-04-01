import sys

from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QLabel,
    QTabWidget,
)

# Import app views
from .app_components.calibration_widget import CalibrationWidget
from .app_components.tiff2dose_widget import Tiff2DoseWidget
# Import app controllers
from .app_controller import CalibrationController, Tiff2DoseController
# Import app model
from .app_model import Model

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dosepy")

        tab_layout = QVBoxLayout()
        self.setLayout(tab_layout)

        tabs = QTabWidget()
        self.cal_widget = CalibrationWidget()
        tabs.addTab(self.cal_widget, "Calibration")
        self.tif_widget = Tiff2DoseWidget()
        tabs.addTab(self.tif_widget, "Film2Dose")
        #tabs.addTab(QLabel("In progress..."), "Analysis")
        tab_layout.addWidget(tabs)

'''
if __name__ == "__main__":

    app = QApplication(sys.argv)

    root_window = MainWindow()
    dosepy_model = Model()
    dosepy_controller = DosepyController(model=dosepy_model, view=root_window)
    root_window.show()

    sys.exit(app.exec())
'''

app = QApplication(sys.argv)

root_window = MainWindow()
dosepy_model = Model()
dosepy_calibration_controller = CalibrationController(model=dosepy_model, view=root_window)
dosepy_tiff2dose_controller = Tiff2DoseController(model=dosepy_model, view=root_window)
root_window.show()

sys.exit(app.exec())