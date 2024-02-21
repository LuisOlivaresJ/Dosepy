import sys

from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QLabel,
    QTabWidget,
)

from gui_widgets.calibration_gui import CalibrationWidget
from controller import DosepyController
from model import Model

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Film dosimetry")

        tab_layout = QVBoxLayout()
        self.setLayout(tab_layout)

        tabs = QTabWidget()
        self.calibration_widget = CalibrationWidget()
        tabs.addTab(self.calibration_widget, "Calibration")
        #self.tab_calibration = tabs.indexOf(CalibrationWidget)
        tabs.addTab(QLabel("In progress..."), "Film2Dose")
        tabs.addTab(QLabel("In progress..."), "Analysis")
        tab_layout.addWidget(tabs)

if __name__ == "__main__":

    app = QApplication(sys.argv)

    root_window = MainWindow()
    dosepy_model = Model()
    dosepy_controller = DosepyController(model=dosepy_model, view=root_window)
    root_window.show()

    sys.exit(app.exec())

