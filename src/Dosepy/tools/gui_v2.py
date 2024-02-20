import sys

from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QLabel,
    QTabWidget,
)

from gui_widgets.calibration_gui import CalibrationWidget

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Film dosimetry")

        tab_layout = QVBoxLayout()
        self.setLayout(tab_layout)

        tabs = QTabWidget()
        tabs.addTab(CalibrationWidget(), "Calibration")
        tabs.addTab(QLabel("In progress..."), "Film2Dose")
        tabs.addTab(QLabel("In progress..."), "Analysis")
        tab_layout.addWidget(tabs)

if __name__ == "__main__":

    app = QApplication(sys.argv)

    root_window = MainWindow()
    root_window.show()

    app.exec()

