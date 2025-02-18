from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QMainWindow,
)

# Import app views
from Dosepy.app_components.calibration_widget import CalibrationWidget
from Dosepy.app_components.tiff2dose_widget import Tiff2DoseWidget
from Dosepy.app_components.tif_widget import TifWidget
from Dosepy.app_components.config_window import ConfigWindow
from Dosepy.app_components.ct_viewer import CTViewer

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()

        self.setWindowTitle("Dosepy")

        self.main_widget = QWidget()

        self._create_actions()
        self._create_toolbar()
        self._create_body()

        self.setCentralWidget(self.main_widget)


    def _create_actions(self):
        self.calib_setings_action = QAction("Settings", self)
        self.ct_viewer_action = QAction("CT Viewer", self)


    def _create_toolbar(self):
        toolbar = self.addToolBar("Settings")
        toolbar.addAction(self.calib_setings_action)
        #toolbar_ct_viewer = self.addToolBar("CT viewer")
        #toolbar_ct_viewer.addAction(self.ct_viewer_action)
        toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.conf_window = ConfigWindow()
        self.ct_viewer = CTViewer()


    def _create_body(self):
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