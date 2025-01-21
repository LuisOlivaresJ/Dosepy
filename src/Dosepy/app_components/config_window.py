# Import necessary libraries
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QApplication
from PySide6.QtWidgets import QPushButton, QComboBox, QGroupBox
from PySide6.QtGui import QIntValidator
import sys

from Dosepy.config.io_settings import load_settings
from .styles.styles import Size

# Qwidget to change configuration settings

class ConfigWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        settings = load_settings()

        # Set the window title
        self.setWindowTitle("Configuration settings")

        # Box group layout for calibration settings
        self._create_calib_box_group(settings)

        # Button to save the settings
        self.save_button = QPushButton("Save")
        self.save_button.setMinimumSize(Size.MAIN_BUTTON.value)

        # Main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.calib_box_group)
        self.main_layout.addWidget(self.save_button)

        self.setLayout(self.main_layout)


    def _create_calib_box_group(self, settings):
        self.calib_box_group = QGroupBox("Calibration settings")
        layout = QVBoxLayout()

        self.roi_size_h_label = QLabel(f"ROI size horizontal (mm): {settings.get_calib_roi_size()[0]}")
        self.roi_size_h = QLineEdit()
        self.roi_size_h.setValidator(QIntValidator(bottom=1))
        self.roi_size_h.setFixedWidth(50)

        self.roi_size_v_label = QLabel(f"ROI size vertical (mm): {settings.get_calib_roi_size()[1]}")
        self.roi_size_v = QLineEdit()
        self.roi_size_v.setValidator(QIntValidator(bottom=1))
        self.roi_size_v.setFixedWidth(50)

        self.channel_label = QLabel(f"Channel: {settings.get_channel()}")
        self.channel = QComboBox()
        self.channel.addItems(["Red", "Green", "Blue"])

        self.fit_label = QLabel(f"Fit function: {settings.get_fit_function()}")
        self.fit_function = QComboBox()
        self.fit_function.addItems(["Rational", "Polynomial"])

        layout.addWidget(self.roi_size_h_label)
        layout.addWidget(self.roi_size_h)
        layout.addWidget(self.roi_size_v_label)
        layout.addWidget(self.roi_size_v)
        layout.addWidget(self.channel_label)
        layout.addWidget(self.channel)
        layout.addWidget(self.fit_label)
        layout.addWidget(self.fit_function)

        self.calib_box_group.setLayout(layout)

# Used for testing the ConfigWindow
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigWindow()
    window.show()
    sys.exit(app.exec_())
