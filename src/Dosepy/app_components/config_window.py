# Import necessary libraries
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QApplication
from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QIntValidator
import sys

from Dosepy.config.io_settings import load_settings

# Qwidget to change configuration settings

class ConfigWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        settings = load_settings()

        # Set the window title
        self.setWindowTitle("Configuration settings")

        self.layout = QVBoxLayout()

        self.roi_size_h_label = QLabel(f"ROI size horizontal (mm): {settings.get_calib_roi_size()[0]}")
        self.roi_size_h = QLineEdit()
        self.roi_size_h.setValidator(QIntValidator(bottom=1))
        self.roi_size_h.setFixedWidth(50)

        self.roi_size_v_label = QLabel(f"ROI size vertical (mm): {settings.get_calib_roi_size()[1]}")
        self.roi_size_v = QLineEdit()
        self.roi_size_v.setValidator(QIntValidator(bottom=1))
        self.roi_size_v.setFixedWidth(50)

        # Button to save the settings
        self.save_button = QPushButton("Save")

        self.layout.addWidget(self.roi_size_h_label)
        self.layout.addWidget(self.roi_size_h)
        self.layout.addWidget(self.roi_size_v_label)
        self.layout.addWidget(self.roi_size_v)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)

# Used for testing the ConfigWindow
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigWindow()
    window.show()
    sys.exit(app.exec_())
