# Import necessary libraries
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QApplication
from PySide6.QtWidgets import QPushButton, QComboBox, QGroupBox
from PySide6.QtGui import QIntValidator
import sys

from Dosepy.config.io_settings import Settings
from .styles.styles import Size

# Qwidget to change configuration settings

class ConfigWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.setWindowTitle("Configuration settings")

        # Main layout
        self.main_layout = QVBoxLayout()


    def _create_calib_box_group(self, settings: Settings):
        self.calib_box_group = QGroupBox("Calibration settings")
        layout = QVBoxLayout()

        roi_automatic_label = QLabel(f"Automatic ROI size")
        self.roi_automatic_cbox = QComboBox()
        self.roi_automatic_cbox.addItems(["Enable", "Disable"])
        auto_roi = settings.get_roi_automatic()
        auto_roi_indices = {"Enable": 0, "Disable": 1}
        self.roi_automatic_cbox.setCurrentIndex(auto_roi_indices[auto_roi])

        roi_size_h_label = QLabel(f"ROI size horizontal (mm):")
        self.roi_size_h = QLineEdit()
        self.roi_size_h.setText(str(settings.get_calib_roi_size()[0]))
        self.roi_size_h.setValidator(QIntValidator(bottom=1))
        self.roi_size_h.setFixedWidth(50)

        roi_size_v_label = QLabel(f"ROI size vertical (mm):")
        self.roi_size_v = QLineEdit()
        self.roi_size_v.setText(str(settings.get_calib_roi_size()[1]))
        self.roi_size_v.setValidator(QIntValidator(bottom=1))
        self.roi_size_v.setFixedWidth(50)

        self.channel_label = QLabel(f"Channel:")
        self.channel = QComboBox()
        self.channel.addItems(["Red", "Green", "Blue"])
        channel = settings.get_channel()
        channel_indices = {"Red": 0, "Green": 1, "Blue": 2}
        self.channel.setCurrentIndex(channel_indices[channel])


        fit_label = QLabel(f"Fit function:")
        self.fit_function = QComboBox()
        self.fit_function.addItems(["Rational", "Polynomial"])
        fit_function = settings.get_fit_function()
        fit_funciton_indices = {"Rational": 0, "Polynomial": 1}
        self.fit_function.setCurrentIndex(fit_funciton_indices[fit_function])

        layout.addWidget(roi_automatic_label)
        layout.addWidget(self.roi_automatic_cbox)
        layout.addWidget(roi_size_h_label)
        layout.addWidget(self.roi_size_h)
        layout.addWidget(roi_size_v_label)
        layout.addWidget(self.roi_size_v)
        layout.addWidget(self.channel_label)
        layout.addWidget(self.channel)
        layout.addWidget(fit_label)
        layout.addWidget(self.fit_function)

        self.calib_box_group.setLayout(layout)

        self._add_and_set_main_layout()


    def _add_and_set_main_layout(self):
        self.main_layout.addWidget(self.calib_box_group)
        self.setLayout(self.main_layout)


# Used for testing the ConfigWindow
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigWindow()
    window.show()
    sys.exit(app.exec_())
