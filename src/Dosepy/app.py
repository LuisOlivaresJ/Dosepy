import sys

from PySide6.QtWidgets import QApplication

# App Controllers
from Dosepy.app_controller.app_controller import (
    CalibrationController,
    Tiff2DoseController,
    ToolbarController,
)
# App Model
from Dosepy.app_model.app_model import Model

# App View
from Dosepy.app_components.main_window import MainWindow


# Create the application
app = QApplication(sys.argv)

# Create the main window (view)
root_window = MainWindow()
# Create the model
dosepy_model = Model()
# Create the controllers
dosepy_calibration_controller = CalibrationController(model=dosepy_model, view=root_window)
dosepy_tiff2dose_controller = Tiff2DoseController(model=dosepy_model, view=root_window)
dosepy_toolbar_controller = ToolbarController(model=dosepy_model, view=root_window)

root_window.show()

sys.exit(app.exec())