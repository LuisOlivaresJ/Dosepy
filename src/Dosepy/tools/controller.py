"""Class used as a controller in a MVC pattern."""

class DosepyController():
    def __init__(self, model, view):
        self._model = model
        self._view = view
        self._connectSignalsAndSlots()


    def _connectSignalsAndSlots(self):
        self._view.calibrationWidget.open_button.clicked.connect(self._open_button)

    def _open_button(self):
        print("Hola boton open")