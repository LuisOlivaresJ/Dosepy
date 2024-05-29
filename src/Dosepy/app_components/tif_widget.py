from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure

from .styles.styles import Size

class TifWidget(QWidget):
	def __init__(self):
		super().__init__()

		self.main_layout = QVBoxLayout()
		self._buildUI()

	def _buildUI(self):

		self.open_button = QPushButton(text="Browse")
		self.rotate_button = QPushButton(text="Rotate")

		self.canvas_tif = Canvas_Tif()

		buttons_layout = QHBoxLayout()
		buttons_layout.addWidget(self.rotate_button)
		
		self.main_layout.addLayout(buttons_layout)
		self.main_layout.addWidget(self.canvas_tif.canvas)

		self.setLayout(self.main_layout)


class Canvas_Tif:
	"""
	Canvas to hold the image.
	"""

	def __init__(self):
		fig = Figure(
			layout='constrained')
		
		self.canvas = FigureCanvas(fig)
		self.ax = fig.add_subplot()

		self.mplI = None  # matplotlib.image.AxesImage
		

	def show_tif(self, tif):
		''' Show the tif file.

		Parameters
		----------
		tif : TiffImage
            The image represented as a tif.
		'''

		self.mplI = self.ax.imshow(tif.array)