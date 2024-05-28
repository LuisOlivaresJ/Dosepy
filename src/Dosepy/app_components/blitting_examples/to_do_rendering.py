
#from matplotlib.backends.qt_compat import QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qtagg import \
	NavigationToolbar2QT as NavigationToolbar
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
import matplotlib.colors as colors
import numpy as np
#import pkg_resources
from importlib import resources


class BlittedCrossHair:
	"""
	A cross hair cursor using blitting for faster redraw. It works from Canvas' events.
	"""
	def __init__(self, ax):
		self.ax = ax
		self.background = None
		self.horizontal_line = ax.axhline(color = 'cornflowerblue', lw = 1.5, ls = '--', alpha = 0.8)
		self.vertical_line = ax.axvline(color = 'orange', lw = 1.5, ls = '--', alpha = 0.8)
		# text location in axes coordinates
		#self.text = ax.text(0.65, 0.9, '', transform=ax.transAxes)
		self._creating_background = False
		ax.figure.canvas.mpl_connect('draw_event', self._on_draw)

	def _on_draw(self, event):
		# Used when draw_event occurs.
		self._create_new_background()
		#print(event)

	def _set_cross_hair_visible(self, visible):
		need_redraw = self.horizontal_line.get_visible() != visible
		self.horizontal_line.set_visible(visible)
		self.vertical_line.set_visible(visible)
		#self.text.set_visible(visible)
		return need_redraw

	def _create_new_background(self):
		if self._creating_background:
			# discard calls triggered from within this function
			return
		self._creating_background = True
		self._set_cross_hair_visible(False)
		self.ax.figure.canvas.draw()
		self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
		self._set_cross_hair_visible(True)
		self._creating_background = False

	def on_event(self, event):
		#if event.inaxes != self.ax:
		#	return
		if self.background is None:
			self._create_new_background()
		if not event.inaxes:
			need_redraw = self._set_cross_hair_visible(False)
			if need_redraw:
				self.ax.figure.canvas.restore_region(self.background)
				self.ax.figure.canvas.blit(self.ax.bbox)
		else:
			if event.inaxes != self.ax:	#Descartar evento fuera de Axes que contiene la imagen
				return
			self._set_cross_hair_visible(True)
			# update the line positions
			x, y = event.xdata, event.ydata
			#dose = 

			self.horizontal_line.set_ydata([y])
			self.vertical_line.set_xdata([x])
			#self.text.set_text('x=%1.0f, y=%1.0f' % (x, y))
			
			self.ax.figure.canvas.restore_region(self.background)
			self.ax.draw_artist(self.horizontal_line)
			self.ax.draw_artist(self.vertical_line)
			#self.ax.draw_artist(self.text)
			self.ax.figure.canvas.blit(self.ax.bbox)


class QtImageWidget(QWidget):
	def __init__(self):
		super().__init__()  # Call QWidget constructor
		self.setWindowTitle('Dose distribution')

		#file_name_film = pkg_resources.resource_filename('Dosepy', 'data/D_FILM.csv')
		file_name_film = str(resources.files("Dosepy") / "data" / "D_FILM.csv")
		self.array_refer = np.genfromtxt(file_name_film, delimiter = ',')
        
		self.main_layout = QVBoxLayout()
		self._iniciarUI()
        
	def _iniciarUI(self):
		
		self.mpl_left = Canvas()
		self.mpl_left.show_dist(self.array_refer)

		self.main_layout.addWidget(NavigationToolbar(self.mpl_left.canvas, self))
		self.main_layout.addWidget(self.mpl_left.canvas)
		
		self.setLayout(self.main_layout)


		self.blitted_cross_hair = BlittedCrossHair(self.mpl_left.ax)
		self.id_on_press_perfil = self.mpl_left.canvas.figure.canvas.mpl_connect('motion_notify_event', self.blitted_cross_hair.on_event)
		#self.id_on_press_profile = self.mpl_left.canvas.figure.canvas.mpl_connect('button_press_event', self.blitted_cross_hair.on_event)

            
class Canvas:
	"""
	Canvas to hold the image.
	"""

	def __init__(self):
		fig = Figure(
			layout='constrained')
		
		self.canvas = FigureCanvas(fig)
		self.ax = fig.add_subplot()

		self.cmap='nipy_spectral'

		self.mplI = None  # matplotlib.image.AxesImage
		self.cbar = None  # matplotlib.colorbar.Colorbar
		
	def show_dist(self, array):
		''' Show the dose distribution.

		Parameters
		----------
		array : numpy.ndarray
            The image represented as a numpy array.
		'''

		self.mplI = self.ax.imshow(array)
		self._setup_colorbar(array)


	def _setup_colorbar(self, array):
		''' 
		Set a colorbar. 
		
		Parameters
		----------
		array : numpy.ndarray
		'''

		bounds = np.linspace(0, round(1.15 * np.percentile(array, 98)), 256)
		norm = colors.BoundaryNorm(boundaries = bounds, ncolors = 256)
		self.mplI.set_norm(norm)
		self.mplI.set_cmap(self.cmap)

		self.cbar = self.canvas.figure.colorbar(
			self.mplI,
			ax = self.ax,
			orientation = 'vertical',
			format = '%.1f',
			shrink = 0.8,
			)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ventana_raiz = QtImageWidget()
    ventana_raiz.setGeometry(100, 150, 500, 350)
    ventana_raiz.show()

    sys.exit(app.exec())
