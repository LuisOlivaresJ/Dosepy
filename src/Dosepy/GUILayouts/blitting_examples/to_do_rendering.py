
#from matplotlib.backends.qt_compat import QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qtagg import \
	NavigationToolbar2QT as NavigationToolbar
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
import matplotlib.colors as colors
import numpy as np
import pkg_resources


class BlittedCursorCrossHair:
	"""
	A cross hair cursor using blitting for faster redraw.
	"""
	def __init__(self, ax):
		self.ax = ax
		self.background = None
		self.horizontal_line = ax.axhline(color = 'cornflowerblue', lw = 1.5, ls = '--', alpha = 0.8)
		self.vertical_line = ax.axvline(color = 'orange', lw = 1.5, ls = '--', alpha = 0.8)
		# text location in axes coordinates
		#self.text = ax.text(0.65, 0.9, '', transform=ax.transAxes)
		self._creating_background = False
		ax.figure.canvas.mpl_connect('draw_event', self.on_draw)

	def on_draw(self, event):
		#De utilidad si hay un cambio en el tamaño de la ventana
		self.create_new_background()
		print(event)

	def set_cross_hair_visible(self, visible):
		need_redraw = self.horizontal_line.get_visible() != visible
		self.horizontal_line.set_visible(visible)
		self.vertical_line.set_visible(visible)
		#self.text.set_visible(visible)
		return need_redraw

	def create_new_background(self):
		if self._creating_background:
			# discard calls triggered from within this function
			return
		self._creating_background = True
		self.set_cross_hair_visible(False)
		self.ax.figure.canvas.draw()
		self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
		self.set_cross_hair_visible(True)
		self._creating_background = False

	def on_mouse_move(self, event):
		#if event.inaxes != self.ax:
		#	return
		if self.background is None:
			self.create_new_background()
		if not event.inaxes:
			need_redraw = self.set_cross_hair_visible(False)
			if need_redraw:
				self.ax.figure.canvas.restore_region(self.background)
				self.ax.figure.canvas.blit(self.ax.bbox)
		else:
			if event.inaxes != self.ax:	#Descartar evento fuera de Axes que contiene la imagen
				return
			self.set_cross_hair_visible(True)
			# update the line positions
			x, y = event.xdata, event.ydata
			#dose = 

			self.horizontal_line.set_ydata(y)
			self.vertical_line.set_xdata(x)
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

		file_name_FILM = pkg_resources.resource_filename('Dosepy', 'data/D_FILM.csv')
		self.array_refer = np.genfromtxt(file_name_FILM, delimiter = ',')
        
		self.main_layout = QVBoxLayout()
		self.iniciarUI()
        
	def iniciarUI(self):
		
		self.Mpl_Izq = Canvas()
		self.Mpl_Izq.show_dist(self.array_refer)
		self.Mpl_Izq.set_colorbar(self.array_refer) 
		self.blitted_cross_hair = BlittedCursorCrossHair(self.Mpl_Izq.ax)
		#self.id_on_press_perfil = self.Mpl_Izq.canvas.figure.canvas.mpl_connect('motion_notify_event', self.blitted_cross_hair.on_mouse_move)
		self.id_on_press_perfil = self.Mpl_Izq.canvas.figure.canvas.mpl_connect('button_press_event', self.blitted_cross_hair.on_mouse_move)
		self.main_layout.addWidget(NavigationToolbar(self.Mpl_Izq.canvas, self))
		self.main_layout.addWidget(self.Mpl_Izq.canvas)
		self.setLayout(self.main_layout)
            
class Canvas:
	"""Widget to show a dose distribution.
	"""

	def __init__(self):
		fig = Figure(
			layout='constrained')
		self.ax = fig.add_subplot()
		self.canvas = FigureCanvas(fig)
		
	def show_dist(self, array):
		''' Show the dose distribution.

		Parameters
		----------
		array : numpy.ndarray
            The image represented as a numpy array.
		'''
		self.npI = array
		self.mplI = self.ax.imshow(self.npI)

	def set_colorbar(self, npI_color_ref):
		''' 
		Set a colorbar. 
		
		Parameters
		----------
		array : numpy.ndarray
			An array used as a reference to define dose boundaries.
		'''
		color_map = 'nipy_spectral'
		bounds = np.linspace(0, round(1.15 * np.percentile(npI_color_ref, 98)), 256)
		norm = colors.BoundaryNorm(boundaries = bounds, ncolors = 256)
		self.mplI.set_norm(norm)
		self.mplI.set_cmap(color_map)

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
