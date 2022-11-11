
from matplotlib.backends.qt_compat import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
import sys
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib import patches
import matplotlib.colors as colors
import numpy as np


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
		self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)
		self._creating_background = False
		ax.figure.canvas.mpl_connect('draw_event', self.on_draw)

	def on_draw(self, event):
		self.create_new_background()
		print(event)

	def set_cross_hair_visible(self, visible):
		need_redraw = self.horizontal_line.get_visible() != visible
		self.horizontal_line.set_visible(visible)
		self.vertical_line.set_visible(visible)
		self.text.set_visible(visible)
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
		#if event is None or event.inaxes != self.Mpl_Izq.ax1:
		if self.background is None:
			self.create_new_background()
		if not event.inaxes:
			need_redraw = self.set_cross_hair_visible(False)
			if need_redraw:
				self.ax.figure.canvas.restore_region(self.background)
				self.ax.figure.canvas.blit(self.ax.bbox)
		else:
			self.set_cross_hair_visible(True)
			# update the line positions
			x, y = event.xdata, event.ydata

			self.horizontal_line.set_ydata(y)
			self.vertical_line.set_xdata(x)
			self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))

			self.ax.figure.canvas.restore_region(self.background)
			self.ax.draw_artist(self.horizontal_line)
			self.ax.draw_artist(self.vertical_line)
			self.ax.draw_artist(self.text)
			self.ax.figure.canvas.blit(self.ax.bbox)


class Bloque_Imagenes(QWidget):
	def __init__(self):
		super().__init__()  #   Llamar al constructor de QWidget
		self.setStyleSheet("background-color: #1d1040;")
		self.setWindowTitle('Acerca de')

		file_name_FILM = "../data/D_FILM.csv"
		self.array_refer = np.genfromtxt(file_name_FILM, delimiter = ',')
        
		self.main_label = QVBoxLayout()
		self.iniciarUI()
        
	def iniciarUI(self):
		
		self.Mpl_Izq = Qt_Figure_Imagen()
		self.Mpl_Izq.Img(self.array_refer)
		self.Mpl_Izq.Colores(self.array_refer) 
		self.blitted_cross_hair = BlittedCursorCrossHair(self.Mpl_Izq.ax1)
		self.id_on_press_perfil = self.Mpl_Izq.Qt_fig.figure.canvas.mpl_connect('motion_notify_event', self.blitted_cross_hair.on_mouse_move)
		self.main_label.addWidget(self.Mpl_Izq.Qt_fig)
		self.setLayout(self.main_label)

"""
#------------------Funciones----------------------------
	def on_move_img_ref(self, event):
		print(event)

		if event is None or event.inaxes != self.Mpl_Izq.ax1:
			return

		dx = event.xdata
		dy = event.ydata

		self.Mpl_Izq.circ.center = dx, dy
		self.Mpl_Izq.hline.set_ydata( int(dy) )
		self.Mpl_Izq.vline.set_xdata( int(dx) )

		#self.Mpl_Izq.hline.figure.canvas.draw()		
		#self.Mpl_Izq.vline.figure.canvas.draw()
		#self.Mpl_Izq.circ.figure.canvas.draw()

		self.Mpl_Izq.Qt_fig.figure.canvas.draw()
"""		
            
class Qt_Figure_Imagen:
	"""
	Clase para contener la distribución de dosis
	"""

	def __init__(self):
		self.fig = Figure(figsize=(3.8,3), facecolor = 'whitesmoke')
		self.Qt_fig = FigureCanvas(self.fig)
		
		#   Axes para la imagen
		#tuple (left, bottom, width, height)
		#The dimensions (left, bottom, width, height) of the new Axes. All quantities are in fractions of figure width and height.

		self.ax1 = self.fig.add_axes([0.08, 0.08, 0.75, 0.85])
		self.ax2 = self.fig.add_axes([0.85, 0.15, 0.04, 0.72])
		
	def Img(self, np_I):
		'''
		Definir la imagen a partir de un array que se proporciona como argumento.
		'''

		self.npI = np_I
		self.mplI = self.ax1.imshow(self.npI)
		print(self.mplI)
		#print(type(self.mplI))
		#print(dir(self.mplI))
		#print(id(self.mplI))

	def Colores(self, npI_color_ref):
		'''
		Definir el mapa de colores a utiliar. Como argumento se requiere de una imagen de referencia para elegui mapa.
		Por lo general se utiliza la distribución calculada por el TPS.
		'''
		color_map = 'viridis'
		bounds = np.linspace(0, round(1.15 * np.percentile(npI_color_ref, 98)), 256)
		norm = colors.BoundaryNorm(boundaries = bounds, ncolors = 256)
		self.mplI.set_norm(norm)
		self.mplI.set_cmap(color_map)
		self.cbar = self.fig.colorbar(self.mplI, cax = self.ax2, orientation = 'vertical', shrink = 0.6, format = '%.1f')




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana_raiz = Bloque_Imagenes()
    ventana_raiz.setGeometry(100, 150, 500, 350)
    ventana_raiz.show()
    sys.exit(app.exec_())
