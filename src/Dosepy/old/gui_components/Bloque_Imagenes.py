#%%
#   Para mostrar distribuciones de dosis y perfiles


#%%
#---------------------------------------------

#   Importaciones

#---------------------------------------------

from matplotlib import patches
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
import sys
from importlib import resources
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
import matplotlib.colors as colors
import numpy as np
#from PyQt5.QtGui import QIcon
import Dosepy.old.dose as dp



#%%
#########################################################
#---------------------------------------------

#   Cuerpo de la app

#---------------------------------------------

class Bloque_Imagenes(QWidget):
    def __init__(self):
        super().__init__()  #   Llamar al constructor de QWidget
        self.iniciarUI()

    def iniciarUI(self):
        self.IniciarWidgets()


    def IniciarWidgets(self):

#%%
        #   Widget para imagen de referencia

        file_name_FILM = str(resources.files("Dosepy") / "data" / "D_FILM.csv")
        file_name_TPS = str(resources.files("Dosepy") / "data" / "D_TPS.csv")

        self.D_ref = dp.from_csv(file_name_FILM, 1)
        self.D_eval = dp.from_csv(file_name_TPS, 1)

        self.Mpl_Izq = Qt_Figure_Imagen()
        self.Mpl_Der = Qt_Figure_Imagen()

        self.Mpl_Izq.ax1.set_title('Referencia', fontsize = 10, loc = 'left')
        self.Mpl_Der.ax1.set_title('A evaluar', fontsize = 10, loc = 'left')

        self.Mpl_Izq.Img(self.D_ref)
        self.Mpl_Der.Img(self.D_eval)

        self.Mpl_Izq.Colores(self.D_ref.array)
        self.Mpl_Der.Colores(self.D_eval.array)

        self.Mpl_Izq.hline.set_linestyle('-')
        self.Mpl_Izq.vline.set_linestyle('-')

        self.Mpl_Der.circ.set_visible(False)

        #self.Mpl_Izq.Cross_Hair_off()
        #self.Mpl_Der.Cross_Hair_off()

        self.id_on_press_perfil = self.Mpl_Izq.Qt_fig.figure.canvas.mpl_connect('button_press_event', self.on_press_img_ref)            # Controlar si hay eventos y acciones
        self.id_on_release_perfil = self.Mpl_Izq.Qt_fig.figure.canvas.mpl_connect('button_release_event', self.on_release_img_ref)
        self.id_on_move_perfil = self.Mpl_Izq.Qt_fig.figure.canvas.mpl_connect('motion_notify_event', self.on_move_img_ref)
        self.pressevent = None

        #   Widget para los perfiles
        self.Mpl_perfiles = Qt_Figure_Perfiles()
        self.Mpl_perfiles.set_data_and_plot(self.Mpl_Izq.npI, self.Mpl_Der.npI, self.Mpl_Izq.circ)

        #   Widgets para los botones
        self.boton_roi = QPushButton('ROI')
        #self.boton_roi.resize(150,50)
        self.boton_roi.setCheckable(True)
        self.boton_roi.setChecked(False)
        self.boton_roi.clicked.connect(self.clic_ROI)

        #corte_icon = QIcon("../Icon/cut_icon.png")
        #corte_name_folder = pkg_resources.resource_filename('Dosepy', 'Icon/cut_icon.png')
        self.boton_recortar_Izq = QPushButton('Corte')
        #self.boton_recortar_Izq.setIcon(corte_icon)
        self.boton_recortar_Izq.setEnabled(False)

        self.compare_button = QPushButton("H. Profile")

        #self.boton_exportar_Izq = QPushButton('Exportar')
        self.boton_exportar_Der = QPushButton('')
        #self.boton_exportar_perfiles = QPushButton('Exportar')

        #   Contenedores
        """
        layout_hijo_Izq = QHBoxLayout()
        layout_hijo_Izq.addWidget(self.boton_roi)
        layout_hijo_Izq.addWidget(self.boton_recortar_Izq)
        #layout_hijo_Izq.addWidget(self.boton_exportar_Izq)
        layout_hijo_Izq.addStretch()

        layout_hijo_Der = QHBoxLayout()
        layout_hijo_Der.addWidget(self.boton_exportar_Der)
        layout_hijo_Der.addStretch()

        layout_hijo_perfiles = QHBoxLayout()
        #layout_hijo_perfiles.addWidget(self.boton_exportar_perfiles)
        layout_hijo_perfiles.addStretch()
        """
        layout_padre_botones = QVBoxLayout()
        layout_padre_botones.addWidget(self.boton_roi)
        layout_padre_botones.addWidget(self.boton_recortar_Izq)
        layout_padre_botones.addWidget(self.compare_button)

        layout_padre_botones.addStretch()

        layout_padre_Izq = QVBoxLayout()
        #layout_padre_Izq.addLayout(layout_hijo_Izq)
        layout_padre_Izq.addWidget(self.Mpl_Izq.Qt_fig)

        layout_padre_Der = QVBoxLayout()
        #layout_padre_Der.addLayout(layout_hijo_Der)
        layout_padre_Der.addWidget(self.Mpl_Der.Qt_fig)

        layout_padre_perfiles = QVBoxLayout()
        #layout_padre_perfiles.addLayout(layout_hijo_perfiles)
        layout_padre_perfiles.addWidget(self.Mpl_perfiles.Qt_fig)

        layout_abuelo = QHBoxLayout()
        layout_abuelo.addLayout(layout_padre_botones)
        layout_abuelo.addLayout(layout_padre_Izq)
        layout_abuelo.addLayout(layout_padre_Der)
        layout_abuelo.addLayout(layout_padre_perfiles)

        self.setLayout(layout_abuelo)


###############################################################################################
#---------------------------------------------

        #   Funciones de la app

    def on_press_img_ref(self, event):

        if self.boton_roi.isChecked():
            if event.inaxes != self.Mpl_Izq.ax1:
                return
            #   Codigo para generar ROI rectangular
            self.Mpl_Izq.ROI_Rect_set_up(event)
            self.pressevent = event


        else:
            if event.inaxes != self.Mpl_Izq.ax1:    #   ¿El axes donde se creó el evento es diferente que el axes de la imagen de referencia?
                return

            if not self.Mpl_Izq.circ.contains(event)[0]:    #   ¿El evento se creó fuera del patch?
                return

            self.pressevent = event

    def on_release_img_ref(self, event):
        self.pressevent = None

        if self.boton_roi.isChecked():
            return


        else:
            self.Mpl_Izq.x0, self.Mpl_Izq.y0 = self.Mpl_Izq.circ.center
            self.Mpl_Der.x0, self.Mpl_Der.y0 = self.Mpl_Der.circ.center


    def on_move_img_ref(self, event):

        if self.boton_roi.isChecked():
            #   Código para generar ROI rectangular

            if self.pressevent is None or event.inaxes != self.pressevent.inaxes:
                return

            dx = abs(event.xdata - self.pressevent.xdata)
            dy = abs(event.ydata - self.pressevent.ydata)
            x_i = min(event.xdata, self.pressevent.xdata)
            y_i = min(event.ydata, self.pressevent.ydata)
            self.Mpl_Izq.Rectangle.set_bounds(x_i, y_i, dx, dy)
            self.boton_recortar_Izq.setEnabled(True)
            #print(self.Mpl_Izq.Rectangle.properties())
            self.Mpl_Izq.fig.canvas.draw()


        else:
            #   Código para generar cross hair y perfiles

            if self.pressevent is None or event.inaxes != self.pressevent.inaxes:
                return

            dx = event.xdata - self.pressevent.xdata
            dy = event.ydata - self.pressevent.ydata
            self.Mpl_Izq.circ.center = self.Mpl_Izq.x0 + dx, self.Mpl_Izq.y0 + dy
            self.Mpl_Der.circ.center = self.Mpl_Der.x0 + dx, self.Mpl_Der.y0 + dy

            self.Mpl_Izq.Cross_Hair_set_up()
            self.Mpl_Der.Cross_Hair_set_up()

            self.Mpl_perfiles.set_data_and_plot(self.Mpl_Izq.npI, self.Mpl_Der.npI, self.Mpl_Izq.circ)

            self.Mpl_Izq.fig.canvas.draw()
            self.Mpl_Der.fig.canvas.draw()


 #%%

    def clic_ROI(self):

        if self.boton_roi.isChecked():
            #   Código para el ROI
            self.Mpl_Izq.Cross_Hair_off()
            self.Mpl_Der.Cross_Hair_off()


        else :
            #   Código para el cross

            self.Mpl_Izq.ROI_Rect_off()
            self.Mpl_Izq.Cross_Hair_on()
            self.Mpl_Der.Cross_Hair_on()

            self.boton_recortar_Izq.setEnabled(False)


#%%
class Qt_Figure_Imagen:
    """
    Clase para contener la distribución de dosis
    """

    def __init__(self):
        self.fig = Figure(figsize=(3.8,3), facecolor = 'whitesmoke')
        self.Qt_fig = FigureCanvas(self.fig)

        self.ax1 = self.fig.add_axes([0.08, 0.08, 0.75, 0.85])
        self.ax2 = self.fig.add_axes([0.85, 0.15, 0.04, 0.72])

        #   Definición y manipulación de patches
        self.x0 = 0.5
        self.y0 = 0.5
        self.circ = patches.Circle((self.x0, self.y0), 5, alpha = 0.3, fc = 'yellow') #Parche para ciruclo del crosshair
        #self.circ.set_fill(False)
        self.circ.set_linestyle('--')
        self.circ.set_linewidth(3)

        self.hline = self.ax1.axhline(self.circ.center[1], color = 'cornflowerblue', lw = 1.5, ls = '--', alpha = 0.8)
        self.vline = self.ax1.axvline(self.circ.center[0], color = 'orange', lw = 1.5, ls = '--', alpha = 0.8)

        self.x0_rec = 0
        self.y0_rec = 0
        self.Rec_width = 0
        self.Rec_height = 0
        self.Rectangle = patches.Rectangle((self.x0, self.y0) , self.Rec_width, self.Rec_height)
        self.Rectangle.set_fill(False)
        self.Rectangle.set_linestyle('--')
        self.Rectangle.set_linewidth(2)
        self.ax1.add_patch(self.Rectangle)



    def Img(self, D):
        '''
        Definir la imagen a partir de un objeto de la clase Dosepy.Dose.
        '''

        self.npI = D.array
        self.mplI = self.ax1.imshow(self.npI)

        self.x0 = int(self.npI.shape[1]/2)
        self.y0 = int(self.npI.shape[0]/2)
        self.circ.center  = self.x0, self.y0
        self.circ.set_radius( 0.04 *  np.min(self.npI.shape))

        self.hline.set_ydata( int(self.circ.center[1]) )
        self.vline.set_xdata( int(self.circ.center[0]) )
        self.ax1.add_patch(self.circ)

        self.Cross_Hair_set_up()



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


    def ROI_Rect_set_up(self, event):
        '''
        Definición del ROI rectangular.
        '''
        self.x0_rec = event.xdata
        self.y0_rec = event.ydata
        self.Rectangle.set_visible(True)

    def ROI_Rect_on(self):
        self.Rectangle.set_visible(True)
        self.fig.canvas.draw()

    def ROI_Rect_off(self):
        self.Rectangle.set_visible(False)
        self.fig.canvas.draw()





    def Cross_Hair_set_up(self):
        '''
        Definición del Cross Hair.
        '''

        self.hline.set_ydata( int(self.circ.center[1]) )
        self.vline.set_xdata( int(self.circ.center[0]) )

    def Cross_Hair_on(self):
        '''
        Enciende el Cross Hair.
        '''
        self.hline.set_visible(True)
        self.vline.set_visible(True)
        self.circ.set_visible(True)
        self.fig.canvas.draw()

    def Cross_Hair_off(self):
        '''
        Enciende el Cross Hair.
        '''
        self.hline.set_visible(False)
        self.vline.set_visible(False)
        self.circ.set_visible(False)
        self.fig.canvas.draw()






class Qt_Figure_Perfiles:
    """
    Clase para contener los perfiles de las distribuciones y graficarlos
    """

    def __init__(self):
        self.fig = Figure(figsize=(4,3), tight_layout = True, facecolor = 'whitesmoke')
        self.Qt_fig = FigureCanvas(self.fig)

        #   Axes para la imagen
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_title('Perfiles')
        self.ax.set_ylabel('Dosis [Gy]')
        self.ax.grid(alpha = 0.3)


    def set_data_and_plot(self, D_ref, D_eval, circ):
        self.perfil_horizontal_ref = D_ref[int(circ.center[1]), :]
        self.perfil_horizontal_eval = D_eval[int(circ.center[1]), :]
        self.perfil_vertical_ref = D_ref[:, int(circ.center[0])]
        self.perfil_vertical_eval = D_eval[:, int(circ.center[0])]
        self.ax.clear()
        self.ax.plot(self.perfil_horizontal_ref, color = 'cornflowerblue')
        self.ax.plot(self.perfil_horizontal_eval, color = 'cornflowerblue', ls = '--')
        self.ax.plot(self.perfil_vertical_ref, color = 'orange')
        self.ax.plot(self.perfil_vertical_eval, color = 'orange', ls = '--')
        self.ax.set_ylabel('Dosis [Gy]')
        self.ax.set_xlabel('Píxel')
        self.ax.grid(alpha = 0.3)

        self.fig.canvas.draw()




#%%

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana_raiz = Bloque_Imagenes()
    ventana_raiz.setGeometry(100, 150, 1200, 300)
    ventana_raiz.show()
    sys.exit(app.exec())
