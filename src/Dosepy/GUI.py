# -*- coding: utf-8 -*-
"""

Última modificación: 30 Octubre 2021
@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico en Radioterapia, La Paz, Baja California Sur, México.

    Derechos Reservados (c) Luis Alfonso Olivares Jimenez 2021
"""
#---------------------------------------------
#   Importaciones

import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QHBoxLayout, QMessageBox, QMainWindow, QAction, QLabel
from PyQt5.QtGui import QIcon
import numpy as np
from Dosepy.GUILayouts.MostrarLabels_Import import MostrarLabels
#from GUILayouts.MostrarLabels_Import import MostrarLabels  # Se importa desde archivo en PC para testear
from Dosepy.GUILayouts.Imagen import Bloque_Inf_Imagenes
#from GUILayouts.Imagen import Bloque_Inf_Imagenes   # Se importa desde archivo en PC para testear
import Dosepy.dose as dp
import matplotlib as mpl
import pkg_resources

from Dosepy.GUILayouts.film_to_doseGUI import Film_to_Dose_Window
#from GUILayouts.film_to_doseGUI import Film_to_Dose_Window
from Dosepy.GUILayouts.about_window import About_Window
#from GUILayouts.about_window import About_Window


#---------------------------------------------


class VentanaPrincipal(QMainWindow):
    """
    Ventana principal
    """
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: whitesmoke;")
        #self.setStyleSheet("background-color: #1d1040;")
        self.setWindowTitle('Dosepy')
        file_name_icon = pkg_resources.resource_filename('Dosepy', 'Icon/Icon.png')
        self.setWindowIcon(QIcon(file_name_icon))

        self.cuerpoUI()
        self.menuUI()

        self.film_to_dose_window = None
        self.about_window = None

        self.show()

    def cuerpoUI(self):

        cuerpo = QWidget()
        self.Imagen = Bloque_Inf_Imagenes()
        self.DatosEntrada = MostrarLabels()
        self.DatosEntrada.Eval_button.clicked.connect(self.mostrar_distribucion)
        self.DatosEntrada.Calcular_Button.clicked.connect(self.Calculo_Gamma)

        LayoutPrincipal = QVBoxLayout()
        LayoutPrincipal.addWidget(self.DatosEntrada, 0.6)
        LayoutPrincipal.addWidget(self.Imagen, 1)

        #self.setLayout(LayoutPrincipal)
        cuerpo.setLayout(LayoutPrincipal)

        self.setCentralWidget(cuerpo)

    def menuUI(self):
        """
        Crear un menú para la aplicación
        """
        # Crear acciones para el menú "Herramientas"
        film_to_dose_action = QAction('Dosimetría con película', self)
        film_to_dose_action.setShortcut('Ctrl+F')
        film_to_dose_action.triggered.connect(self.film_to_dose)   #   Descomentar para desarrollo

        about_action = QAction('Acerca de', self)
        about_action.triggered.connect(self.about)

        # Crear barra del menu
        barra_menu = self.menuBar()
        barra_menu.setNativeMenuBar(False)

        # Agregar menú herramientas y su acción a la barra del menú
        herram_menu = barra_menu.addMenu('Herramientas')
        herram_menu.addAction(film_to_dose_action)
        about_manu = barra_menu.addMenu('Ayuda')
        about_manu.addAction(about_action)


######################################################################
#   Funciones para menu herramientas

    def film_to_dose(self):
        if self.film_to_dose_window == None:
            self.film_to_dose_window = Film_to_Dose_Window()
        self.film_to_dose_window.show()

    def about(self):
        if self.about_window == None:
            self.about_window = About_Window()
        self.about_window.show()

######################################################################
#   Funciones para botones

    def mostrar_distribucion(self):
        if self.DatosEntrada.Formatos_ok == True:   # ¿Los archivos cumplen con las especificaciones?
            self.Imagen.Mpl_Izq.Img(self.DatosEntrada.Refer_npy)
            self.Imagen.Mpl_Izq.Colores(self.DatosEntrada.Eval_npy)

            self.Imagen.Mpl_Der.Img(self.DatosEntrada.Eval_npy)
            self.Imagen.Mpl_Der.Colores(self.DatosEntrada.Eval_npy)

            self.Imagen.Mpl_perfiles.ax.clear()
            self.Imagen.Mpl_perfiles.set_data_and_plot(self.DatosEntrada.Refer_npy, self.DatosEntrada.Eval_npy, self.Imagen.Mpl_Izq.circ)

            self.Imagen.Mpl_Izq.fig.canvas.draw()
            self.Imagen.Mpl_Der.fig.canvas.draw()

            self.Imagen.Mpl_perfiles.fig.canvas.draw()

        else:
            self.displayMessageBox()

    def Calculo_Gamma(self):
        D_ref = dp.Dose(self.Imagen.Mpl_Izq.npI, float(self.DatosEntrada.Resolution.text()))
        D_eval = dp.Dose(self.Imagen.Mpl_Der.npI, float(self.DatosEntrada.Resolution.text()))
        g, p = D_eval.gamma2D(D_ref, float(self.DatosEntrada.Toler_dosis.text()), float(self.DatosEntrada.Toler_dist.text()), float(self.DatosEntrada.Umbral_dosis.text()))

        self.DatosEntrada.Mpl_Histograma.Mostrar_Histograma(g)
        self.DatosEntrada.Indice_gamma_porcentaje_Label.setText('Porcentaje de aprobación: ' + str(  round(p, 1)  ) + '%' )
        self.DatosEntrada.Indice_gamma_promedio_Label.setText('Índice gamma promedio: ' + str(  round(np.mean(g[~np.isnan(g)]), 1)  ))
        #self.DatosEntrada.Indice_gamma_maximo_Label.setText('Máximo: ' + str(  round(np.max(g[~np.isnan(g)]), 1)  ))
        #self.DatosEntrada.Indice_gamma_mediana_Label.setText('Mediana: ' + str(  round(np.median(g[~np.isnan(g)]), 1)  ))

        self.DatosEntrada.Mpl_Img_gamma.ax1.clear()
        self.DatosEntrada.Mpl_Img_gamma.ax2.clear()
        self.DatosEntrada.Mpl_Img_gamma.Img(g)
        self.DatosEntrada.Mpl_Img_gamma.ax2.get_yaxis().set_visible(True)
        self.DatosEntrada.Mpl_Img_gamma.ax1.set_title('Distribución gamma', fontsize = 11)
        self.DatosEntrada.Mpl_Img_gamma.Colores(g[~np.isnan(g)])

        viridis = mpl.cm.get_cmap('viridis',256)
        hot = mpl.cm.get_cmap('hot',256)

        bounds_gamma = np.linspace(0,1.5,16)
        norm_gamma = mpl.colors.BoundaryNorm(boundaries = bounds_gamma, ncolors = 16)

        new_viridis = viridis(np.linspace(0,1,10))
        new_hot = np.flip(   hot(np.linspace(0,1,40)), 0   )
        new_hot = new_hot[20:26, :]
        new_color_gamma = np.vstack((new_viridis, new_hot))
        new_cmp_gamma = mpl.colors.ListedColormap(new_color_gamma)
        self.DatosEntrada.Mpl_Img_gamma.mplI.set_norm(norm_gamma)
        self.DatosEntrada.Mpl_Img_gamma.mplI.set_cmap(new_cmp_gamma)

       # I_g = self.DatosEntrada.Mpl_Img_gamma.ax1.pcolormesh(g, cmap = new_cmp_gamma, norm = norm_gamma)
        #cbar_gamma = self.DatosEntrada.Mpl_Img_gamma.colorbar(I_g, orientation='vertical', shrink = 0.6, cax = self.DatosEntrada.Mpl_Img_gamma.axe2)
        #cbar_gamma.ax.set_ylabel('Índice gamma', rotation=90, fontsize= 11)

        #self.DatosEntrada.Mpl_Img_gamma.Mostrar_Imagen(g)
        self.DatosEntrada.Mpl_Histograma.fig.canvas.draw()
        self.DatosEntrada.Mpl_Img_gamma.fig.canvas.draw()


######################################################################
#   Ventanas para mensajes
    def displayMessageBox(self):
        """
        Si la variable self.DatosEntrada.Formatos_ok es True, los archivos
        para las distribuciones de dosis se cargaron correctamente.
        En caso contrario se emite un mensaje de error.
        """
        QMessageBox().critical(self, "Error", "Error con la lectura de archivos.", QMessageBox.Ok, QMessageBox.Ok)





#if __name__ == '__main__':
#    app = QApplication(sys.argv)
#    window = VentanaPrincipal()

#    sys.exit(app.exec_())

app = QApplication(sys.argv)
window = VentanaPrincipal()
sys.exit(app.exec_())
