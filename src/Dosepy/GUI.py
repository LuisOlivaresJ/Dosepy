# -*- coding: utf-8 -*-
"""

Última modificación: 20 Julio 2021
@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico en Radioterapia, La Paz, Baja California Sur, México.

"""
#---------------------------------------------
#   Importaciones

import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QHBoxLayout
from PyQt5.QtGui import QIcon
import numpy as np
from Dosepy.Layouts.MostrarLabels_Import import MostrarLabels
from Dosepy.Layouts.Imagen import VentanaRaiz
import Dosepy.dose as dp
import matplotlib as mpl
import pkg_resources
#---------------------------------------------


class VentanaPrincipal(QWidget):
    """
    Ventana principal
    """
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: whitesmoke;")
        self.setWindowTitle('Dosepy')
        file_name_icon = pkg_resources.resource_filename('Dosepy', 'Icon/Icon.png')
        #folder_icon = QIcon(file_name_icon)
        self.setWindowIcon(QIcon(file_name_icon))
        self.iniciarUI()
        self.show()

    def iniciarUI(self):

        self.Imagen = VentanaRaiz()
        self.DatosEntrada = MostrarLabels()
        self.DatosEntrada.Eval_button.clicked.connect(self.mostrar_distribucion)
        self.DatosEntrada.Calcular_Button.clicked.connect(self.Calculo_Gamma)

        LayoutPrincipal = QVBoxLayout()
        LayoutPrincipal.addWidget(self.DatosEntrada, 0.6)
        LayoutPrincipal.addWidget(self.Imagen, 1)

        self.setLayout(LayoutPrincipal)

######################################################################
#   Funciones para botones

    def mostrar_distribucion(self):
        self.Imagen.Mpl_Izq.Img(self.DatosEntrada.Refer_npy)
        self.Imagen.Mpl_Izq.Colores(self.DatosEntrada.Eval_npy)

        self.Imagen.Mpl_Der.Img(self.DatosEntrada.Eval_npy)
        self.Imagen.Mpl_Der.Colores(self.DatosEntrada.Eval_npy)

        self.Imagen.Mpl_perfiles.ax.clear()
        self.Imagen.Mpl_perfiles.set_data_and_plot(self.DatosEntrada.Refer_npy, self.DatosEntrada.Eval_npy, self.Imagen.Mpl_Izq.circ)

        self.Imagen.Mpl_Izq.fig.canvas.draw()
        self.Imagen.Mpl_Der.fig.canvas.draw()

        self.Imagen.Mpl_perfiles.fig.canvas.draw()


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









#if __name__ == '__main__':
#    app = QApplication(sys.argv)
#    window = VentanaPrincipal()

#    sys.exit(app.exec_())

app = QApplication(sys.argv)
window = VentanaPrincipal()
sys.exit(app.exec_())
