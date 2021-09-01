# -*- coding: utf-8 -*-

"""

Última modificación: 28 Agosto 2021
@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico en Radioterapia, La Paz, Baja California Sur, México.

"""

#---------------------------------------------
#   Importaciones
from Dosepy.tools.film_to_dose import calibracion, cubico
from Dosepy.tools.resol import equalize

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QHBoxLayout, QMessageBox, QMainWindow, QAction, QLabel, QPushButton, QFileDialog, QLayout, QCheckBox, QLineEdit, QFormLayout
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.colors as colors
import tifffile as tiff
import numpy as np
import pkg_resources
import sys
import os

#---------------------------------------------

class Film_to_Dose_Window(QWidget):
    """
    Ventana para dosimetría con película
    """
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: whitesmoke;")
        self.setWindowTitle('Dosimetría con película')

        self.iniciarUI_calibracion()


    def iniciarUI_calibracion(self):

        file_name_folder = pkg_resources.resource_filename('Dosepy', 'Icon/folder.png')
        folder_icon = QIcon(file_name_folder)

        layout_bisabuelo = QVBoxLayout()

        layout_abuelo = QHBoxLayout()
        layout_abuelo_2inf = QHBoxLayout()

        layout_padre_botones = QVBoxLayout()
        layout_padre_botones_2inf = QVBoxLayout()


        self.button_leer_tiff_pre = QPushButton('Calib.')
        self.button_leer_tiff_pre.clicked.connect(self.leer_tiff_pre)
        self.button_leer_tiff_pre.setIcon(folder_icon)

        #self.button_leer_tiff_post = QPushButton(' Post')
        #self.button_leer_tiff_post.clicked.connect(self.leer_tiff_post)
        #self.button_leer_tiff_post.setIcon(folder_icon)

        self.label_a0 = QLabel()
        self.label_a1 = QLabel()
        self.label_a2 = QLabel()
        self.label_a3 = QLabel()

        layout_padre_botones.addSpacing(50)
        layout_padre_botones.addWidget(self.button_leer_tiff_pre)
        layout_padre_botones.addWidget(self.label_a0)
        layout_padre_botones.addWidget(self.label_a1)
        layout_padre_botones.addWidget(self.label_a2)
        layout_padre_botones.addWidget(self.label_a3)
        layout_padre_botones.addSpacing(70)
        layout_padre_botones.setAlignment(Qt.AlignTop)

        #   Widget para los perfiles
        self.Qt_Mpl_curva_calib = Qt_Figure_CurvaCalibracion()

        layout_abuelo.addWidget(self.Qt_Mpl_curva_calib.Qt_fig)
        layout_abuelo.addLayout(layout_padre_botones)

        file_name_image_logo = pkg_resources.resource_filename('Dosepy', 'Icon/Logo_Dosepy.png')
        #file_name_image_logo = 'Logo_Dosepy.png'
        pixmap_logo = QPixmap(file_name_image_logo)
        self.label_logo = QLabel(self)
        self.label_logo.setAlignment(Qt.AlignCenter)
        self.label_logo.setPixmap(pixmap_logo)
        self.Qt_Mpl_distribucion = Qt_Figure_Imagen()

        self.button_distr_pre = QPushButton('Dist.')
        self.button_distr_pre.setEnabled(False)
        self.button_distr_pre.clicked.connect(self.leer_tiff_pre_distr)
        self.button_distr_pre.setIcon(folder_icon)

        #self.button_distr_post = QPushButton(' Post')
        #self.button_distr_post.setEnabled(False)
        #self.button_distr_post.clicked.connect(self.leer_tiff_post_distr)
        #self.button_distr_post.setIcon(folder_icon)


        self.check_button_tif = QCheckBox("TIFF", self)
        self.check_button_csv = QCheckBox("CSV", self)

        self.button_guardar = QPushButton('Guardar')
        self.button_guardar.setEnabled(False)
        self.button_guardar.clicked.connect(self.guardar_distribucion)

        self.button_reducir = QPushButton('Reducir')
        self.button_reducir.setEnabled(False)
        self.button_reducir.clicked.connect(self.reducir_tamano)

        self.QLineEdit_resol = QLineEdit()
        self.QLineEdit_resol.setFixedWidth(45)
        Qform_layout_resol = QFormLayout()
        Qform_layout_resol.addRow('Ref.', self.QLineEdit_resol)
        #self.QLineEdit_resol.setText("1.0")

        #label_fondo = QLabel(self)
        #label_fondo.setStyleSheet("background-color: lightgreen")
        #label_fondo.setStyleSheet("background-color: darkblue")

        layout_padre_botones_2inf.addSpacing(40)
        layout_padre_botones_2inf.addWidget(self.button_distr_pre)
        #layout_padre_botones_2inf.addWidget(self.button_distr_post)
        layout_padre_botones_2inf.addWidget(self.button_guardar)
        layout_padre_botones_2inf.addWidget(self.check_button_tif)
        layout_padre_botones_2inf.addWidget(self.check_button_csv)
        layout_padre_botones_2inf.addWidget(self.button_reducir)
        layout_padre_botones_2inf.addLayout(Qform_layout_resol)
        layout_padre_botones_2inf.setAlignment(Qt.AlignTop)

        layout_abuelo_2inf.addWidget(self.Qt_Mpl_distribucion.Qt_fig)
        layout_abuelo_2inf.addLayout(layout_padre_botones_2inf)

        layout_bisabuelo.addWidget(self.label_logo)
        layout_bisabuelo.addLayout(layout_abuelo)
        layout_bisabuelo.addSpacing(40)
        layout_bisabuelo.addLayout(layout_abuelo_2inf)
        #layout_bisabuelo.addWidget(label_fondo)

        #layout_abuelo.setContentMargins(top, bottom)

        self.setLayout(layout_bisabuelo)

    ##############################################################################
    #   Funciones para botones que leen un archivo

    def leer_tiff_pre(self):

        file_name_pre, _ = QFileDialog.getOpenFileName(self, "Archivo tif PRE irradiación", filter="*.tif")
        _ , extension = os.path.splitext(file_name_pre)

        if file_name_pre:    #   Se obtuvo algún archivo?
            if extension == '.tif':

                self.imagen_calib_pre = tiff.imread( file_name_pre )
                print(file_name_pre)
                #self.button_leer_tiff_pre.setStyleSheet("background-color: rgb(88,200,138)")

                self.leer_tiff_post()

            else:
                QMessageBox().critical(self, "Error", "Formato no válido.", QMessageBox.Ok, QMessageBox.Ok)
                print('Formato no válido')

    def leer_tiff_post(self):

        file_name_post, _ = QFileDialog.getOpenFileName(self, "Archivo tiff POST irradiación", filter="*.tif" )
        _ , extension = os.path.splitext(file_name_post)

        if file_name_post:    #   Se obtuvo algún archivo?
            if extension == '.tif':

                self.imagen_calib_post = tiff.imread( file_name_post )

                if self.imagen_calib_pre.shape != self.imagen_calib_post.shape:
                    QMessageBox().critical(self, "Error", "Las imágenes debe tener el mismo tamaño.", QMessageBox.Ok, QMessageBox.Ok)
                    #raise Exception("No es posible el cálculo con matrices de diferente tamaño.")

                else:

                    print(file_name_post)
                    self.button_leer_tiff_pre.setStyleSheet("background-color: rgb(88,200,138)")
                    self.coef_calib, Dens_optica, Dosis_imaprtida = calibracion(self.imagen_calib_pre, self.imagen_calib_post)
                    x = np.linspace(0, Dens_optica[9])
                    y = cubico(x, self.coef_calib[0], self.coef_calib[1], self.coef_calib[2], self.coef_calib[3])
                    self.Qt_Mpl_curva_calib.ax.plot(x,y)
                    self.Qt_Mpl_curva_calib.ax.plot(Dens_optica, Dosis_imaprtida, 'g*', label = 'Datos')
                    #self.Qt_Mpl_curva_calib.ax.text(0.3*np.amax(x), 0.05*np.amax(y), r'$D = {:.3f} + {:.3f}x + {:.3f}x^2 + {:.3f}x^3$'.format( coef_calib[0], coef_calib[1], coef_calib[2], coef_calib[3] ))
                    self.Qt_Mpl_curva_calib.ax.text(0.45*np.amax(x), 0.05*np.amax(y), r'$D = a_0 + a_1x + a_2x^2 + a_3x^3$')
                    self.Qt_Mpl_curva_calib.fig.canvas.draw()
                    self.label_a0.setText("a0: {:.4f}".format(self.coef_calib[0]))
                    self.label_a1.setText("a1: {:.4f}".format(self.coef_calib[1]))
                    self.label_a2.setText("a2: {:.4f}".format(self.coef_calib[2]))
                    self.label_a3.setText("a3: {:.4f}".format(self.coef_calib[3]))
                    self.button_distr_pre.setEnabled(True)

            else:
                QMessageBox().critical(self, "Error", "Formato no válido.", QMessageBox.Ok, QMessageBox.Ok)
                print('Formato no válido')


    def leer_tiff_pre_distr(self):

        file_name_pre, _ = QFileDialog.getOpenFileName(self, "Archivo tif PRE irradiación", filter="*.tif")
        _ , extension = os.path.splitext(file_name_pre)

        if file_name_pre:    #   Se obtuvo algún archivo?
            if extension == '.tif':

                self.imagen_distr_pre = tiff.imread( file_name_pre )
                imagen_tiff = tiff.TiffFile(file_name_pre)
                resolucion = imagen_tiff.pages[0].tags['XResolution'].value[0] / imagen_tiff.pages[0].tags['XResolution'].value[1]
                self.image_distr_resolucion_mm_punto = 25.4 / resolucion
                #print(file_name_pre)
                #self.button_distr_pre.setStyleSheet("background-color: rgb(88,200,138)")

                self.leer_tiff_post_distr()


            else:
                QMessageBox().critical(self, "Error", "Formato no válido.", QMessageBox.Ok, QMessageBox.Ok)
                print('Formato no válido')


    def leer_tiff_post_distr(self):

        file_name_post, _ = QFileDialog.getOpenFileName(self, "Archivo tiff POST irradiación", filter="*.tif" )
        _ , extension = os.path.splitext(file_name_post)

        if file_name_post:    #   Se obtuvo algún archivo?
            if extension == '.tif':

                self.imagen_distr_post = tiff.imread( file_name_post )
                print(file_name_post)

                if self.imagen_distr_pre.shape != self.imagen_distr_post.shape:
                    QMessageBox().critical(self, "Error", "Las imágenes debe tener el mismo tamaño.", QMessageBox.Ok, QMessageBox.Ok)
                    #raise Exception("No es posible el cálculo con matrices de diferente tamaño.")

                else:

                    self.button_distr_pre.setStyleSheet("background-color: rgb(88,200,138)")

                    film_pre_prom = np.mean(self.imagen_distr_pre, axis = 2)
                    film_post_prom = np.mean(self.imagen_distr_post, axis = 2)
                    film_DO = -np.log10(film_post_prom / (film_pre_prom + 1))   #La suma de uno se utiliza para evitar la división por cero

                    self.Dosis_FILM = cubico(film_DO, self.coef_calib[0], self.coef_calib[1], self.coef_calib[2], self.coef_calib[3])

                    self.Qt_Mpl_distribucion.Img(self.Dosis_FILM)
                    self.Qt_Mpl_distribucion.Colores(self.Dosis_FILM)
                    self.Qt_Mpl_distribucion.fig.canvas.draw()

                    self.button_guardar.setEnabled(True)
                    self.button_reducir.setEnabled(True)

            else:
                QMessageBox().critical(self, "Error", "Formato no válido.", QMessageBox.Ok, QMessageBox.Ok)
                print('Formato no válido')


    def guardar_distribucion(self):


        if self.check_button_tif.isChecked() == True:
            file_name_tif, _ = QFileDialog.getSaveFileName(self, "Guardar distribución de dosis", "", "*.tif")
            D_tiff = np.uint16(self.Dosis_FILM*100)
            tiff.imwrite(file_name_tif + '.tif', D_tiff)
            print('Guardar tiff')

        if self.check_button_csv.isChecked() == True:

            file_name_csv, _ = QFileDialog.getSaveFileName(self, "Guardar distribución de dosis", "", "*.csv")
            np.savetxt(file_name_csv + '.csv', self.Dosis_FILM, fmt = '%.3f', delimiter = ',')
            print('Guardar csv')

    def reducir_tamano(self):

        self.Dosis_FILM = equalize(self.Dosis_FILM, self.image_distr_resolucion_mm_punto, float(self.QLineEdit_resol.text()))
        self.Qt_Mpl_distribucion.Img(self.Dosis_FILM)
        self.Qt_Mpl_distribucion.Colores(self.Dosis_FILM)
        self.Qt_Mpl_distribucion.fig.canvas.draw()

class Qt_Figure_CurvaCalibracion:
    """
    Clase para generar el widget que contendrá la curva de calibración
    """

    def __init__(self):

        self.fig = Figure(figsize=(4.5,3), facecolor = 'whitesmoke', tight_layout = True)
        self.Qt_fig = FigureCanvas(self.fig)

        #   Axes para la imagen
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_title('Curva de calibración', fontsize = 12)
        #self.ax.set_title('Curva de calibración', fontsize = 14)
        self.ax.set_ylabel('Dosis [Gy]')
        #self.ax.set_ylabel('Dose [Gy]')
        self.ax.set_xlabel('Densidad óptica')
        #self.ax.set_xlabel('Optical density')
        self.ax.grid(alpha = 0.3)

class Qt_Figure_Imagen:
    """
    Clase para contener la distribución de dosis
    """

    def __init__(self):
        #self.fig = Figure(figsize=(4,3), tight_layout = True, facecolor = 'whitesmoke')
        self.fig = Figure(figsize=(5.5,5), facecolor = 'whitesmoke', tight_layout = True)
        self.Qt_fig = FigureCanvas(self.fig)

        self.ax1 = self.fig.add_axes([0.08, 0.08, 0.75, 0.85])
        self.ax2 = self.fig.add_axes([0.85, 0.15, 0.04, 0.72])
        self.ax2.set_ylabel('Dosis [Gy]')
        self.ax2.yaxis.set_label_position("right")
        self.ax2.yaxis.tick_right()
        self.ax1.set_title('Distribución de dosis', fontsize = 12)

    def Img(self, np_I):
        '''
        Definir la imagen a partir de un array que se proporciona como argumento.
        '''

        self.npI = np_I
        self.mplI = self.ax1.imshow(self.npI)

    def Colores(self, npI_color_ref):
        '''
        Definir el mapa de colores a utiliar.
        '''
        color_map = 'viridis'
        bounds = np.linspace(0, round(1.15 * np.percentile(npI_color_ref, 98)), 256)
        norm = colors.BoundaryNorm(boundaries = bounds, ncolors = 256)
        self.mplI.set_norm(norm)
        self.mplI.set_cmap(color_map)
        self.cbar = self.fig.colorbar(self.mplI, cax = self.ax2, orientation = 'vertical', shrink = 0.6, format = '%.1f')
        self.ax2.set_ylabel('Dosis [Gy]')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Film_to_Dose_Window()
    window.show()

    sys.exit(app.exec_())
