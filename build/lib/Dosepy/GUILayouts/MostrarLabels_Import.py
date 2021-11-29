import sys
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFormLayout, QLineEdit, QHBoxLayout, QVBoxLayout, QMessageBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from matplotlib.figure import Figure
import numpy as np
from .Imagen import Qt_Figure_Imagen
import pkg_resources
import os
import Dosepy.dose as dp


class MostrarLabels(QWidget):
    """
    Permite mostrar los archivos a cargar, parámetros gamma y resultados.
    """
    def __init__(self):
        super().__init__()
        self.iniciarUI()

    def iniciarUI(self):
        self.Widgets_Labels()
        #self.show()

    def Widgets_Labels(self):

        # Crear botones para cargar archivos
        #folder_icon = QIcon("Icon/folder.png")
        file_name_folder = pkg_resources.resource_filename('Dosepy', 'Icon/folder.png')
        folder_icon = QIcon(file_name_folder)

        self.Refer_button = QPushButton()
        self.Refer_button.clicked.connect(self.Leer_archivo_Referencia)
        self.Refer_button.setIcon(folder_icon)
        Refer_label = QLabel('D. de referencia')


        self.Eval_button = QPushButton()
        self.Eval_button.clicked.connect(self.Leer_archivo_Evaluacion)
        self.Eval_button.setIcon(folder_icon)
        Eval_label = QLabel('D. a evaluar')

        self.Formatos_ok = False

        #Resolution_Label = QLabel('Resolución [mm]')
        self.Resolution = QLineEdit()
        self.Resolution.setFixedWidth(35)
        self.Resolution.setText("1.0")

        # Crear LineEdit para parámetros gamma
        self.Toler_dosis = QLineEdit()
        self.Toler_dosis.setFixedWidth(30)
        self.Toler_dosis.setText("3.0")
        self.Toler_dosis.setAlignment(Qt.AlignRight)
        self.Toler_dist = QLineEdit()
        self.Toler_dist.setFixedWidth(30)
        self.Toler_dist.setText("3.0")
        self.Toler_dist.setAlignment(Qt.AlignRight)
        self.Umbral_dosis = QLineEdit()
        self.Umbral_dosis.setFixedWidth(30)
        self.Umbral_dosis.setText("10")
        self.Umbral_dosis.setAlignment(Qt.AlignRight)

        # Crear LineEdit para parámetros gamma

        self.Indice_gamma_porcentaje_Label = QLabel('Porcentaje de aprobación: ')
        self.Indice_gamma_porcentaje_Label.setStyleSheet("font-size: 16px")
        self.Indice_gamma_promedio_Label = QLabel('Índice gamma promedio: ')
        #self.Indice_gamma_maximo_Label = QLabel('Máximo: ')
        #self.Indice_gamma_mediana_Label = QLabel('Mediana: ')

        #   Crear boton para calcular

        self.Calcular_Button = QPushButton('Calcular')

        #Calcular_Button.setStyleSheet("border-radius: 10px")

        #   Crear histograma

        self.Mpl_Histograma = Qt_Figure_Histograma()


        #   Crear imagen para distribucion gamma
        self.Mpl_Img_gamma = Qt_Figure_Imagen()
        self.Mpl_Img_gamma.ax1.set_title('Distribución gamma', fontsize = 11)
        self.Mpl_Img_gamma.Cross_Hair_off()
        self.Mpl_Img_gamma.ax2.get_xaxis().set_visible(False)
        self.Mpl_Img_gamma.ax2.get_yaxis().set_visible(False)




        ##############################################################################
        ####    Crear contenedores


        #   Crear Layout horizontal para sección de archivos
        archivos_Pre_h_box = QHBoxLayout()
        archivos_Pre_h_box.addWidget(Refer_label)
        archivos_Pre_h_box.addWidget(self.Refer_button)

        archivos_Post_h_box = QHBoxLayout()
        archivos_Post_h_box.addWidget(Eval_label)
        archivos_Post_h_box.addWidget(self.Eval_button)

        Resolution_Form = QFormLayout()
        Resolution_Form.addRow('Distancia entre puntos [mm]', self.Resolution)

        #   Crear FormLayout
        Parametros_gamma_Layout = QFormLayout()
        #Parametros_gamma_Layout.setLabelAlignment(Qt.AlignLeft)
        Parametros_gamma_Layout.setFormAlignment(Qt.AlignRight)
        Parametros_gamma_Layout.addRow('Toler. en dosis [%]', self.Toler_dosis)
        Parametros_gamma_Layout.addRow('Toler. en distancia [mm]', self.Toler_dist)
        Parametros_gamma_Layout.addRow('Umbral de dosis [%]', self.Umbral_dosis)

        #   Crear vertical Layout

        Padre_Info_V_Layout = QVBoxLayout()
        Padre_Info_V_Layout.addLayout(archivos_Pre_h_box)
        Padre_Info_V_Layout.addLayout(archivos_Post_h_box)
        Padre_Info_V_Layout.addLayout(Resolution_Form)

        Padre_Info_V_Layout.addLayout(Parametros_gamma_Layout)
        Padre_Info_V_Layout.addStretch()
        Padre_Info_V_Layout.addWidget(self.Indice_gamma_porcentaje_Label)
        Padre_Info_V_Layout.addWidget(self.Indice_gamma_promedio_Label)
        #Padre_Info_V_Layout.addWidget(self.Indice_gamma_maximo_Label)
        #Padre_Info_V_Layout.addWidget(self.Indice_gamma_mediana_Label)
        Padre_Info_V_Layout.addWidget(self.Calcular_Button)


        Padre_Hist_V_Layout = QVBoxLayout()
        Padre_Hist_V_Layout.addWidget(self.Mpl_Histograma.Qt_fig)

        Padre_Img_Gamma_V_Layout = QVBoxLayout()
        Padre_Img_Gamma_V_Layout.addWidget(self.Mpl_Img_gamma.Qt_fig)

        Abuelo_H_Layout = QHBoxLayout()
        Abuelo_H_Layout.addLayout(Padre_Hist_V_Layout)
        Abuelo_H_Layout.addLayout(Padre_Img_Gamma_V_Layout)
        Abuelo_H_Layout.addLayout(Padre_Info_V_Layout)
        Abuelo_H_Layout.setStretchFactor(Padre_Info_V_Layout, 20)
        Abuelo_H_Layout.setStretchFactor(Padre_Hist_V_Layout, 40)
        Abuelo_H_Layout.setStretchFactor(Padre_Img_Gamma_V_Layout, 40)

        #Abuelo_H_Layout.addStretch()


        self.setLayout(Abuelo_H_Layout)




    ##############################################################################
    #   Funciones para botones que leen un archivo

    def Leer_archivo_Referencia(self):
        file_name_Referencia, _ = QFileDialog.getOpenFileName()
        _ , extension = os.path.splitext(file_name_Referencia)

        if file_name_Referencia:    #   Se obtuvo algún archivo?
            if extension == '.csv':

                self.Refer_npy = np.genfromtxt(file_name_Referencia, delimiter = ',')
                self.Refer_button.setStyleSheet("background-color: rgb(88,200,138)")

            #elif extension == '.dcm':
            #    self.Refer_npy = dp.from_dicom('file_name_Referencia').array
            #    self.Refer_button.setStyleSheet("background-color: rgb(88,200,138)")
            #    self.Resolution.setText(str(dp.from_dicom(file_name_Referencia).resolution))

            else:
                QMessageBox().critical(self, "Error", "Formato no válido.", QMessageBox.Ok, QMessageBox.Ok)
                print('Formato no valido')

    def Leer_archivo_Evaluacion(self):
        file_name_Evaluacion, _ = QFileDialog.getOpenFileName()
        _ , extension = os.path.splitext(file_name_Evaluacion)

        if file_name_Evaluacion:
            if extension == '.dcm':
                self.Eval_npy = dp.from_dicom(file_name_Evaluacion).array
                self.Resolution.setText(str(dp.from_dicom(file_name_Evaluacion).resolution))
                #self.Resolution.setReadOnly(True)

                if self.Eval_npy.shape != self.Refer_npy.shape:
                    QMessageBox().critical(self, "Error", "No es posible el cálculo con matrices de diferente tamaño.", QMessageBox.Ok, QMessageBox.Ok)
                    #raise Exception("No es posible el cálculo con matrices de diferente tamaño.")
                else:
                    self.Eval_button.setStyleSheet("background-color: rgb(88,200,138)")
                    self.Formatos_ok = True

            elif extension == '.csv':

                self.Eval_npy = np.genfromtxt(file_name_Evaluacion, delimiter = ',')

                if self.Eval_npy.shape != self.Refer_npy.shape:
                    QMessageBox().critical(self, "Error", "No es posible el cálculo con matrices de diferente tamaño.", QMessageBox.Ok, QMessageBox.Ok)
                    #raise Exception("No es posible el cálculo con matrices de diferente tamaño.")

                else:
                    self.Eval_button.setStyleSheet("background-color: rgb(88,200,138)")
                    self.Formatos_ok = True
            #print(self.file_name_Evaluacion)


class Qt_Figure_Histograma:
    """
    Clase para contener el histograma gamma
    """

    def __init__(self):
        self.fig = Figure(figsize=(3.5,2.0), facecolor = 'whitesmoke')
        self.Qt_fig = FigureCanvas(self.fig)

        #   Axes para la imagen
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_title('Histograma gamma', fontsize = 11)
        self.ax.grid(alpha = 0.3)


    def Mostrar_Histograma(self, g):
        self.ax.hist( g[~np.isnan(g)] , bins = 50, range = (0,3), alpha = 0.7)

        self.fig.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MostrarLabels()
    window.show()
    sys.exit(app.exec_())
