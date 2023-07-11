# -*- coding: utf-8 -*-
"""
@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico, La Paz, Baja California Sur, México.

    Derechos Reservados (c) Luis Alfonso Olivares Jimenez 2021
"""
#---------------------------------------------
#   Importaciones

import sys

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication, QHBoxLayout, QMessageBox, QMainWindow, QLabel, QLineEdit
from PyQt6.QtGui import QIcon, QFont,  QAction
from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import QFileDialog, QInputDialog
from relative_dose_1d.tools import build_from_array_and_step, gamma_1D
#from relative_dose_1d.GUI import Q_Graphic_Block



import numpy as np
#from Dosepy.GUILayouts.Bloque_gamma import Bloque_gamma
from GUILayouts.Bloque_gamma import Bloque_gamma  # Se importa desde archivo en PC para testear
#from Dosepy.GUILayouts.Bloque_Imagenes import Bloque_Imagenes
from GUILayouts.Bloque_Imagenes import Bloque_Imagenes   # Se importa desde archivo en PC para testear
import Dosepy.dose as dp
import matplotlib as mpl
import pkg_resources

from Dosepy.GUILayouts.film_to_doseGUI import Film_to_Dose_Window
#from GUILayouts.film_to_doseGUI import Film_to_Dose_Window
from Dosepy.GUILayouts.about_window import About_Window
#from GUILayouts.about_window import About_Window
from Dosepy.GUILayouts.licencia_window import Licencia_Window
#from GUILayouts.licencia_window import Licencia_Window
#---------------------------------------------


class VentanaPrincipal(QMainWindow):
    """
    Ventana principal
    """
    def __init__(self, Us):
        super().__init__()
        self.Us = Us
        self.setStyleSheet("background-color: whitesmoke;")
        #self.setStyleSheet("background-color: #1d1040;")
        self.setWindowTitle('Dosepy')
        file_name_icon = pkg_resources.resource_filename('Dosepy', 'Icon/Icon.png')
        self.setWindowIcon(QIcon(file_name_icon))
        self.setGeometry(150, 100, 1300, 800)

        self.cuerpoUI()
        self.menuUI()

        #Ventanas secundarias
        self.film_to_dose_window = None
        self.about_window = None
        self.licencia_window_sec = None

        #   Validez de formatos
        self.formatos_ok = False

        self.show()

    def cuerpoUI(self):

        cuerpo = QWidget()
        self.Bloque_Imagen = Bloque_Imagenes()

        self.Bloque_Imagen.boton_recortar_Izq.clicked.connect(self.Cortar_Imagen)
        self.Bloque_Imagen.compare_button.clicked.connect(self.Compare_profiles)


        self.Bloque_Gamma = Bloque_gamma(self.Us)

        self.Bloque_Gamma.Refer_button.clicked.connect(self.Leer_archivo_Referencia)
        self.Bloque_Gamma.Eval_button.clicked.connect(self.Leer_archivo_Evaluacion)
        self.Bloque_Gamma.Eval_button.clicked.connect(self.mostrar_distribucion)
        self.Bloque_Gamma.Calcular_Button.clicked.connect(self.Calculo_Gamma)

        LayoutPrincipal = QVBoxLayout()
        LayoutPrincipal.addWidget(self.Bloque_Gamma)
        LayoutPrincipal.addWidget(self.Bloque_Imagen)

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

        about_action = QAction('Acerca de...', self)
        about_action.triggered.connect(self.about_ventana)

        licencia_action = QAction('Licencia', self)
        licencia_action.triggered.connect(self.licencia_ventana)

        # Crear barra del menu
        barra_menu = self.menuBar()
        barra_menu.setNativeMenuBar(False)

        # Agregar menú herramientas y su acción a la barra del menú
        herram_menu = barra_menu.addMenu('Herramientas')
        herram_menu.addAction(film_to_dose_action)
        about_menu = barra_menu.addMenu('Ayuda')
        about_menu.addAction(about_action)
        about_menu.addAction(licencia_action)
        #licencia_action
        #about_manu.AddAction(licencia_action)


######################################################################
#   Funciones para menu herramientas

    def film_to_dose(self):
        if self.film_to_dose_window == None:
            self.film_to_dose_window = Film_to_Dose_Window()
        self.film_to_dose_window.show()

    def about_ventana(self):
        if self.about_window == None:
            self.about_window = About_Window()
        self.about_window.show()

    def licencia_ventana(self):
        if self.licencia_window_sec == None:
            self.licencia_window_sec = Licencia_Window()
        self.licencia_window_sec.show()

######################################################################
#   Funciones para botones

    ##############################################################################
    #   Funciones para botones que leen un archivo

    def Leer_archivo_Referencia(self):
        file_name_Referencia, _ = QFileDialog.getOpenFileName()
        _ , extension = os.path.splitext(file_name_Referencia)

        if file_name_Referencia:    #   Se obtuvo algún archivo?
            if extension == '.csv':

                resolution, ok_resol = QInputDialog.getDouble(
                    self, 
                    "Resolución", 
                    "Resolución [mm/punto]",
                    decimals = 5)
                Refer_npy = np.genfromtxt(file_name_Referencia, delimiter = ',')
                self.Bloque_Imagen.D_ref = dp.Dose(Refer_npy, resolution)
                self.Bloque_Gamma.Refer_button.setStyleSheet("background-color: rgb(88,200,138)")

                self.Bloque_Gamma.Eval_button.setEnabled(True)

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
                self.Bloque_Imagen.D_eval = dp.from_dicom(file_name_Evaluacion)
                #self.Resolution.setText(str(dp.from_dicom(file_name_Evaluacion).resolution))
                #self.Resolution.setReadOnly(True)

                if self.Bloque_Imagen.D_eval.array.shape != self.Bloque_Imagen.D_ref.array.shape:
                    QMessageBox().critical(
                    self,
                    "Error",
                    """No es posible el análisis con matrices de diferente tamaño.\n
                    Referencia: {}\n
                    A evaluar: {}""".format(self.Refer_npy.shape, self.Eval_npy.shape),
                    QMessageBox.Ok, QMessageBox.Ok
                    )

                else:
                    self.Bloque_Gamma.Eval_button.setStyleSheet("background-color: rgb(88,200,138)")
                    self.formatos_ok = True

            elif extension == '.csv':

                resolution, ok_resol = QInputDialog.getDouble(self, 
                                                              "Resolución", 
                                                              "Resolución [mm/punto]",
                                                              decimals = 5)      
                Eval_npy = np.genfromtxt(file_name_Evaluacion, delimiter = ',')
                self.Bloque_Imagen.D_eval = dp.Dose(Eval_npy, resolution)

                if self.Bloque_Imagen.D_eval.array.shape != self.Bloque_Imagen.D_ref.array.shape:
                    QMessageBox().critical(
                    self,
                    "Error",
                    """No es posible el análisis con matrices de diferente tamaño.\n
                    Referencia: {}\n
                    A evaluar: {}""".format(self.Bloque_Imagen.D_ref.array.shape, self.Bloque_Imagen.array.D_eval.shape),
                    QMessageBox.Ok, QMessageBox.Ok
                    )

                else:
                    self.Bloque_Gamma.Eval_button.setStyleSheet("background-color: rgb(88,200,138)")
                    self.formatos_ok = True
            #print(self.file_name_Evaluacion)

    def mostrar_distribucion(self):
        if self.formatos_ok == True:   # ¿Los archivos cumplen con las especificaciones?
            self.Bloque_Imagen.Mpl_Izq.Img(self.Bloque_Imagen.D_ref)
            self.Bloque_Imagen.Mpl_Izq.Colores(self.Bloque_Imagen.D_eval.array)

            self.Bloque_Imagen.Mpl_Der.Img(self.Bloque_Imagen.D_eval)
            self.Bloque_Imagen.Mpl_Der.Colores(self.Bloque_Imagen.D_eval.array)

            self.Bloque_Imagen.Mpl_perfiles.ax.clear()
            self.Bloque_Imagen.Mpl_perfiles.set_data_and_plot(self.Bloque_Imagen.D_ref.array, self.Bloque_Imagen.D_eval.array, self.Bloque_Imagen.Mpl_Izq.circ)

            self.Bloque_Imagen.Mpl_Izq.fig.canvas.draw()
            self.Bloque_Imagen.Mpl_Der.fig.canvas.draw()

            self.Bloque_Imagen.Mpl_perfiles.fig.canvas.draw()

        else:
            self.displayMessageBox()

    def Calculo_Gamma(self):
        #D_ref = dp.Dose(self.Bloque_Imagen.Mpl_Izq.npI, float(self.Bloque_Gamma.Resolution.text()))
        D_ref = self.Bloque_Imagen.D_ref
        #D_eval = dp.Dose(self.Bloque_Imagen.Mpl_Der.npI, float(self.Bloque_Gamma.Resolution.text()))
        D_eval = self.Bloque_Imagen.D_eval
        g, p = D_eval.gamma2D(D_ref, float(self.Bloque_Gamma.Toler_dosis.text()), float(self.Bloque_Gamma.Toler_dist.text()), float(self.Bloque_Gamma.Umbral_dosis.text()))

        self.Bloque_Gamma.Mpl_Histograma.Mostrar_Histograma(g)
        self.Bloque_Gamma.Indice_gamma_porcentaje_Label.setText('Porcentaje de aprobación: ' + str(  round(p, 1)  ) + '%' )
        self.Bloque_Gamma.Indice_gamma_promedio_Label.setText('Índice gamma promedio: ' + str(  round(np.mean(g[~np.isnan(g)]), 1)  ))
        #self.Bloque_Gamma.Indice_gamma_maximo_Label.setText('Máximo: ' + str(  round(np.max(g[~np.isnan(g)]), 1)  ))
        #self.Bloque_Gamma.Indice_gamma_mediana_Label.setText('Mediana: ' + str(  round(np.median(g[~np.isnan(g)]), 1)  ))

        self.Bloque_Gamma.Mpl_Img_gamma.ax1.clear()
        self.Bloque_Gamma.Mpl_Img_gamma.ax2.clear()
        _Dg = dp.Dose(g, float(self.Bloque_Imagen.D_ref.resolution))
        self.Bloque_Gamma.Mpl_Img_gamma.Img(_Dg)
        self.Bloque_Gamma.Mpl_Img_gamma.ax2.get_yaxis().set_visible(True)
        self.Bloque_Gamma.Mpl_Img_gamma.ax1.set_title('Distribución gamma', fontsize = 11)
        self.Bloque_Gamma.Mpl_Img_gamma.Colores(g[~np.isnan(g)])

        viridis = mpl.cm.get_cmap('viridis',256)
        hot = mpl.cm.get_cmap('hot',256)

        bounds_gamma = np.linspace(0,1.5,16)
        norm_gamma = mpl.colors.BoundaryNorm(boundaries = bounds_gamma, ncolors = 16)

        new_viridis = viridis(np.linspace(0,1,10))
        new_hot = np.flip(   hot(np.linspace(0,1,40)), 0   )
        new_hot = new_hot[20:26, :]
        new_color_gamma = np.vstack((new_viridis, new_hot))
        new_cmp_gamma = mpl.colors.ListedColormap(new_color_gamma)
        self.Bloque_Gamma.Mpl_Img_gamma.mplI.set_norm(norm_gamma)
        self.Bloque_Gamma.Mpl_Img_gamma.mplI.set_cmap(new_cmp_gamma)

       # I_g = self.Bloque_Gamma.Mpl_Img_gamma.ax1.pcolormesh(g, cmap = new_cmp_gamma, norm = norm_gamma)
        #cbar_gamma = self.Bloque_Gamma.Mpl_Img_gamma.colorbar(I_g, orientation='vertical', shrink = 0.6, cax = self.Bloque_Gamma.Mpl_Img_gamma.axe2)
        #cbar_gamma.ax.set_ylabel('Índice gamma', rotation=90, fontsize= 11)

        #self.Bloque_Gamma.Mpl_Img_gamma.Mostrar_Imagen(g)
        self.Bloque_Gamma.Mpl_Histograma.fig.canvas.draw()
        self.Bloque_Gamma.Mpl_Img_gamma.fig.canvas.draw()

    def Cortar_Imagen(self):

        xi = int(self.Bloque_Imagen.Mpl_Izq.Rectangle.get_x())
        width = int(self.Bloque_Imagen.Mpl_Izq.Rectangle.get_width())
        yi = int(self.Bloque_Imagen.Mpl_Izq.Rectangle.get_y())
        height = int(self.Bloque_Imagen.Mpl_Izq.Rectangle.get_height())

        npI_Izq = self.Bloque_Imagen.Mpl_Izq.npI[  yi : yi + height , xi : xi + width ]
        npI_Der = self.Bloque_Imagen.Mpl_Der.npI[  yi : yi + height , xi : xi + width ]
        self.Bloque_Imagen.D_ref = dp.Dose(npI_Izq, self.Bloque_Imagen.D_ref.resolution)
        self.Bloque_Imagen.D_eval = dp.Dose(npI_Der, self.Bloque_Imagen.D_eval.resolution)

        self.Bloque_Imagen.Mpl_Izq.Img(self.Bloque_Imagen.D_ref)
        self.Bloque_Imagen.Mpl_Der.Img(self.Bloque_Imagen.D_eval)

        self.Bloque_Imagen.Mpl_Izq.Colores(npI_Der)
        self.Bloque_Imagen.Mpl_Der.Colores(npI_Der)

        self.Bloque_Imagen.Mpl_Izq.Cross_Hair_on()
        self.Bloque_Imagen.Mpl_Der.Cross_Hair_on()

        self.Bloque_Imagen.Mpl_perfiles.set_data_and_plot(npI_Izq, npI_Der, self.Bloque_Imagen.Mpl_Izq.circ)

        self.Bloque_Imagen.Mpl_Izq.ROI_Rect_off()
        self.Bloque_Imagen.boton_recortar_Izq.setEnabled(False)
        self.Bloque_Imagen.boton_roi.setChecked(False)



    def Cortar_Imagen(self):

        xi = int(self.Bloque_Imagen.Mpl_Izq.Rectangle.get_x())
        width = int(self.Bloque_Imagen.Mpl_Izq.Rectangle.get_width())
        yi = int(self.Bloque_Imagen.Mpl_Izq.Rectangle.get_y())
        height = int(self.Bloque_Imagen.Mpl_Izq.Rectangle.get_height())

        npI_Izq = self.Bloque_Imagen.Mpl_Izq.npI[  yi : yi + height , xi : xi + width ]
        npI_Der = self.Bloque_Imagen.Mpl_Der.npI[  yi : yi + height , xi : xi + width ]
        self.Bloque_Imagen.D_ref = dp.Dose(npI_Izq, self.Bloque_Imagen.D_ref.resolution)
        self.Bloque_Imagen.D_eval = dp.Dose(npI_Der, self.Bloque_Imagen.D_eval.resolution)

        self.Bloque_Imagen.Mpl_Izq.Img(self.Bloque_Imagen.D_ref)
        self.Bloque_Imagen.Mpl_Der.Img(self.Bloque_Imagen.D_eval)

        self.Bloque_Imagen.Mpl_Izq.Colores(npI_Der)
        self.Bloque_Imagen.Mpl_Der.Colores(npI_Der)

        self.Bloque_Imagen.Mpl_Izq.Cross_Hair_on()
        self.Bloque_Imagen.Mpl_Der.Cross_Hair_on()

        self.Bloque_Imagen.Mpl_perfiles.set_data_and_plot(npI_Izq, npI_Der, self.Bloque_Imagen.Mpl_Izq.circ)

        self.Bloque_Imagen.Mpl_Izq.ROI_Rect_off()
        self.Bloque_Imagen.boton_recortar_Izq.setEnabled(False)
        self.Bloque_Imagen.boton_roi.setChecked(False)

    def Compare_profiles(self):

        resolution = self.Bloque_Imagen.D_ref.resolution

        D_profile_ref = build_from_array_and_step(
            self.Bloque_Imagen.Mpl_perfiles.perfil_horizontal_ref,
            resolution
            )
        D_profile_eval = build_from_array_and_step(
            self.Bloque_Imagen.Mpl_perfiles.perfil_horizontal_eval,
            resolution
            )
        
        gamma, gamma_percent = gamma_1D(
            D_profile_ref, 
            D_profile_eval,
            dose_t = 3, dist_t = 2, dose_tresh = 10, interpol = 1)
        #print(gamma)
        print(gamma_percent)

        # TO_DO 
        # gamma_1D requieres data to be normalized
        # new fuction inside relative_dose_1d to whow a plot 
        # plot_gamma_profiles() 


######################################################################
#   Ventanas para mensajes
    def displayMessageBox(self):
        """
        Si la variable self.formatos_ok es True, los archivos
        para las distribuciones de dosis se cargaron correctamente.
        En caso contrario se emite un mensaje de error.
        """
        QMessageBox().critical(self, "Error", "Error con la lectura de archivos.", QMessageBox.Ok, QMessageBox.Ok)

class Ventana_Secundaria(QMainWindow):
    'Clase para mantenimiento de Dosepy'
    def __init__(self):
        super().__init__()

        self.about_window_sec = None
        self.licencia_window_sec = None

        self.iniciarUI()
        self.menu_ayuda()
        #self.setCentralWidget(cuerpo_sec)

    def iniciarUI(self):

        cuerpo = QWidget()
        layout_principal = QVBoxLayout()
        cuerpo.setLayout(layout_principal)
        self.setCentralWidget(cuerpo)

        self.setStyleSheet("background-color: whitesmoke;")
        self.setWindowTitle('Dosepy')
        file_name_icon = pkg_resources.resource_filename('Dosepy', 'Icon/Icon.png')    #   Obtenido desde paquete Dosepy
        self.setWindowIcon(QIcon(file_name_icon))

        label_usuario = QLabel(self)
        label_usuario.setText('Ingrese clave de acceso:')
        label_usuario.move(80, 20)
        layout_principal.addWidget(label_usuario)
        label_usuario.setFont(QFont('Arial', 14))

        self.name_entry = QLineEdit(self)
        self.name_entry.setFont(QFont('Arial', 18))
        self.name_entry.setEchoMode(QLineEdit.Password)
        self.name_entry.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.name_entry.returnPressed.connect(self.cerrar_UI)
        self.name_entry.move(100, 70)
        layout_principal.addWidget(self.name_entry)
        layout_principal.setSpacing(10)
        self.show()

    def cerrar_UI(self):
        if self.name_entry.text() == 'self':
            #self.close()
            self.Us = 'P'
            self.window = VentanaPrincipal(self.Us)
        else:
            self.Us = 'O'
            self.window = VentanaPrincipal(self.Us)

    def menu_ayuda(self):
        """
        Crear un menú de ayuda
        """
        # Crear acciones para el menú "Herramientas"

        about_action_secundaria = QAction('Acerca de', self)
        about_action_secundaria.triggered.connect(self.about_ventana)

        licencia_action_secundaria = QAction('Licencia', self)
        licencia_action_secundaria.triggered.connect(self.licencia_ventana)

        # Crear barra del menu
        barra_menu_sec = self.menuBar()
        barra_menu_sec.setNativeMenuBar(False)

        # Agregar menú herramientas y su acción a la barra del menú

        ayuda_menu_sec = barra_menu_sec.addMenu('Ayuda')
        ayuda_menu_sec.addAction(about_action_secundaria)
        ayuda_menu_sec.addAction(licencia_action_secundaria)


    def about_ventana(self):
        if self.about_window_sec == None:
            self.about_window = About_Window()
        self.about_window.show()

    def licencia_ventana(self):
        if self.licencia_window_sec == None:
            self.licencia_window_sec = Licencia_Window()
        self.licencia_window_sec.show()

app = QApplication(sys.argv)
#windowA = Ventana_Secundaria() 
windowA = VentanaPrincipal('P')

sys.exit(app.exec())
