# -*- coding: utf-8 -*-
"""
Entrance and setup for GUI
"""

import sys
import os

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QApplication,
    QMessageBox,
    QMainWindow,
    QLabel,
    QLineEdit,
)
from PyQt6.QtGui import QIcon, QFont, QAction
from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import QFileDialog, QInputDialog
from relative_dose_1d.tools import build_from_array_and_step
from relative_dose_1d.GUI_tool import plot

import numpy as np
from .gui_components.Bloque_gamma import Bloque_gamma
# For testing, import from local PC
# from GUILayouts.Bloque_gamma import Bloque_gamma
from .gui_components.Bloque_Imagenes import Bloque_Imagenes
# For testing, import from local PC
# from GUILayouts.Bloque_Imagenes import Bloque_Imagenes
#import dose as dp
import Dosepy.old as dp
import matplotlib as mpl
from importlib import resources

from .gui_components.film_to_doseGUI import Film_to_Dose_Window
# from GUILayouts.film_to_doseGUI import Film_to_Dose_Window
from .gui_components.about_window import About_Window
# from GUILayouts.about_window import About_Window
from .gui_components.licencia_window import Licencia_Window
# from GUILayouts.licencia_window import Licencia_Window
# ---------------------------------------------


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: whitesmoke;")
        self.setWindowTitle('Dosepy')
        file_name_icon = str(resources.files("Dosepy") / "Icon" / "Icon.png")
        self.setWindowIcon(QIcon(file_name_icon))
        self.setGeometry(150, 100, 1300, 800)

        self.setup_main_widget()
        self.setup_menu()

        # State of secondary windows
        self.film_to_dose_window = None
        self.about_window = None
        self.licencia_window_sec = None

        # Format validation
        self.formatos_ok = False

        self.show()

    def setup_main_widget(self):
        """
        Main app body
        """

        body = QWidget()
        main_layout = QVBoxLayout()
        body.setLayout(main_layout)
        self.setCentralWidget(body)

        self.box_for_gamma = Bloque_gamma()
        main_layout.addWidget(self.box_for_gamma)

        self.box_for_gamma.Refer_button.clicked.connect(self.read_reference_file)
        self.box_for_gamma.Eval_button.clicked.connect(self.read_evaluation_file)
        self.box_for_gamma.Eval_button.clicked.connect(self.show_dose_distribution)
        self.box_for_gamma.Calcular_Button.clicked.connect(self.gamma_calculation)

        self.box_images = Bloque_Imagenes()
        main_layout.addWidget(self.box_images)

        self.box_images.boton_recortar_Izq.clicked.connect(self.cut_image)
        self.box_images.compare_button.clicked.connect(self.Compare_profiles)

    def setup_menu(self):
        """
        App menu
        """

        # Crear barra del menu
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        # Setup actions for menu tools
        film_to_dose_action = QAction('Film dosimetry', self)
        film_to_dose_action.setShortcut('Ctrl+F')
        film_to_dose_action.triggered.connect(self.film_to_dose)

        about_action = QAction('About...', self)
        about_action.triggered.connect(self.about_window)

        licencia_action = QAction('License', self)
        licencia_action.triggered.connect(self.license_window)

        # Agregar menú herramientas y su acción a la barra del menú
        herram_menu = menu_bar.addMenu('Tools')
        herram_menu.addAction(film_to_dose_action)
        about_menu = menu_bar.addMenu('Help')
        about_menu.addAction(about_action)
        about_menu.addAction(licencia_action)

    ######################################################################
    #   Slots for menu actions

    def film_to_dose(self):
        if self.film_to_dose_window is None:
            self.film_to_dose_window = Film_to_Dose_Window()
        self.film_to_dose_window.show()

    def about_window(self):
        if self.about_window is None:
            self.about_window = About_Window()
        self.about_window.show()

    def license_window(self):
        if self.licencia_window_sec is None:
            self.licencia_window_sec = Licencia_Window()
        self.licencia_window_sec.show()

    ######################################################################
    #   Funciones para botones

    ######################################################################
    #   Funciones para botones que leen un archivo

    def read_reference_file(self):
        file_name_Referencia, _ = QFileDialog.getOpenFileName()
        _, extension = os.path.splitext(file_name_Referencia)

        if file_name_Referencia:    # Se obtuvo algún archivo?
            if extension == '.csv':

                resolution, ok_resol = QInputDialog.getDouble(
                    self,
                    "Resolución",
                    "Resolución [mm/punto]",
                    decimals=5)
                Refer_npy = np.genfromtxt(file_name_Referencia, delimiter=',')
                self.box_images.D_ref = dp.Dose(Refer_npy, resolution)
                self.box_for_gamma.Refer_button.setStyleSheet(
                    "background-color: rgb(88,200,138)"
                )

                self.box_for_gamma.Eval_button.setEnabled(True)

                """
                elif extension == '.dcm':
                    self.Refer_npy = dp.from_dicom('file_name_Referencia').array
                    self.Refer_button.setStyleSheet(
                        "background-color: rgb(88,200,138)"
                    )
                    self.Resolution.setText(
                        str(dp.from_dicom(file_name_Referencia).resolution)
                    )
                """

            else:
                QMessageBox().critical(
                    self,
                    "Error",
                    "Formato no válido.",
                    QMessageBox.Ok,
                    QMessageBox.Ok
                )
                print('Formato no valido')

    def read_evaluation_file(self):
        file_name_Evaluacion, _ = QFileDialog.getOpenFileName()
        _, extension = os.path.splitext(file_name_Evaluacion)

        if file_name_Evaluacion:
            if extension == '.dcm':
                self.box_images.D_eval = dp.from_dicom(file_name_Evaluacion)
                """
                self.Resolution.setText(
                    str(dp.from_dicom(file_name_Evaluacion).resolution)
                )
                #self.Resolution.setReadOnly(True)
                """

                if self.box_images.D_ref.array.shape != self.box_images.D_eval.array.shape:
                    QMessageBox().critical(
                        self,
                        "Error",
                        """No es posible el análisis con matrices de diferente tamaño.\n
                        Referencia: {}\n
                        A evaluar: {}""".format(
                            self.box_images.D_ref.array.shape,
                            self.box_images.D_eval.array.shape
                        ),
                        QMessageBox.Ok, QMessageBox.Ok
                    )

                else:
                    self.box_for_gamma.Eval_button.setStyleSheet(
                        "background-color: rgb(88,200,138)"
                    )
                    self.formatos_ok = True

            elif extension == '.csv':

                resolution, ok_resol = QInputDialog.getDouble(self,
                                                              "Resolución",
                                                              "Resolución [mm/punto]",
                                                              decimals=5)
                Eval_npy = np.genfromtxt(file_name_Evaluacion, delimiter=',')
                self.box_images.D_eval = dp.Dose(Eval_npy, resolution)

                if self.box_images.D_eval.array.shape != self.box_images.D_ref.array.shape:
                    QMessageBox().critical(
                        self,
                        "Error",
                        """No es posible el análisis con matrices de diferente tamaño.\n
                        Referencia: {}\n
                        A evaluar: {}""".format(
                            self.box_images.D_ref.array.shape,
                            self.box_images.D_eval.array.shape
                        ),
                        QMessageBox.Ok, QMessageBox.Ok
                    )

                else:
                    self.box_for_gamma.Eval_button.setStyleSheet(
                        "background-color: rgb(88,200,138)"
                    )
                    self.formatos_ok = True
            # print(self.file_name_Evaluacion)

    def show_dose_distribution(self):
        if self.formatos_ok:
            self.box_images.Mpl_Izq.Img(self.box_images.D_ref)
            self.box_images.Mpl_Izq.Colores(self.box_images.D_eval.array)

            self.box_images.Mpl_Der.Img(self.box_images.D_eval)
            self.box_images.Mpl_Der.Colores(self.box_images.D_eval.array)

            self.box_images.Mpl_perfiles.ax.clear()
            self.box_images.Mpl_perfiles.set_data_and_plot(
                self.box_images.D_ref.array,
                self.box_images.D_eval.array,
                self.box_images.Mpl_Izq.circ
            )

            self.box_images.Mpl_Izq.fig.canvas.draw()
            self.box_images.Mpl_Der.fig.canvas.draw()

            self.box_images.Mpl_perfiles.fig.canvas.draw()

        else:
            self.displayMessageBox()

    def gamma_calculation(self):
        D_ref = self.box_images.D_ref
        D_eval = self.box_images.D_eval
        g, p = D_eval.gamma2D(
            D_ref,
            float(self.box_for_gamma.Toler_dosis.text()),
            float(self.box_for_gamma.Toler_dist.text()),
            float(self.box_for_gamma.Umbral_dosis.text())
        )

        self.box_for_gamma.Mpl_Histograma.Mostrar_Histograma(g)

        self.box_for_gamma.Indice_gamma_porcentaje_Label.setText(
            'Porcentaje de aprobación: ' + str(round(p, 1)) + '%'
        )
        self.box_for_gamma.Indice_gamma_promedio_Label.setText(
            'Índice gamma promedio: ' + str(round(np.mean(g[~np.isnan(g)]), 1))
        )
        """
        # self.box_for_gamma.Indice_gamma_maximo_Label.setText(
            'Máximo: ' + str(round(np.max(g[~np.isnan(g)]), 1))
        )
        # self.box_for_gamma.Indice_gamma_mediana_Label.setText(
            'Mediana: ' + str(round(np.median(g[~np.isnan(g)]), 1))
        )
        """

        self.box_for_gamma.Mpl_Img_gamma.ax1.clear()
        self.box_for_gamma.Mpl_Img_gamma.ax2.clear()
        _Dg = dp.Dose(g, float(self.box_images.D_ref.resolution))
        self.box_for_gamma.Mpl_Img_gamma.Img(_Dg)
        self.box_for_gamma.Mpl_Img_gamma.ax2.get_yaxis().set_visible(True)
        self.box_for_gamma.Mpl_Img_gamma.ax1.set_title(
            'Distribución gamma',
            fontsize=11
        )
        self.box_for_gamma.Mpl_Img_gamma.Colores(g[~np.isnan(g)])

        viridis = mpl.cm.get_cmap('viridis', 256)
        hot = mpl.cm.get_cmap('hot', 256)

        bounds_gamma = np.linspace(0, 1.5, 16)
        norm_gamma = mpl.colors.BoundaryNorm(boundaries=bounds_gamma, ncolors=16)

        new_viridis = viridis(np.linspace(0, 1, 10))
        new_hot = np.flip(hot(np.linspace(0, 1, 40)), 0)
        new_hot = new_hot[20:26, :]
        new_color_gamma = np.vstack((new_viridis, new_hot))
        new_cmp_gamma = mpl.colors.ListedColormap(new_color_gamma)
        self.box_for_gamma.Mpl_Img_gamma.mplI.set_norm(norm_gamma)
        self.box_for_gamma.Mpl_Img_gamma.mplI.set_cmap(new_cmp_gamma)

        self.box_for_gamma.Mpl_Histograma.fig.canvas.draw()
        self.box_for_gamma.Mpl_Img_gamma.fig.canvas.draw()

    def cut_image(self):

        xi = int(self.box_images.Mpl_Izq.Rectangle.get_x())
        width = int(self.box_images.Mpl_Izq.Rectangle.get_width())
        yi = int(self.box_images.Mpl_Izq.Rectangle.get_y())
        height = int(self.box_images.Mpl_Izq.Rectangle.get_height())

        npI_Izq = self.box_images.Mpl_Izq.npI[yi: yi + height, xi: xi + width]
        npI_Der = self.box_images.Mpl_Der.npI[yi: yi + height, xi: xi + width]

        self.box_images.D_ref = dp.Dose(
            npI_Izq, self.box_images.D_ref.resolution
        )
        self.box_images.D_eval = dp.Dose(
            npI_Der, self.box_images.D_eval.resolution
        )

        self.box_images.Mpl_Izq.Img(self.box_images.D_ref)
        self.box_images.Mpl_Der.Img(self.box_images.D_eval)

        self.box_images.Mpl_Izq.Colores(npI_Der)
        self.box_images.Mpl_Der.Colores(npI_Der)

        self.box_images.Mpl_Izq.Cross_Hair_on()
        self.box_images.Mpl_Der.Cross_Hair_on()

        self.box_images.Mpl_perfiles.set_data_and_plot(
            npI_Izq,
            npI_Der,
            self.box_images.Mpl_Izq.circ
        )

        self.box_images.Mpl_Izq.ROI_Rect_off()
        self.box_images.boton_recortar_Izq.setEnabled(False)
        self.box_images.boton_roi.setChecked(False)

    def Compare_profiles(self):

        resolution = self.box_images.D_ref.resolution

        D_profile_ref = build_from_array_and_step(
            self.box_images.Mpl_perfiles.perfil_horizontal_ref,
            resolution
            )
        D_profile_eval = build_from_array_and_step(
            self.box_images.Mpl_perfiles.perfil_horizontal_eval,
            resolution
            )

        self.profile_plot = plot(D_profile_ref, D_profile_eval)
        self.profile_plot.setWindowTitle("Horizontal profile")
        self.profile_plot.open_file_button.setEnabled(False)
        self.profile_plot.clear_button.setEnabled(False)
        self.profile_plot.show()

######################################################################
#   Ventanas para mensajes

    def displayMessageBox(self):
        """
        Si la variable self.formatos_ok es True, los archivos
        para las distribuciones de dosis se cargaron correctamente.
        En caso contrario se emite un mensaje de error.
        """
        QMessageBox().critical(
            self,
            "Error",
            "Error con la lectura de archivos.",
            QMessageBox.Ok,
            QMessageBox.Ok
        )


class Ventana_Secundaria(QMainWindow):
    """Entry"""
    def __init__(self):
        super().__init__()

        self.about_window_sec = None
        self.licencia_window_sec = None

        self.iniciarUI()
        self.menu_ayuda()
        # self.setCentralWidget(cuerpo_sec)

    def iniciarUI(self):

        cuerpo = QWidget()
        layout_principal = QVBoxLayout()
        cuerpo.setLayout(layout_principal)
        self.setCentralWidget(cuerpo)

        self.setStyleSheet("background-color: whitesmoke;")
        self.setWindowTitle('Dosepy')

        file_name_icon = str(resources.files("Dosepy") / "Icon" / "Icon.png")

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
            # self.close()
            self.Us = 'P'
            self.window = MainWindow(self.Us)
        else:
            self.Us = 'O'
            self.window = MainWindow(self.Us)

    def menu_ayuda(self):
        """
        Crear un menú de ayuda
        """
        # Crear acciones para el menú "Herramientas"

        about_action_secundaria = QAction('Acerca de', self)
        about_action_secundaria.triggered.connect(self.about_window)

        licencia_action_secundaria = QAction('Licencia', self)
        licencia_action_secundaria.triggered.connect(self.license_window)

        # Crear barra del menu
        menu_bar_sec = self.menuBar()
        menu_bar_sec.setNativeMenuBar(False)

        # Agregar menú herramientas y su acción a la barra del menú

        ayuda_menu_sec = menu_bar_sec.addMenu('Ayuda')
        ayuda_menu_sec.addAction(about_action_secundaria)
        ayuda_menu_sec.addAction(licencia_action_secundaria)

    def about_window(self):
        if self.about_window_sec is None:
            self.about_window = About_Window()
        self.about_window.show()

    def license_window(self):
        if self.licencia_window_sec is None:
            self.licencia_window_sec = Licencia_Window()
        self.licencia_window_sec.show()


app = QApplication(sys.argv)
# windowA = Ventana_Secundaria()
windowA = MainWindow()

sys.exit(app.exec())
