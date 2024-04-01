# -*- coding: utf-8 -*-

"""

Última modificación: 12 Octubre 2022
@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico en Radioterapia, La Paz, Baja California Sur, México.

    Derechos Reservados (c) Luis Alfonso Olivares Jimenez 2021

"""

import sys
from importlib import resources

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QApplication, QPlainTextEdit
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class Licencia_Window(QWidget):
    """
    Ventana para mostrar licencia.
    """
    def __init__(self):
        super().__init__()
        #self.setStyleSheet("background-color: whitesmoke;")
        self.setStyleSheet("background-color: #1d1040;")
        self.setWindowTitle('Términos y condiciones')
        self.setFixedWidth(1100)
        self.setFixedHeight(850)

        self.iniciar_ventana()


    def iniciar_ventana(self):

        layout_padre_V = QVBoxLayout()

        file_name_logo = str(resources.files("Dosepy") / "Icon" / "Logo_Dosepy.png")
        logo = QPixmap(file_name_logo)
        #logo.scaled(0.5, 0.5) #Qt.KeepAspectRatio)
        label_logo = QLabel(self)
        label_logo.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        label_logo.setPixmap(logo)
        label_logo.setStyleSheet(
            "border-radius: 15px;" +
            "margin-top: 15px;" +
            "margin-bottom: 15px;" +
            "margin-left: 80px;" +
            "margin-right: 80px;"
        )

        layout_padre_V.addWidget(label_logo)

        label_version = QLabel(self)
        label_version.setText('PROPRIETARY LICENSE')
        label_version.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        label_version.setStyleSheet(
            "margin-top: 10px;" +
            "font-size: 22px;" +
            "margin-bottom: 5px;" +
            "color: 'whitesmoke';"
        )

        label_derechos = QLabel(self)
        label_info_licencia = QPlainTextEdit()
        label_info_licencia.appendPlainText('''
Derechos Reservados (c) Luis Alfonso Olivares Jimenez 2021

DECLARACIONES

DOSEPY se encuentra registrado ante el Instituto Nacional del Derecho de Autor, México.
Número de registro: 03-2021-093012460400-01

Luis Alfonso Olivares Jiménez es el autor y titular del derecho moral de la obra DOSEPY.

Condiciones de uso 

Toda persona tiene acceso a la lectura y uso del código con fines académicos o de enseñanza. 
Sin embargo, para el uso clínico del programa se requiere contar con una licencia 
(disponible próximamente), conocida como “Acuerdo de licencia de usuario final” 
(EULA, por sus siglas en inglés), así como contratos que garanticen el cumplimiento 
de la legislación de cada país. 

El código o software derivado, tales como arreglos, compendios, ampliaciones, traducciones, adaptaciones,
paráfrasis, compilaciones, colecciones y transformaciones del software DOSEPY, podrán ser explotadas
cuando hayan sido autorizadas por el titular del derecho patrimonial sobre la obra DOSEPY,
previo consentimiento del titular del derecho moral, en los casos previstos en la Fracción III
del Artículo 21 de la Ley Federal del Derecho de Autor.

GARANTÍA

El software Dosepy se ofrece sin ninguna garantía de cualquier tipo. Su uso es responsabilidad del usuario.

Para mayor información contactar al correo electrónico dosepy@gmail.com. 

''')
        label_info_licencia.setReadOnly(True)
        #label_info_licencia.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        label_info_licencia.setStyleSheet(
            "margin-top: 5px;" +
            "font-size: 17px;" +
            "margin-bottom: 20px;" +
            "color: 'whitesmoke';"
        )

        link_label = QLabel(self)
        link_label.setText(
            "<a href=\"https://dosepy.readthedocs.io/en/latest/intro.html\">Home page</a>"
        )
        #link_label.setTextFormat(Qt.RichText)
        #link_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        link_label.setOpenExternalLinks(True)
        #link_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        link_label.setStyleSheet(
            "font-size: 17px;" +
            "margin-bottom: 15px;" +
            "color: 'whitesmoke';"
        )

        layout_padre_V.addWidget(label_version)
        layout_padre_V.addWidget(label_info_licencia)
        layout_padre_V.addWidget(link_label)


        self.setLayout(layout_padre_V)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Licencia_Window()
    window.show()
    sys.exit(app.exec())
