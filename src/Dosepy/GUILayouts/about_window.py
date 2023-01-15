# -*- coding: utf-8 -*-

"""
@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico en Radioterapia, La Paz, Baja California Sur, México.

    Derechos Reservados (c) Luis Alfonso Olivares Jimenez 2021

"""

import sys
import pkg_resources

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QApplication
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt


class About_Window(QWidget):
    """
    Ventana para mostrar información del programa.
    """
    def __init__(self):
        super().__init__()
        #self.setStyleSheet("background-color: whitesmoke;")
        self.setStyleSheet("background-color: #1d1040;")
        self.setWindowTitle('Acerca de')

        self.iniciar_ventana()


    def iniciar_ventana(self):

        layout_padre_V = QVBoxLayout()

        file_name_logo = pkg_resources.resource_filename('Dosepy', 'Icon/Logo_Dosepy.png')
        logo = QPixmap(file_name_logo)
        #logo.scaled(0.5, 0.5) #Qt.KeepAspectRatio)
        label_logo = QLabel(self)
        label_logo.setAlignment(Qt.AlignCenter)
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
        label_version.setText('Versión 0.3.5')
        label_version.setAlignment(Qt.AlignCenter)
        label_version.setStyleSheet(
            "margin-top: 10px;" +
            "font-size: 22px;" +
            "margin-bottom: 5px;" +
            "color: 'whitesmoke';"
        )

        label_derechos = QLabel(self)
        label_derechos.setText('Derechos Reservados (c) \n Luis Alfonso Olivares Jiménez 2021')
        label_derechos.setAlignment(Qt.AlignCenter)
        label_derechos.setStyleSheet(
            "margin-top: 5px;" +
            "font-size: 17px;" +
            "margin-bottom: 20px;" +
            "color: 'whitesmoke';"
        )

        link_label = QLabel(self)
        link_label.setText(
            "<a href=\"https://luisolivaresj.github.io/Dosepy//\">Documentación</a>"
        )
        link_label.setTextFormat(Qt.RichText)
        #link_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        link_label.setOpenExternalLinks(True)
        link_label.setAlignment(Qt.AlignCenter)
        link_label.setStyleSheet(
            "font-size: 17px;" +
            "margin-bottom: 15px;" +
            "color: 'whitesmoke';"
        )

        layout_padre_V.addWidget(label_version)
        layout_padre_V.addWidget(label_derechos)
        layout_padre_V.addWidget(link_label)


        self.setLayout(layout_padre_V)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = About_Window()
    window.show()
    sys.exit(app.exec_())













#
