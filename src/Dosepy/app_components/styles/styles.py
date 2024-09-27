from PySide6.QtCore import Qt, QSize
from enum import Enum

# Sizes

class Size(Enum):
    ZERO = 0
    #SMALL = "0.6em"
    MEDIUM = QSize(25, 25)
    #DEFAULT = "1.1em"
    MAIN_BUTTON = QSize(150, 50)
    #LARGE = "1.6em"
    #BIG = "1.8em"

class SizeButton(Enum):
    TOOL = QSize(25, 25)
