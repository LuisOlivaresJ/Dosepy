from PySide6.QtCore import Qt, QSize
from enum import Enum

# Sizes

class Size(Enum):
    ZERO = 0
    #SMALL = "0.6em"
    #MEDIUM = "0.9em"
    #DEFAULT = "1.1em"
    MAIN_BUTTON = QSize(150, 50)
    #LARGE = "1.6em"
    #BIG = "1.8em"

