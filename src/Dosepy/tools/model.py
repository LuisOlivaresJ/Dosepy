"""Functions used as a model. VMC pattern."""

#from Dosepy.tools.image import _is_RGB
from image import _is_RGB

class Model:
    def __init__(self):
        pass

    def valid_tif_files(self, files: list) -> bool:
        print("Inside model valid tif files")
        print([file for file in files])
        list_files = [_is_RGB(file) for file in files]
        print(list_files)
        return all([_is_RGB(file) for file in files])