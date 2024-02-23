"""Functions used as a model. VMC pattern."""

#from Dosepy.tools.image import _is_RGB
from image import _is_RGB, _is_image_file, load_multiples, TiffImage, load
import imageio.v3 as iio

class Model:
    def __init__(self):
        pass


    def are_valid_tif_files(self, files: list) -> bool:
        return all([_is_image_file(file) and _is_RGB(file) for file in files])
        

    def are_files_equal_shape(self, files: list) -> bool:
        first_img_shape = self.props = iio.improps(files[0]).shape
        for file in files:
            if iio.improps(file).shape != first_img_shape:
                return False
        return True
    
    def load_files(self, files: list) -> TiffImage:
        if len(files) == 1:
            return load(files[0])
        
        elif len(files) > 1:
            return load_multiples(files)