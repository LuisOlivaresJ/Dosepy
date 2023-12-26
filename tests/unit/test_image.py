import unittest

from pathlib import Path

from Dosepy.tools import image

cwd = Path(__file__).parent

class TestReadableImage(unittest.TestCase):
    def test_bad_file(self):
        #cwd = Path(__file__).parent
        file_path = cwd / "fixtures" / "an_incorrect_image_file.md"
        with self.assertRaises(TypeError):
            img = image.load(file_path)

    def test_bad_format(self):
        #cwd = Path(__file__).parent
        file_path = cwd / "fixtures" / "calibracion.png"
        self.assertFalse(image._is_tif_file(file_path))

    def test_bad_RGB(self):
        file_path = cwd / "fixtures" / "red_channel.tif"
        self.assertFalse(image._is_RGB(file_path))

    def test_readable_image(self):
        #cwd = Path(__file__).parent
        file_path = cwd / "fixtures" / "image.tif"
        img = image.load(file_path)
        self.assertIsInstance(img, image.TiffImage)

    #TO DO: A test function for TiffImage.dpi