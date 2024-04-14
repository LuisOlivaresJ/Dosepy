import unittest

from Dosepy import image
import numpy as np

class TestGammaParameters(unittest.TestCase):

    def test_dose_ta_above(self):
        
        a = np.zeros((10, 10)) + 96  # 
        b = np.zeros((10, 10)) + 100

        D_ref = image.load(a, dpi = 75)   # Reference dose distribution
        D_eval = image.load(b, dpi = 75)  # Evaluated dose distribution

        gamma_distribution, pass_rate = D_eval.gamma2D(D_ref, 3, 1)
        self.assertEqual(first=pass_rate, second = 0.0)

    def test_dose_ta_below(self):

        a = np.zeros((10, 10)) + 98
        b = np.zeros((10, 10)) + 100

        D_ref = image.load(a, dpi = 75)   # Reference dose distribution
        D_eval = image.load(b, dpi = 75)  # Evaluated dose distribution

        gamma_distribution, pass_rate = D_eval.gamma2D(D_ref, 3, 1)
        self.assertEqual(first=pass_rate, second = 100.0)

    def test_exclude_above(self):
        return

"""
row = np.arange(1, 11)
a = np.tile(row, (10, 1))
a[:, -1] = 9
b = np.tile(row, (10, 1))

img_a = image.load(a, dpi=1)
img_b = image.load(b, dpi=1)

img_b.gamma2D(
    reference=img_a,
    dose_ta=1,
    dist_ta=1,    
    )

print(a)
print(a.shape)
"""