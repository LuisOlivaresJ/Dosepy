import unittest

from Dosepy import image
import numpy as np

class TestGammaParameters(unittest.TestCase):

    def test_dose_ta_above(self):
        
        # All points differ by 4 %
        a = np.zeros((10, 10)) + 96  
        b = np.zeros((10, 10)) + 100
        D_ref = image.load(a, dpi = 75)
        D_eval = image.load(b, dpi = 75)

        # Execute the code being tested
        _, pass_rate = D_eval.gamma2D(D_ref, 3, 1)

        # With a dose to agreement 3 %, pass rate = 0.0
        self.assertEqual(first=pass_rate, second = 0.0)


    def test_dose_ta_below(self):

        # All points differ by 2 %
        a = np.zeros((10, 10)) + 98
        b = np.zeros((10, 10)) + 100
        D_ref = image.load(a, dpi = 75)
        D_eval = image.load(b, dpi = 75)

        # Execute the code being tested
        _, pass_rate = D_eval.gamma2D(D_ref, 3, 1)

        # With a dose to agreement 3 %, pass rate = 100.0
        self.assertEqual(first=pass_rate, second = 100.0)

    
    def test_dose_threshold(self):

        row = np.arange(10, 101, 10)
        a = np.tile(row, (10, 1))
        b = np.tile(row, (10, 1))
        D_ref = image.load(a, dpi = 75)
        D_eval = image.load(b, dpi = 75)
        """
        a = [
            [10, 20, ..., 90, 100],
	        [10, 20, ..., 90, 100],
	        ...,
	        [10, 20, ..., 90, 100],
            ]
        a.shape # (10, 10)
        """

        # Execute the code being tested
        gamma, _ = D_eval.gamma2D(D_ref, 3, 1, dose_threshold=21)

        # The positions where we have nan are the first and second columns.
        #nan_points = np.shape(np.where(np.isnan(gamma)))[1]
        nan_points = np.isnan(gamma).sum(where=True)

        self.assertEqual(nan_points, 20)

    def test_exclude_above(self):
        return

"""
A = [[1, 2, 3, ..., 9, 9],
	 [1, 2, 3, ..., 9, 9],
	 ...,
	 [1, 2, 3, ..., 9, 9]]
row = np.arange(0, 101, 10)
a = np.tile(row, (10, 1))
a[:, -1] = 90
b = np.tile(row, (10, 1))

img_a = image.load(a, dpi=75)
img_b = image.load(b, dpi=75)

img_b.gamma2D(
    reference=img_a,
    dose_ta=1,
    dist_ta=1,
    dose_threshold = 0,  
    )

print(a)
print(a.shape)
"""