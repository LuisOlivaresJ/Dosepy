# -*- coding: utf-8 -*-
"""
NAME
    Resolution Function

DESCRIPTION
    Script to match the number of rows and columns between two arrays based on physical resolution.

"""

"""
@author:
    Luis Alfonso Olivares Jimenez
"""

import numpy as np

def _points_to_average(res_A, res_B, points):
	"""
	Function to generate a list where each element represents 
	the number of sub-points that must be averaged to equalize the given resolutions.

	Parameters
	----------
		res_A : float
			Spatial resolution given as the distance, in mm, between two points.

		res_B : float
			Spatial resolution given as the distance, in mm, between two points.

		points: int
			Number of points.

	Returns
	-------
		out : ndarray
			List with the numbers to be averaged for array size reduction.

	"""
	if res_A > res_B:
		highest_resolution = res_A
		lowest_resolution = res_B

	else:
		highest_resolution = res_B
		lowest_resolution = res_A

	remaining_points = points  # Remaining points to be averaged. It starts with all the points.
	points_to_average = int(highest_resolution // lowest_resolution)
	residue = highest_resolution % lowest_resolution
	points_to_average_list = []  # Corrected variable name for clarity
	accumulated_residue = residue

	while remaining_points >= points_to_average:
		if accumulated_residue > lowest_resolution/2:
			points_to_average_list.append(points_to_average + 1)
			remaining_points -= (points_to_average + 1)
			accumulated_residue = accumulated_residue - ( (points_to_average + 1)*lowest_resolution - highest_resolution )
		else:
			points_to_average_list.append(points_to_average)
			remaining_points -= points_to_average
			accumulated_residue += residue

	if remaining_points > 0:
		points_to_average_list.append(remaining_points)

	return points_to_average_list

def equate_resolution(array, array_resolution, target_resolution):
	"""
	Reduces the array size so that its new spatial resolution equals the target resolution. 
	The algorithm averages a number of points given by array_resolution // target_resolution

	Parameters
	----------
	array : ndarray
		The array that needs its size reduced.

	array_resolution : float
		Spatial resolution of the array, in millimeters per point.

	target_resolution : float
		Target spatial resolution, in millimeters per point.
		
	Returns
	-------
	ndarray
		Array with a reduced size.

	Examples
	--------
	**Example 1**

	Let A be an array of (100, 100) with a 0.1 mm/point spatial resolution, and B another 
	array of (10, 10) with a 1 mm/point resolution. To perform a point-by-point comparison, we need a
	new (10, 10) representative array of A::

	>>> import numpy as np
	>>> A = np.random.rand(100, 100)  # With an associated spatial resolution of 0.1 mm/point.
	>>> new_A = equate_resolution(A, 0.1, 1)
	

	**Example 2**

	Let A and B be two arrays of size (2362 x 2362) and (256 x 256), with spatial resolutions of 0.08467 mm/point and 0.78125 mm/point, respectively.

	The physical dimension of array A is 200.06 mm
	(2362 points * 0.08467 mm/point = 200.06 mm).
	The physical dimension of array B is 199.99 mm
	(256 points * 0.78125 mm/point = 199.99 mm).

	To reduce the size of array A to match the size of
	array B, the equate_resolution function is used as::

	>>> import Dosepy.tools.resol as resol
	>>> import numpy as np
	>>> A = np.zeros((2362, 2362))
	>>> C = resol.equate_resolution(A, 0.08467, 0.78125)
	>>> C.shape
	>>> (256, 256)
	"""

	list_points_column = _points_to_average(array_resolution, target_resolution, array.shape[1])
	list_points_row = _points_to_average(array_resolution, target_resolution, array.shape[0])
	new_reduced_array = np.zeros((len(list_points_row), len(list_points_column)))
	f = 0  # Row counter
	for i in np.arange(len(list_points_row)):
		c = 0  # Column counter
		for j in np.arange(len(list_points_column)):
			temporal = array[f : f + list_points_row[i], c : c + list_points_column[j]]
			new_reduced_array[i,j] = np.mean(temporal)
			c = c + list_points_column[j]
		f = f + list_points_row[i]
	return new_reduced_array


def main():  # For testing
	a = _points_to_average(6, 20, 13)
	print(a)

	A = np.zeros((2362, 2362))
	C = equate_resolution(A, 0.08467, 0.78125)
	print(C.shape)  # Expected output: (256, 256)

if __name__ == "__main__":
	main()
