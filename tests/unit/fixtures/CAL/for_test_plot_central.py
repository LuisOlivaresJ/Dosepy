from Dosepy.calibration import LUT, _get_dose_from_fit

import numpy as np
import matplotlib.pyplot as plt
from Dosepy.image import load

path_file = "film20240620_002.tif"
img = load(path_file)
cal = LUT(img)
cal.set_central_rois((8, 8), show = True)
cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
#cal.compute_central_lut(filter = 3)
cal.compute_central_lut()

cal.plot_fit()
cal.plot_dose_fit_uncertainty(0, "red", "polynomial")

cal.to_yaml_file("calibration.yaml")

print(cal.lut)

plt.show()