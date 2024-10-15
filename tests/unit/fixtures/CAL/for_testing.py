from Dosepy.calibration import CalibrationLUT
from Dosepy.image import load, TiffImage

import matplotlib.pyplot as plt

path_file = "film20240620_002.tif"
img = load(path_file)
cal = CalibrationLUT(img)
cal.create_central_rois((180,8))
cal.compute_lateral_lut()

position = 0

print(cal.lut[(position, 0)]["I_red"])
print(cal.lut[(position, 0)]["S_red"])

print(cal.lut[(position, 0)]["I_green"])
print(cal.lut[(position, 0)]["S_green"])

print(cal.lut[(position, 0)]["I_blue"])
print(cal.lut[(position, 0)]["S_blue"])

print(cal.lut[(position, 0)]["I_mean"])
print(cal.lut[(position, 0)]["S_mean"])

#print(cal.lut["lateral_limits"])
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
cal._plot_rois(ax)
cal.plot_lateral_response()
plt.show()