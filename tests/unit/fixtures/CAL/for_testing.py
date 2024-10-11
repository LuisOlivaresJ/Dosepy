from Dosepy.calibration import CalibrationLUT
from Dosepy.image import load, TiffImage

path_file = "film20240620_002.tif"
img = load(path_file)
cal = CalibrationLUT(img)
cal.create_central_rois((180,8))
cal.compute_lateral_lut()

print(cal.lut[(2, 0)]["I_red"])
print(cal.lut[(2, 0)]["S_red"])

print(cal.lut[(2, 0)]["I_green"])
print(cal.lut[(2, 0)]["S_green"])

print(cal.lut[(2, 0)]["I_blue"])
print(cal.lut[(2, 0)]["S_blue"])

print(cal.lut[(2, 0)]["I_mean"])
print(cal.lut[(2, 0)]["S_mean"])

#print(cal.lut["lateral_limits"])