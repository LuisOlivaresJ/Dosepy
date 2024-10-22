from Dosepy.calibration import CalibrationLUT
from Dosepy.image import load, TiffImage

import matplotlib.pyplot as plt

path_file = "film20240620_002.tif"
img = load(path_file)
cal = CalibrationLUT(img)
cal.create_central_rois((180,8))
cal.set_doses([0, 2, 4, 6, 8, 10])
cal.set_beam_profile(beam_profile="BeamProfile.csv")
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
#fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#cal._plot_rois(ax)
#cal.plot_lateral_response(channel = "red")
#cal.plot_lateral_response(channel = "green")
#cal.plot_lateral_response(channel = "blue")

#print(cal._get_lateral_doses(position = -105))

"""
cal.plot_fit(
    fit_type="rational",
    position=0,
    channel="red",
    )
"""
cal.plot_fit(
    fit_type="rational",
    position=0,
    channel="green",
    )

plt.show()
