import logging

logging.basicConfig(
    filename="calibration.log",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
    encoding="utf-8",
    filemode="w",
    )

from Dosepy.calibration import CalibrationLUT
from Dosepy.image import load, TiffImage

import numpy as np
import matplotlib.pyplot as plt

path_file = "film20240620_002.tif"
img = load(path_file)
cal = CalibrationLUT(img)
cal.create_central_rois((180,8))
cal.set_doses([0, 1, 2, 4, 6.5, 9.5])
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


cal.plot_lateral_response(channel = "red")

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
position = 5
print(f"Lateral position: {position}")

channel = "red"
print("Channel: {channel}")

#print(cal._get_calibration_positions())
intensities, std = cal._get_intensities(
    lateral_position = position,
    channel = channel,
    )

print(f"Lateral doses at position: {position}")
print(cal._get_lateral_doses(position = position))

print("Intensities normalized")
print(intensities/intensities[0])

print("Intensities")
print(intensities)
logging.debug(f"Intensities: {intensities}")

print("Standard deviation")
print(std)

fig, axes = plt.subplots(1, 2)

cal.plot_fit(
    fit_type="rational",
    position=position,
    channel=channel,
    ax=axes[0],
    )

cal.plot_dose_fit_uncertainty(
    position=position,
    channel=channel,
    fit_function="rational",
    ax=axes[1],
)

plt.show()