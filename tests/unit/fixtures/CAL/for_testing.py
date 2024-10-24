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

#print(cal.lut)

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

fit_function = "polynomial"

#print(cal._get_calibration_positions())
intensities, std = cal._get_intensities(
    lateral_position = position,
    channel = channel,
    )

print(f"Lateral doses at position: {position}")
print(cal._get_lateral_doses(position = position))

#print("Intensities normalized")
#print(intensities/intensities[0])

#print("Intensities")
#print(intensities)
logging.debug(f"Intensities: {intensities}")

print("Standard deviation without filter")
print(std)

# Without filter

fig, axes = plt.subplots(1, 2)

cal.plot_fit(
    fit_type=fit_function,
    position=position,
    channel=channel,
    ax=axes[0],
    )

cal.plot_dose_fit_uncertainty(
    position=position,
    channel=channel,
    fit_function=fit_function,
    ax=axes[1],
)

# With filter

fig_filter, axes_filter = plt.subplots(1, 2)

cal_filter = CalibrationLUT(img)
cal_filter.create_central_rois((180,8))
cal_filter.set_doses([0, 1, 2, 4, 6.5, 9.5])
cal_filter.set_beam_profile(beam_profile="BeamProfile.csv")
cal_filter.compute_lateral_lut(filter = 3)


cal_filter.plot_fit(
    fit_type=fit_function,
    position=position,
    channel=channel,
    ax=axes_filter[0],
    )

cal_filter.plot_dose_fit_uncertainty(
    position=position,
    channel=channel,
    fit_function=fit_function,
    ax=axes[1],
    alpha = 0.5,
)

#cal_filter.plot_lateral_response(channel = "red")

intensities_filter, std_filter = cal_filter._get_intensities(
    lateral_position = position,
    channel = channel,
    )

print("Filter")

print("Intensities")
print(intensities)
print(intensities_filter)
print("Standard deviation ")
print(std)
print(std_filter)

plt.show()