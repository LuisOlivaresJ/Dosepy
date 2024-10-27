from Dosepy.calibration import CalibrationLUT
import matplotlib.pyplot as plt

cal = CalibrationLUT.from_yaml_file("calibration.yaml")

cal.plot_fit(fit_type="polynomial", position=0, channel="red")
plt.show()