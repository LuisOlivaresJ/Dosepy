from Dosepy.calibration import LUT
import matplotlib.pyplot as plt

cal = LUT.from_yaml_file("calibration.yaml")

cal.plot_fit(
    fit="polynomial",
    position=0,
    channel="red"
    )

plt.show()