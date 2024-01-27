# Getting started

![Portada_Dosepy](../assets/Perfiles_1.png)

## Usign a GUI

The easiest way to use Dosepy is through a graphical user interface (GUI). Open a python interpreter (for example opening *Anaconda Prompt* and typing python) and import Dosepy.GUI as follows:

```python
import Dosepy.GUI
```

Dosepy has two pre-loaded with two dose distributions examples, to allow interaction with the available tools.

## Scripting

### Film calibration

Import libraries

```python
from Dosepy.tools.image import load
from pathlib import Path
```

Read the tiff file that will be used for film calibration and define the imparted doses.

```python
file_path = Path("/home/user/tif_files") / "some.tif"
cal_image = load(file_path, for_calib = True)

imparted_doses = [0, 0.5, 1, 1.5, 2, 3, 5, 8, 10]
```

Produce the calibration curve using the red channel, a roi size of 16 mm width, 8 mm height, and a rational function.

```python
cal = cal_image.get_calibration(doses = imparted_doses, channel = "R", roi = (16, 8), func = "RF")
cal.plot(color = 'red')
```

### Film to dose

Load another tif file

```python
verif_path = Path("/home/user/tif_files") / "other.tif"
verif = load(verif_path)
```

Apply the calibration curve

```python
dose_img = verif.to_dose(cal)
```

Show it

```python
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(ncols=1)

max_dose = np.percentile(dose_img.array, [99.9])[0]
pos = ax.imshow(dose_img.array, cmap='nipy_spectral')
pos.set_clim(-.05, max_dose)

# add the colorbar
fig.colorbar(pos, ax=ax)
plt.plot()
```

Get mean doses from central rois in each founded film.

```python
doses_in_central_rois = verif.doses_in_central_rois(cal, roi = (20, 8), show=True)
print(doses_in_central_rois)
```

Save the dose distribution as a tif file (in cGy)

```python
dose_img.save_as_tif("dose_in_tif_file")
```