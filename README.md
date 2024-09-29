![Dosepy-Logo](https://dosepy.readthedocs.io/en/latest/_static/Logo_Dosepy.png)

![PyPI - Version](https://img.shields.io/pypi/v/Dosepy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Dosepy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/Dosepy)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/LuisOlivaresJ/Dosepy)

# Welcome to Dosepy

Main [documentation](https://dosepy.readthedocs.io/en/latest/intro.html)

Dosepy is an open source application to perform radiochromic film dosimetry.

Dosepy is intended to be an accessible tool for medical physicist in radiation oncology to perform patient-specific dose verification based on film measurements.

The software works with scanned films (in TIFF format) and a DICOM file (typically exported from a treatment planning system, TPS).
A 2D gamma analysis can be performed in order to evaluate the similarity between the measured (scanned film) and the planed (DICOM file) dose distributions.

## Installation

Dosepy is distributed as a Python library under the [Python Package Index](https://pypi.org/) (PyPI).
Open a console (or 'terminal', or 'command prompt') and use the pip command:

```bash
pip install Dosepy
```

See the Python for Beginners [getting started tutorial](https://opentechschool.github.io/python-beginners/en/getting_started.html#what-is-python-exactly) for an introduction to using your operating system’s console and interacting with Python.

## Features

## Film dosimetry

Dosepy has a graphical user interface (GUI) to perform film dosimetry. Once a TIFF file is loaded, scanned films are automatically detected. Multiple scans of the same film can be loaded and averaged automatically for noise reduction.

## Gamma index

 Dose distributions comparison can be performed using the 2-dimensional gamma index test according to Low's definition [Daniel_Low_gamma_1998](https://doi.org/10.1118/1.598248), as well as some AAPM TG-218 [Miften_TG218_2018](https://doi.org/10.1002/mp.12810) recommendations:

* The acceptance criteria for dose difference can be selected in absolute mode (in Gy) or relative mode (in %).
  * In relative mode, the percentage could be interpreted with respect to the maximum dose (global normalization), or with respect to the local dose (local normalization); according to user selection.
* Dose threshold can be adjusted by the user.
* The reference distribution can be selected by the user.
* It is possible to define a search radius as an optimization process for calculation.
* By default, percentile 99 from dose distribution is used as maximum dose. This is used to avoid the possible inclusion of artifacts or user markers.
* Interpolation is not yet supported.

## Used technologies

* [Matplotlib](https://matplotlib.org/) for data visualization.
* [Numpy](https://numpy.org/) for data array manipulation.
* [PySide6](https://doc.qt.io/qtforpython-6/) for graphical user interface (GUI).
* [Pydicom](https://pydicom.github.io/) to read files in DICOM format.
* [Imageio](imageio) to read files in TIFF format.
* [Scikit-image](https://scikit-image.org/) and [scipy](https://scipy.org/) for image processing.

## Warning!
To use a software as a [medical device](https://www.imdrf.org/documents/software-medical-device-samd-key-definitions), it is required to demonstrate its safety and efficacy through a [risk categorization structure](https://www.imdrf.org/documents/software-medical-device-possible-framework-risk-categorization-and-corresponding-considerations), a [quality management system](https://www.imdrf.org/documents/software-medical-device-samd-application-quality-management-system) and a [clinical evaluation](https://www.imdrf.org/documents/software-medical-device-samd-clinical-evaluation); as described in the International Forum of Medical Device Regulators working group guidelines (IMDRF).

Dosepy is currently **under development** to meet quality standards. To achieve this in Mexico the regulatory mechanism is through NOM-241-SSA1-2021, in addition to the IMDRF guidelines.

## Contributing

Dosepy uses GitHub as a plataform to store and develop the software.
* To report software bugs create a issue [here](https://github.com/LuisOlivaresJ/Dosepy/issues)
* To commit changes, create an issue, [fork](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) the respository, make your changes and make a new pull request.

## Discussion
Have questions? Ask them on the Dosepy [discussion forum](https://groups.google.com/g/dosepy).