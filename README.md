# Welcome to Dosepy

[Documentation](https://dosepy.readthedocs.io/en/latest/intro.html)

Dosepy allows to easily perform film dosimetry and 2D gamma analysis.

The software uses tif images for film dosimetry. A DICOM file exported from a treatment planning system can be used to perform gamma index comparison.

## Installation

Install via pip:

```bash
pip install Dosepy
```

## Film dosimetry

Dosepy has a graphical user interface to perform film dosimetry. Once a tif file is loaded, scanned films are automatically detected. Multiple scans of the same film can be loaded and averaged automatically for noise reduction.

## Gamma index

 Dose distributions comparison can be performed using the 2-dimensional gamma index test according to Low's definition [Daniel_Low_gamma_1998](https://doi.org/10.1118/1.598248), as well as some AAPM TG-218 [Miften_TG218_2018](https://doi.org/10.1002/mp.12810) recommendations:

* The acceptance criteria for dose difference can be selected in absolute mode (in Gy) or relative mode (in %).
  * In relative mode, the percentage could be interpreted with respect to the maximum dose (global normalization), or with respect to the local dose (local normalization); according to user selection.
* Dose threshold can be adjusted by the user.
* The reference distribution can be selected by the user.
* It is possible to define a search radius as an optimization process for calculation.
* By default, percentile 99 from dose distribution is used as maximum dose. This is used to avoid the possible inclusion of artifacts or user markers.
* Interpolation is not yet supported.

```{warning}
To use a software as a [medical device](https://www.imdrf.org/documents/software-medical-device-samd-key-definitions), it is required to demonstrate its safety and efficacy through a [risk categorization structure](https://www.imdrf.org/documents/software-medical-device-possible-framework-risk-categorization-and-corresponding-considerations), a [quality management system](https://www.imdrf.org/documents/software-medical-device-samd-application-quality-management-system) and a [clinical evaluation](https://www.imdrf.org/documents/software-medical-device-samd-clinical-evaluation); as described in the International Forum of Medical Device Regulators working group guidelines (IMDRF).

Dosepy is currently **under development** to meet quality standards. To achieve this in Mexico the regulatory mechanism is through NOM-241-SSA1-2021, in addition to the IMDRF guidelines.
```

## Discussion
Have questions? Ask them on the Dosepy [discussion forum](https://groups.google.com/g/dosepy).