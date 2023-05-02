# Welcome to Dosepy

![Portada_Dosepy](../assets/portada_DOSEPY.png)

Dosepy is a python library for easy 2D/1D gamma analysis and film dosimetry used in radiotherapy. 

```{note}
Dose distribution can be in DICOM (.dmc) or CVS format. For digital film a TIFF format is required.
```

```{caution}
In order to generate easy-to-use software for users who use radiochromic film, the dose distributions must meet the following characteristics:
* The dose distributions to be compared must have the same physical dimensions and spatial resolution (same number of rows and columns). **TODO** Agregar referencia a función *reducir*
* The distributions must be registered, that is, the coordinate of a point in the reference distribution must be equal to the coordinate of the same point in the distribution to be evaluated.
* Gray (Gy) and millimeters (mm) are the units used for absorbed dose and physical distance, respectively.
```

## Dose comparison
### Gamma index

![Imagen_gamma](../assets/Image_gamma.png)

 Dose distributions comparison can be performed using the 2-dimensional gamma index test according to Low's definition {cite}`Daniel_Low_gamma_1998`, as well as some AAPM TG-218 {cite}`Miften_TG218_2018` recommendations:

* The acceptance criteria for dose difference can be selected in absolute mode (in Gy) or in relative mode (in %).
  * In relative mode, the percentage could be interpreted with respect to the maximum dose (global normalization), or with respect to the local dose (local normalization); according to user selection.
* Dose threshold can be adjusted by the user.
* The reference distribution can be selected by the user.
* It is allowed to define a search radius as an optimization process for the calculation.
* By default, percentile 99 from dose distribution is used as maximum dose. This makes it possible to avoid the possible inclusion of artifacts or user labels in specific positions of the distribution (useful with radiochromic film).
* Interpolation is not yet supported.

### Dose profiles

![Imagen_perfil_1](../assets/Perfiles_1.png)

![Imagen_perfil_2](../assets/Perfiles_2.png)

It is also possible to compare two dose distributions using vertical and horizontal profiles. The position of each profile must be selected with the help of a graphical user interface.

```{warning}
To use a software as a [medical device](https://www.imdrf.org/documents/software-medical-device-samd-key-definitions), it is required to demonstrate its safety and efficacy through a [risk categorization structure](https://www.imdrf.org/documents/software-medical-device-possible-framework-risk-categorization-and-corresponding-considerations), a [quality management system](https://www.imdrf.org/documents/software-medical-device-samd-application-quality-management-system) and a [clinical evaluation](https://www.imdrf.org/documents/software-medical-device-samd-clinical-evaluation); as described in the International Forum of Medical Device Regulators working group guidelines (IMDRF).

Dosepy is currently **under development** to meet quality standards. To achieve this in Mexico the regulatory mechanism is through NOM-241-SSA1-2021, in addition to the IMDRF guidelines.
```

## Installation

To install the software, do:

```{code-block}
---
emphasize-lines: 1
---
(.venv) $ pip install Dosepy
```

## Usage

The easiest way to use Dosepy is through a graphical user interface (GUI). Open a python interpreter and import Dosepy.GUI as follows:

```python
import Dosepy.GUI
```

Dosepy comes pre-loaded with two examples of dose distributions, with the aim that the user can interact with the available tools.

You can use the {py:class}`Dosepy.dose.Dose` like this:

## API

```{eval-rst}
.. autoclass:: Dosepy.dose.Dose
```


```{tableofcontents}
```
