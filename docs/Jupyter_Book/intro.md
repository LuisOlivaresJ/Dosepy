# Welcome to Dosepy

![Portada_Dosepy](../assets/Calibration_tab.png)

Dosepy allows to easily perform film dosimetry and 2D gamma analysis.

The software uses tif images for film dosimetry. A DICOM file exported from a treatment planning system can be loaded to perform gamma index comparison.

## Film dosimetry

Dosepy has a graphical user interface to perform film dosimetry. Once a tif file is loaded, scanned films are automatically detected. Multiple scans of the same film can be loaded and averaged automatically for noise reduction.

## Gamma index

For gamma index analysis, you need to write some Python code. There are some examples in the [gamma section](gamma.ipynb).

 Dose distributions comparison can be performed using the 2-dimensional gamma index test according to Low's definition {cite}`Daniel_Low_gamma_1998`, as well as some AAPM TG-218 {cite}`Miften_TG218_2018` recommendations:

* The acceptance criteria for dose difference can be selected in absolute mode (in Gy) or relative mode (in %).
  * In relative mode, the percentage could be interpreted with respect to the maximum dose (global normalization), or with respect to the local dose (local normalization); according to user selection.
* Dose threshold can be adjusted by the user.
* The reference distribution can be selected by the user.
* It is possible to define a search radius as an optimization process for calculation.
* By default, percentile 99 from dose distribution is used as maximum dose. This is used to avoid the possible inclusion of artifacts or user markers.
* Interpolation is not supported yet.

```{warning}
To use a software as a [medical device](https://www.imdrf.org/documents/software-medical-device-samd-key-definitions), it is required to demonstrate its safety and efficacy through a [risk categorization structure](https://www.imdrf.org/documents/software-medical-device-possible-framework-risk-categorization-and-corresponding-considerations), a [quality management system](https://www.imdrf.org/documents/software-medical-device-samd-application-quality-management-system) and a [clinical evaluation](https://www.imdrf.org/documents/software-medical-device-samd-clinical-evaluation); as described in the International Forum of Medical Device Regulators working group guidelines (IMDRF).

Dosepy is currently **under development** to meet quality standards. To achieve this in Mexico the regulatory mechanism is through NOM-241-SSA1-2021, in addition to the IMDRF guidelines.
```

## Algorithm validation


**Film dosimetry**

Using EBT 3 radiochromic film, total dispersion factors (also known as Output factors) were measured for a 6 FFF beam from a Clinac-iX linear accelerator. Following the IAEA-AAPM TRS 483 code of practice, the results were compared with measurements from two ionization chambers. [The results](https://smf.mx/programas/congreso-nacional-de-fisica/memorias-cnf/) were presented at the LXIII National Physics Congress (2020).
![Image_factores_campo](../assets/Factores_de_campo_6FFF.png)


**Gamma index**

[Abstract](https://github.com/LuisOlivaresJ/Dosepy/blob/2bf579e6c33c347ef8f0cdd6f4ee7534798f0d13/docs/assets/validation.pdf)<br/>
Validation for the gamma index algorithm was carried out by comparing Dosepy results against DoseLab 4.11 and VeriSoft 7.1.0.199. The work was presented at the 7th Congress of the Mexican Federation of Medical Physics Organizations in 2021. [(Video)](https://youtu.be/HM4qkYGzNFc).

![valid_gamma](../assets/valid_gamma_1.png)

## Discussion
Have questions? Ask them on the Dosepy [discussion forum](https://groups.google.com/g/dosepy).

```{tableofcontents}
```
