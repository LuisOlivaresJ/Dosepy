# Welcome to Dosepy

![Portada_Dosepy](../assets/portada_DOSEPY.png)

**Dosepy** is a python library for 2D/1D gamma analysis and film dosimetry used in radiotherapy.

```{warning}
For the use of a software as a [medical device](https://www.imdrf.org/documents/software-medical-device-samd-key-definitions), it is required to demonstrate its safety and efficacy through a [risk categorization structure](https://www.imdrf.org/documents/software-medical-device-possible-framework-risk-categorization-and-corresponding-considerations), a [quality management system](https://www.imdrf.org/documents/software-medical-device-samd-application-quality-management-system) and a [clinical evaluation](https://www.imdrf.org/documents/software-medical-device-samd-clinical-evaluation); as described in the International Forum of Medical Device Regulators working group guidelines (IMDRF).

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
