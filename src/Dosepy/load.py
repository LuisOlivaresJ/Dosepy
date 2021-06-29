"""
Última modificación: 29 Junio 2021

@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico en Radioterapia, La Paz, Baja California Sur, México.
"""
#   Consideraciones
#   La distribución de dosis en el archivo DICOM debe contener solo dos dimensiones.
#   La resolución espacial debe de ser igual en ambas dimensiones.
#   No se utiliza la coordenada espacial del primer pixel, según se puede obtener desde el archivo DICOM.

import pydicom
from dose import Dose

def from_dicom(file_name):
    DS = pydicom.dcmread(file_name)
    array = DS.pixel_array
    image_orientation = DS.ImageOrientationPatient
    if array.ndim != 2:
        raise Exception("La distribución de dosis debe ser en dos dimensiones")
    dgs = DS.DoseGridScaling
    D_array = array * dgs
    resolution = DS.PixelSpacing
    #   Verificar la ocurrencia de excepciones
    if resolution[0] != resolution[1]:
        raise Exception("La resolución espacial debe ser igual en ambas dimensiones.")

    Dose_DICOM = Dose(D_array, 1 / float(resolution[0]))
    return Dose_DICOM


def from_csv(file_name):
    pass
