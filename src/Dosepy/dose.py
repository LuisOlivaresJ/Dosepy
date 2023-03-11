# -*- coding: utf-8 -*-
"""
Última modificación: 15 Julio 2021

@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico en Radioterapia, La Paz, Baja California Sur, México.
"""

import numpy as np
import pydicom
import matplotlib.pyplot as plt

class Dose:
    ''' Clase para la representación de una distribución de dosis absorbida.

    Regresa un objeto Dose que contiene la distribución de dosis y la resolución espacial.

    Parámetros
    -----------
    array : numpy.ndarray
        Arreglo o matriz de datos. Cada valor numérico representa la dosis absorbida en un punto en el espacio.

    resolution : float
        Resolución espacial dada como la distancia física (en milímetros) entre dos puntos consecutivos.

    '''

    def __init__(self, array, resolution):
        ''' Inicialización del objeto Dose. Como primer argumento, array, se espera un objeto ndarray 2D (arreglo matricial de datos).
            Como segundo argumento, resolution, se espera la distancia física (en mm) entre dos puntos consecutivos.'''

        self.array = array                  #   Valores numéricos de la distribución de dosis
        self.columns = array.shape[1]       #   Número de columnas en la matriz.
        self.rows = array.shape[0]          #   Número de filas en la matriz.
        self.resolution = resolution;       #   Resolución espacial dada como la distancia entre dos puntos consecutivos [mm].

    def gamma2D(self, D_reference, dose_t=3, dist_t=3, dose_tresh=10, dose_t_Gy=False, local_norm=False, mask_radius=10, max_as_percentile = True):
        ''' Cálculo del índice gamma contra una distribución de referencia.
            Se obtiene una matriz que representa los índices gamma en cada posición de la distribución de dosis,
            así como el índice de aprobación definido como el porcentaje de valores gamma que son menor o igual a 1.
            Se asume el registro de las distribuciones de dosis, es decir, que la coordenada espacial de un punto en la distribución de
            referencia es igual a la coordenada del mismo punto en la distribución a evaluar.

        Parámetros
        ----------
        D_reference : Objeto Dose
            Distribución de dosis de referencia contra la cual se realizará la comparación.
            El número de filas y columnas debe de ser igual a la distribución a evaluar (self.array).
            Lo anterior implica que las dimesiones espaciales de las distribuciones deben de ser iguales.

        dose_t : float, default = 3
            Tolerancia para la diferencia en dosis.
            Este valor puede interpretarse de 3 formas diferentes según los parámetros dose_t_Gy,
            local_norm y max_as_percentil, los cuales se describen más adelante.

        dist_t : float, default = 3
            Tolerancia para la distancia, en milímetros (criterio DTA [1]).

        dose_tresh : float, default = 10
            Umbral de dosis, en porcentaje (0 a 100) con respecto a la dosis máxima de la 
            distribución de referencia (o al percentil 99 si max_as_percentile = TRUE). 
            Todo punto en la distribución de dosis con un valor menor al umbral
            de dosis, es excluido del análisis.
            
        dose_t_Gy : bool, default: False
            Si el argumento es True, entonces "dose_t" (la dosis de tolerancia) se interpreta como un valor fijo y absoluto en Gray [Gy].
            Si el argumento es False (default), "dose_t" se interpreta como un porcentaje.

        local_norm : bool, default: False
            Si el argumento es True (normalización local), el porcentaje de dosis de tolerancia "dose_t" se interpreta con respecto a la dosis local
            en cada punto de la distribución de referencia.
            Si el argumento es False (normalización global), el porcentaje de dosis de tolerancia "dose_t" se interpreta con respecto al
            máximo de la distribución a evaluar (o al percentil 99.1 si max_as_percentile = TRUE).
            Notas:
                1.- Los argumentos dose_t_Gy y local_norm NO deben ser seleccionados como True de forma simultánea.
                2.- Si se desea utilizar directamente el máximo de la distirbución, utilizar el parámetro max_as_percentile = False (ver mas adelante)

        mask_radius : float, default: 10
            Distancia física en milímetros que se utiliza para acotar el cálculo con posiciones que estén dentro de una vecindad dada por mask_radius.

            Para lo anterior, se genera un área de busqueda cuadrada o "máscara" aldrededor de cada punto o posición en la distribución de referencia.
            El uso de esta máscara permite reducir el tiempo de cálculo debido al siguiente proceso:
                Por cada punto en la distribución de referencia, el cálculo de la función Gamma se realiza solamente
                con aquellos puntos o posiciones de la distribución a evaluar que se encuentren a una distancia relativa
                menor o igual a mask_radius, es decir, con los puntos que están dentro de la vecindad dada por mask_radius.
                La longitud de uno de los lados de la máscara cuadrada es de 2*mask_radius + 1.
            Por otro lado, si se prefiere comparar con todos los puntos de la distribución a evaluar, es suficiente con ingresar
            una distancia mayor a las dimensiones de la distribución de dosis (por ejemplo mask_radius = 1000).

        max_as_percentile : bool, default: True
            -> Si el argumento es True, se utiliza el percentil 99 como una aproximación del valor máximo de la
               distribución de dosis. Lo anterior permite excluir artefactos o errores en posiciones puntuales
               (de utilidad por ejemplo cuando se utiliza película radiocrómica o etiquetas puntuales en la distribución).
            -> Si el argumento es False, se utiliza directamente el valor máximo de la distribución a evaluar.

        Retorno
        ----------
        ndarray :
            Array, o matriz bidimensional con la distribución de índices gamma.

        float :
            Índice de aprobación. Se calcula como el porcentaje de valores gamma <= 1, sin incluir las posiciones
            en donde la dosis es menor al umbral de dosis.

        Consideraciones
        ----------
             Es posible utilizar el percentil 99 de la distribución de dosis como una aproximación del valor máximo.
             Esto permite evitar la posible inclusión de artefactos o errores en posiciones puntuales de la distribución
             (de utilidad por ejemplo cuando se utiliza película radiocrómica o etiquetas puntuales en la distribución).

             Se asume que ambas distribuciones a evaluar representan exactamente las mismas dimensiones físicas, y las posiciones
             espaciales para cada punto conciden entre ellas, es decir, las imagenes de cada distribución están registradas.

             No se realiza interpolación entre puntos.

        Referencias
        ------------
            Para mayor información sobre los mecanismos de operación, efectividad y exactitud de la herramienta gamma consultar:

                [1] M. Miften, A. Olch, et. al. "Tolerance Limits and Methodologies for IMRT Measurement-Based
                    Verification QA: Recommendations of AAPM Task Group No. 218" Medical Physics, vol. 45, nº 4, pp. e53-e83, 2018.
                [2] D. Low, W. Harms, S. Mutic y J. Purdy, «A technique for the quantitative evaluation of dose distributions,»
                    Medical Physics, vol. 25, nº 5, pp. 656-661, 1998.
                [3] L. A. Olivares-Jimenez, "Distribución de dosis en radioterapia de intensidad modulada usando películas de tinte
                    radiocrómico : irradiación de cerebro completo con protección a hipocampo y columna con protección a médula"
                    (Tesis de Maestría) Posgrado en Ciencias Físicas, IF-UNAM, México, 2019

        Example. Archivo en formato CSV (comma separated values)
        ---------

        import Dosepy.dose as dp

        #   Cargamos los archivos "D_TPS.csv" y "D_FILM.csv"
        #   (Los archivos de ejemplo .csv se encuentran dentro del paquete Dosepy, en la carpeta src/Dosepy/data)
        >>> D_eval = dp.from_csv("D_TPS.csv", 1)
        >>> D_ref = dp.from_csv("D_FILM.csv", 1)

        #   Llamamos al método gamma2D, con criterio 3 %, 2 mm.
        >>> g, pass_rate = D_eval.gamma2D(D_ref, 3, 2)

        #   Imprimimos el resultado
        >>> print(f'El índice de aprobación es: {pass_rate:.1f} %')
        >>> plt.imshow(g, vmax = 1.4)
        >>> plt.show()

        El índice de aprobación es: 98.9 %

        '''

        #%%

        #   Verificar la ocurrencia de excepciones
        if D_reference.array.shape != self.array.shape:
            raise Exception("No es posible el cálculo con matrices de diferente tamaño.")

        if local_norm and dose_t_Gy:
            raise Exception("No es posible la selección simultánea de dose_t_Gy y local_norm.")

        if D_reference.resolution != self.resolution:
            raise Exception("No es posible el cálculo con resoluciones diferentes para cada distribución.")

        #%%

        D_ref = D_reference.array
        D_eval = self.array

        if max_as_percentile:
            maximum_dose = np.percentile(D_eval, 99)
        else:
            maximum_dose = np.amax(D_eval)
        print(f'Dosis máxima: {maximum_dose:.1f}')
        #  Umbral de dosis
        Dose_threshold = (dose_tresh/100)*maximum_dose
        print(f'Umbral de dosis: {Dose_threshold:.1f}')

        #   Dosis de tolerancia absoluta o relativa
        if dose_t_Gy:
            pass
        elif local_norm:
            pass
        else:
            dose_t = (dose_t/100) * maximum_dose

        #   Número de pixeles que se usarán para definir una vecindad sobre la que se calculará el índice gamma.
        neighborhood = round(mask_radius*1./self.resolution)

        #   Matriz que guardará el resultado del índice gamma.
        gamma = np.zeros( (self.rows, self.columns) )




        #%%
        for i in np.arange( D_ref.shape[0] ):
            #   Código que permite incluir puntos cerca de la frontera de la distribución de dosis
            mi = -(neighborhood - max(0, neighborhood - i))
            mf = neighborhood - max(0, neighborhood - (D_eval.shape[0] - (i+1))) + 1

            for j in np.arange( D_ref.shape[1] ):
                ni = -(neighborhood - max(0, neighborhood - j))
                nf = neighborhood - max(0, neighborhood - (D_eval.shape[1] - (j+1))) + 1

                #   Para almacenar temporalmente los valores de la función Gamma por cada punto en la distribución de referencia
                Gamma = []

                for m in np.arange(mi , mf):
                    for n in np.arange(ni, nf):

                        # Distancia entre dos posiciones (en milímetros), por fila
                        dm = m*self.resolution
                        # Distancia entre dos posiciones (en milímetros), por columna
                        dn = n*self.resolution
                        # Distancia total entre dos puntos
                        distance = np.sqrt(dm**2 + dn**2)

                        # Diferencia en dosis
                        dose_dif = D_eval[i + m, j + n] - D_ref[i,j]


                        if local_norm:
                            # La dosis de tolerancia se actualiza al porcentaje con respecto al valor
                            # de dosis local en la distribución de referencia.
                            dose_t_local = dose_t * D_ref[i,j] / 100

                            Gamma.append(
                                np.sqrt(
                                    (distance**2) / (dist_t**2)
                                    + (dose_dif**2) / (dose_t_local**2))
                                        )

                        else :
                            Gamma.append(
                                np.sqrt(
                                    (distance**2) / (dist_t**2)
                                    + (dose_dif**2) / (dose_t**2))
                                        )

                gamma[i,j] = min(Gamma)

                # Para la posición en cuestión, si la dosis es menor al umbral de dosis,
                # entonces dicho punto no se toma en cuenta en el porcentaje de aprobación.
                if D_eval[i,j] < Dose_threshold:
                    gamma[i,j] = np.nan

        # Arroja las coordenadas en donde los valores gamma son menor o igual a 1
        less_than_1_coordinate = np.where(gamma <= 1)
        # Cuenta el número de coordenadas en donde se cumple que gamma <= 1
        less_than_1 = np.shape(less_than_1_coordinate)[1]
        # Número de valores gamma diferentes de np.nan
        total_points = np.shape(gamma)[0]*np.shape(gamma)[1] - np.shape(np.where(np.isnan(gamma)))[1]

        #   Índice de aprobación
        gamma_percent = float(less_than_1)/total_points*100
        return gamma, gamma_percent

def from_dicom(file_name):
    """
    Importación de un archivo de dosis en formato DICOM

    Parámetros
    -----------
    file_name : str
        Nombre del archivo en formato string

    Return
    --------
    Dosepy.dose.Dose
        Objeto Dose del paquete Dosepy que representa a la distribución de dosis

    Consideraciones
    ----------------
        La distribución de dosis en el archivo DICOM debe contener solo dos dimensiones.
        La resolución espacial debe de ser igual en ambas dimensiones.
        No se hace uso de las coordenadas dadas en el archivo DICOM. Ver segunda consideración en la nota del método gamma2D de la clase Dose.

    """
    DS = pydicom.dcmread(file_name)
    array = DS.pixel_array
    #image_orientation = DS.ImageOrientationPatient
    if array.ndim != 2:
        raise Exception("La distribución de dosis debe de ser en dos dimensiones")
    dgs = DS.DoseGridScaling
    D_array = array * dgs
    resolution = DS.PixelSpacing
    if resolution[0] != resolution[1]:
        raise Exception("La resolución espacial debe ser igual en ambas dimensiones.")

    Dose_DICOM = Dose(D_array, resolution[0])
    return Dose_DICOM


def from_csv(file_name, PixelSpacing):
    """
    Importación de un archivo de dosis en formato CSV (Comma separated values).
    Dentro del archivo .csv, utilizar el caracter # al inicio de una fila para
    que sea descartada (inicio de un comentario).

    Parámetros
    -----------
    file_name : str
        Nombre del archivo en formato string

    PixelSpacing : float
        Distancia en milímetros entre dos puntos consecutivos.

    Return
    --------
    Dosepy.dose.Dose
        Objeto Dose del paquete Dosepy que representa a la distribución de dosis.

    """

    array = np.genfromtxt(file_name, delimiter = ",", comments = "#")
    Dose_csv = Dose(array, PixelSpacing)
    return Dose_csv


#%%
def main():

    "Bloque de código para realizar pruebas"

    D_eval = from_csv("D_TPS.csv", 1)
    D_ref = from_csv("D_FILM.csv", 1)

    g, pass_percent = D_eval.gamma2D(D_ref, dose_t=3, dist_t=2)

    print(f'El índice de aprobación es: {pass_percent:.1f} %')
    plt.imshow(g, vmax = 1.4)
    plt.show()

#%%

if __name__ == "__main__":
    main()
