# *Dosepy*

*Dosepy* es un paquete escrito en python para la comparación de distribuciones de dosis usadas en radioterapia.<br/>

## Gamma index
La comparación se realiza mediante el índice gamma 2-dimensional siguiendo las recomendaciónes del [TG-218]( https://doi.org/10.1002/mp.12810) de la AAPM:

* El criterio de aceptación para la diferencia en dosis puede ser seleccionado en modo absoluto (en Gy) o relativo.
  * En modo relativo, el porcentaje se interpreta con respecto al máximo de la distribución de dosis (global normalization), o con respecto a la dosis local (local normalization), según la selección del usuario.
* El umbral de dosis es ajustable.
* La ditribución de referencia puede ser seleccionada por el usuario.
* Se permite definir un radio de búsqueda como proceso de optimización para el cálculo.

*Consideraciones*

* Durante el cálculo gamma, se asume que ambas distribuciones a evaluar tienen exactamente las mismas dimensiones físicas, y las posiciones espaciales para cada punto conciden entre ellas.

* No se realiza interpolación entre puntos.

* Es posible utilizar el percentil 99.1 de la distribución de dosis como una aproximación del valor máximo. Esto permite evitar la posible inclusión de artefactos o errores en posiciones puntuales de la distribución (de utilidad por ejemplo cuando se utiliza película radiocrómica).


## Installation

Using [pip](https://pip.pypa.io/en/stable/):
```
pip install Dosepy
```

## Support

If you are having issues, please let us know.<br/>
We have a mailing list located at: alfonso.cucei.udg@gmail.com

## Getting started, example 1

En *Dosepy* una distribución de dosis es representada como un objeto de la clase **Dose** del paquete Dosepy. Para crear el objeto son necesarios dos argumentos: las dosis de la distribución en formato [ndarray](https://numpy.org/doc/stable/reference/index.html#module-numpy) y la resolución espacial en puntos por milímetro.

```
>>> import numpy as np
>>> import Dosepy.dose as dp

>>> a = np.zeros((10,10)) + 100
>>> b = np.zeros((10,10)) + 96  

>>> dose_reference = dp.Dose(a, 1)
>>> dose_evaluation = dp.Dose(b, 1)
```

La comparación entre dos distribuciones se realiza mediante el método *gamma2D*. Como argumentos se requiere:
la distribución de referencia, la diferencia en dosis de tolerancia y la distancia de tolerancia o criterio DTA en mm.

```
#   Llamamos al método gamma2D, con criterio 3 %, 1 mm.
>>> g, g_percent = dose_evaluation.gamma2D(dose_reference, 3, 1)
>>> print(g_percent)
0.0
```

## Data in CSV format, example 2

Es posible cargar archivos de datos en fromato CSV (comma separate values) mediante la función *from_csv* del paquete Dosepy.
Para descartar filas dentro del archivo, utilizar el caracter # al inicio de cada fila (inicio de un comentario).
```
import Dosepy.dose as dp

#   Cargamos los archivos "D_TPS.csv" y "D_FILM.csv", ambos con 1 milímetro de espacio entre un pixel y otro.
#   (Los archivos de ejemplo .csv se encuentran dentro del paquete Dosepy, en la carpeta src/data)
>>> D_eval = dp.from_csv("D_TPS.csv", PixelSpacing = 1)
>>> D_ref = dp.from_csv("D_FILM.csv", PixelSpacing = 1)

#   Llamamos al método gamma2D, con criterio 3 %, 2 mm.
>>> g, pass_rate = D_eval.gamma2D(D_ref, 3, 2)

#   Imprimimos el resultado
>>> print(f'El índice de aprobación es: {pass_rate:.1f} %')
>>> plt.imshow(g, vmax = 1.4)
>>> plt.show()

El índice de aprobación es: 98.9 %

```
## Data in DICOM format, example 3

Importación de un archivo de dosis en formato DICOM

*Consideraciones*

* La distribución de dosis en el archivo DICOM debe contener solo dos dimensiones.
* El espacio entre píxeles debe de ser igual en ambas dimensiones.
* No se hace uso de las coordenadas dadas en el archivo DICOM. Ver primera consideración en el apartado Gamma index.

```
import Dosepy.dose as dp

#   Cargamos los archivos "RD_file.dcm" y "D_FILM_2mm.csv", ambos con 2 milímetro de espacio entre un pixel y otro.
>>> D_eval = dp.from_dicom("RD_file.dcm")
>>> D_ref = dp.from_csv("D_FILM_2mm.csv", PixelSpacing = 2)

#   Llamamos al método gamma2D, con criterio 2 %, 3 mm.
>>> g, pass_rate = D_eval.gamma2D(D_ref, 2, 3)

#   Imprimimos el resultado
>>> print(pass_rate)

```

# Documentation
```
Dosepy.dose.Dose(data, resolution)
  Clase para la representación de una distribución de dosis absorbida.
  Regresa un objeto Dose que contiene la distribución de dosis y la resolución espacial.

Parameters:
           data : numpy.ndarray
                Arreglo o matriz de datos que representa una distribución de dosis.

           resolution : float
                Resolución espacial en puntos por milímetro.

Dose methods

Dose.gamma2D(
  D_reference,
  dose_t = 3,
  dist_t = 3,
  dose_tresh = 10,
  dose_t_Gy = False,
  local_norm = False,
  mask_radius = 5,
  max_as_percentile = True
  )

Cálculo del índice gamma contra una distribución de referencia.
Se obtiene una matriz que representa los índices gamma en cada posición de la distribución de dosis, así como el índice de aprobación
definido como el porcentaje de valores gamma que son menor o igual a 1.

Parameters:
            D_reference : Objeto Dose
                Distribución de dosis de referencia contra la cual se realizará la comparación.
                El número de filas y columnas debe de ser igual a la distribución a evaluar.
                Lo anterior implica que las dimesiones espaciales de las distribuciones deben de ser iguales.

            dose_t : float, default = 3
                Tolerancia para la diferencia en dosis.
                Este valor puede interpretarse de 3 formas diferentes, según los parámetros dose_t_Gy y
                local_norm que se describen más adelante.

            dist_t : float, default = 3
                Tolerancia para la distancia, en milímetros (criterio DTA).

            dose_tresh : float, default = 10
                Umbral de dosis, en porcentaje (0 a 100). Todo punto en la distribución de dosis con un valor menor al umbral
                de dosis, es excluido del análisis.
                Por default, el porcentaje se interpreta con respecto al percentil 99.1 (aproximadamente el máximo)
                de la distribución a evaluar. Si el porcentaje se requiere con respecto al máximo, modificar
                el parámetro max_as_percentile = False (ver más adelante).

            dose_t_Gy : bool, default: False
                Si el argumento es True, entonces "dose_t" (la dosis de tolerancia) se interpreta como un valor fijo y absoluto en Gray [Gy].
                Si el argumento es False (default), "dose_t" se interpreta como un porcentaje.

            local_norm : bool, default: False
                Si el argumento es True (local normalization), el porcentaje de dosis de tolerancia "dose_t" se interpreta con respecto a la dosis local.
                Si el argumento es False (global normalization), el porcentaje de dosis de tolerancia "dose_t" se interpreta con respecto al
                máximo de la distribución a evaluar.
                Nota: Los argumentos dose_t_Gy y local_norm no deben ser seleccionados como True de forma simultánea.

            mask_radius : float, default: 5
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
                -> Si el argumento es True, se utiliza el percentil 99.1 como una aproximación del valor máximo de la
                   distribución de dosis. Lo anterior permite excluir artefactos o errores en posiciones puntuales
                   (de utilidad por ejemplo cuando se utiliza película radiocrómica).
                -> Si el argumento es False, se utiliza directamente el valor máximo de la distribución.

Retorno:

          ndarray :
                Array, o matriz bidimensional con la distribución de índices gamma.

          float :
                Índice de aprobación. Se calcula como el porcentaje de valores gamma <= 1, sin incluir las posiciones en donde la
                dosis es menor al umbral de dosis.



```

Functions
```

from_csv(file_name, PixelSpacing)

    Importación de un archivo de dosis en formato CSV (Comma separated values).
    Dentro del archivo .csv, utilizar el caracter # al inicio de una fila para
    que sea descartada (inicio de un comentario).

    Parameters
    -----------
    file_name : str
        Nombre del archivo en formato string

    PixelSpacing : float
        Distancia entre dos píxeles, en mm

    Return
    --------
    Dosepy.dose.Dose
        Objeto Dose del paquete Dosepy que representa a la distribución de dosis.



from_dicom(file_name)

    Importación de un archivo de dosis en formato DICOM

    Parameters
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

```
