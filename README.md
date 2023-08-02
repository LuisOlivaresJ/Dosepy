# *Dosepy*

[ReadTheDocs Documentation](https://dosepy.readthedocs.io/en/latest/intro.html)

*Dosepy* es un paquete de código escrito en Python para la comparación mediante índice gamma de dos distribuciones de dosis, 2-dimensional. Adicionalmente, se cuenta con una herramienta para realizar dosimetría con película radiocrómica.<br/>

El formato de los archivos que contengan la distribución de dosis puede ser DICOM (.dmc) o CVS. Para la película se requiere un formato TIFF.<br/>

> **Condiciones de uso.** Toda persona tiene acceso a la lectura y uso del código con fines académicos o de enseñanza. Sin embargo, para el uso clínico del programa se requiere contar con una licencia (disponible próximamente), conocida como “Acuerdo de licencia de usuario final” (EULA, por sus siglas en inglés), así como contratos que garanticen el cumplimiento de la legislación de cada país.<br/>  

Para mayor información contactar al correo electrónico dosepy@gmail.com. 

Derechos Reservados (c) Luis Alfonso Olivares Jimenez 2021

## Documentación

Para la instalación, correr ejemplos de prueba y aprender sobre el programa Dosepy, [visita la Documentación](https://dosepy.readthedocs.io/en/latest/intro.html) en Read The Docs.

## Métodos de comparación

### Comparación por índice gamma
La comparación de dos distribuciones puede realizarse mediante la prueba del índice gamma 2-dimensional de acuerdo a la definición dada por [Low D. A.](https://doi.org/10.1118/1.598248) así como algunas recomendaciones del [TG-218]( https://doi.org/10.1002/mp.12810) de la AAPM:

* El criterio de aceptación para la diferencia en dosis puede ser seleccionado en modo absoluto (en Gy) o en modo relativo (en %).
  * En modo relativo, el porcentaje puede interpretarse con respecto al máximo de la distribución de dosis a evaluar (normalización global), o con respecto a la dosis local en la distribución de referencia (normalización local); según la selección del usuario.
* El umbral de dosis puede ser ajustado por el usuario.
* La distribución de referencia puede ser seleccionada por el usuario.
* Se permite definir un radio de búsqueda como proceso de optimización para el cálculo.
* Es posible utilizar el percentil 99.1 de la distribución de dosis como una aproximación del valor máximo. Esto permite evitar la posible inclusión de artefactos o errores en posiciones puntuales de la distribución (de utilidad por ejemplo cuando se utiliza película radiocrómica).
* No se realiza interpolación entre puntos.

**Proceso de una primera validación del algoritmo gamma**<br/>

[Resumen](https://github.com/LuisOlivaresJ/Dosepy/blob/2bf579e6c33c347ef8f0cdd6f4ee7534798f0d13/docs/assets/validation.pdf)<br/>
La validación del algoritmo para la prueba del índice gamma se realizó mediante la comparación de resultados contra los softwares DoseLab 4.11 y VeriSoft 7.1.0.199. Dicho trabajo se presentó en el 7mo Congreso de la Federación Mexicana de Organizaiones de Física Médica en el año 2021 [(Video)](https://youtu.be/HM4qkYGzNFc).

**¡Consideraciones!**

* Ambas distribuciones deben tener las mismas dimensiones físicas y resolución espacial (mismo número de filas y columnas).
* Las distribuciones deben de  encontrarse registradas, es decir, la coordenada espacial de un punto en la distribución de referencia debe ser igual a la coordenada del mismo punto en la distribución a evaluar.<br/>

En caso contrario, *Dosepy* dispone de algunas funciones para cumplir con lo anterior.

## Instalación
**En Linux**<br/>
El método más sencillo para instalar Dosepy es escribiendo en una terminal:
```bash
pip install Dosepy
```
**En Windows**<br/>
Previo a la instalación de Dosepy, es necesario contar con un administrador de paquetes. Para quienes no estén familiarizados con los paquetes Python, se recomienda utilizar la plataforma [ANACONDA](https://www.anaconda.com/products/individual).
Una vez que se ha instalado ANACONDA, abrir el inicio de Windows y buscar *Anaconda Prompt*. Dentro de la terminal (ventana con fondo negro), seguir la indicación descrita para Linux (párrafo anterior).

### Versión Beta
Dosepy se encuentra en una versión beta, especificada por el formato 0.X.X. Lo anterior implica que en la práctica, un código que utiliza el paquete Dosepy en una versión, pudiera no ser ejecutado en una versión posterior.  La versión estable será publicada con el formato 1.X.X.<br/>
Para mantener actualizado el paquete Dosepy, utilizar [pip](https://pip.pypa.io/en/stable/):
```bash
pip install --upgrade Dosepy
```

### Ejemplos
 
**Ejemplo con interfaz gráfica**

Para utilizar *Dosepy* con una interfaz gráfica , abrimos una terminal (o Anaconda Prompt en el caso de Windows) y escribimos el comando **python**:

```bash
python
```
Posteriormente, escribimos:

```python
import Dosepy.GUI
```

**Uso de un Notebook**

Para aprender a utilizar todas las herramientas de *Dosepy* se recomienda el uso de un Notebook del entorno [Jupyter](https://jupyter.org/). [*Aquí*](Notebook.md) puedes consultar una guía para ello.

### Importación de archivo en formato csv
La importación de la distribución de referencia puede realizarse sólo si el archivos se encuentra en formato .csv (valores separados por comas). Adicionalmente:
* El archivo deberá contener sólo los valores de dosis.
* Toda información adicional deberá estar precedida con el carácter "#". Ello indicará que todos los caracteres que se encuentren en la misma linea después de "#" debe de ser ignorados por Dosepy.
* La unidad para la dosis deberá ser el Gray (Gy).

### Importación de archivo en formato dcm
La distribución a evaluar puede importarse en un archivo con formato .csv o en formato .dcm (archivo DICOM). Si el formato es DICOM:
* Deberá contener sólo un plano de dosis.
* La resolución espacial debe ser igual en cada dimensión.
* La unidad para la dosis deberá ser el Gray (Gy).

## Ejemplo utilizando una terminal
En *Dosepy*, una distribución de dosis es representada como un objeto de la [clase](https://docs.python.org/es/3/tutorial/classes.html) **Dose** del paquete *Dosepy*. Para crear el objeto son necesarios dos argumentos: las dosis de la distribución en formato [ndarray](https://numpy.org/doc/stable/reference/index.html#module-numpy) y la resolución espacial dada por la distancia (en milímetros) entre dos puntos consecutivos.
Para utilizar *Dosepy*, abrimos una terminal (o Anaconda Prompt en el caso de Windows) y escribimos el comando python:

```bash
python
```

Dentro de Python, escribimos el siguiente código de prueba:

```python
import numpy as np
import Dosepy.dose as dp

a = np.zeros((10,10)) + 96   # Matrices de prueba
b = np.zeros((10,10)) + 100  

D_ref = dp.Dose(a, 1)   # Se crea la distribución de referencia
D_eval = dp.Dose(b, 1)  # Se crea la distribución a evaluar
```

La comparación gamma entre dos distribuciones de dosis se realiza mediante el método *gamma2D*. Como argumentos se requiere:
* La distribución de dosis de referencia
* El porcentaje para la diferencia en dosis de tolerancia
* La distancia de tolerancia o criterio DTA en mm.

```python
#   Llamamos al método gamma2D, con criterio 3 %, 1 mm.
gamma_distribution, pass_rate = D_eval.gamma2D(D_ref, dose_t = 3, dist_t = 1)
print(pass_rate)
```

## Datos en formato CSV, usando un umbral de dosis

Es posible cargar archivos de datos en fromato CSV (comma separate values) mediante la función *from_csv* del paquete Dosepy.
Para descartar filas dentro del archivo, utilizar el caracter # al inicio de cada fila (inicio de un comentario).
```python
import Dosepy.dose as dp
import matplotlib.pyplot as plt

#   Cargamos los archivos "D_TPS.csv" y "D_FILM.csv", ambos con 1.0 milímetro de espacio entre un punto y otro.
#   (Los archivos de ejemplo .csv se encuentran dentro del paquete Dosepy, en la carpeta src/Dosepy/data/)
D_eval = dp.from_csv("D_TPS.csv", PixelSpacing = 1)
D_ref = dp.from_csv("D_FILM.csv", PixelSpacing = 1)

#   Llamamos al método gamma2D, con criterio 3 %, 2 mm, descartando puntos con dosis por debajo del 10 %.
g, pass_rate = D_eval.gamma2D(D_ref, dose_t = 3, dist_t = 2, dose_tresh = 10)

#   Imprimimos el resultado
print(f'El índice de aprobación es: {pass_rate:.1f} %')
plt.imshow(g, vmax = 1.4)
plt.show()

#El índice de aprobación es: 98.9 %

```
## Datos en formato DICOM y modo de dosis absoluto

Importación de un archivo de dosis en formato DICOM

*Consideraciones*

* La distribución de dosis en el archivo DICOM debe contener solo dos dimensiones (2D).
* El espacio entre dos puntos (pixeles) debe de ser igual en ambas dimensiones.
* No se hace uso de las coordenadas dadas en el archivo DICOM. Ver primera consideración en el apartado Gamma index.

```python
import Dosepy.dose as dp

#   Cargamos los archivos "RD_file.dcm" y "D_FILM_2mm.csv", ambos con 2 milímetro de espacio entre un punto y otro.
D_eval = dp.from_dicom("RD_file.dcm")
D_ref = dp.from_csv("D_FILM_2mm.csv", PixelSpacing = 2)

#   Llamamos al método gamma2D, con criterio de 0.5 Gy para la diferencia en dosis y 3 mm para la diferencia en distancia.
g, pass_rate = D_eval.gamma2D(D_ref, 0.5, 3, dose_t_Gy = True)

#   Imprimimos el resultado
print(pass_rate)

```

**Presentación en eventos científicos**

* (2021) 7mo Congreso de la Federación Mexicana de Organizaiones de Física Médica, "Desarrollo y validación de un software de código abierto para la comparación de distribuciones de dosis usadas en radioterapia" [(Video disponible)](https://youtu.be/HM4qkYGzNFc)

# Discussion
Have questions? Ask them on the Dosepy [discussion forum](https://groups.google.com/g/dosepy).

04-2023 Versión 0.4.0<br/>
* On the GUI, dose objects are created after loading. 
* When a csv file is open, a new window shows to ask for resolution.
* Quality control tests for new versions or post installation acceptance.
* New tool for horizontal profile analysis, based on [relative_dose_1d](https://github.com/LuisOlivaresJ/relative_dose_1d) package. 

