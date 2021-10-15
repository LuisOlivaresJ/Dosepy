---
title: "Dosepy"
---

![Portada_Dosepy](/assets/portada_DOSEPY.png)

# Bienvenido
*Dosepy* es un paquete escrito en Python para la comparación de distribuciones de dosis 2-dimensional usadas en radioterapia.<br/>
Para su uso, se puede emplear una interfaz gráfica incluida dentro del paquete. Sin embargo, para tener acceso a todas las funcionalidades de Dosepy, se requiere un intérprete de python (por ejemplo, escribiendo Python dentro de una terminal Linux, o utilizando el entorno [Spyder](https://www.spyder-ide.org)).


## Métodos de comparación

**¡Consideraciones!**

* Ambas distribuciones deben tener las mismas dimensiones físicas y resolución espacial (mismo número de filas y columnas).
* Las distribuciones deben de  encontrarse registradas, es decir, la coordenada espacial de un punto en la distribución de referencia debe ser igual a la coordenada del mismo punto en la distribución a evaluar.<br/>

En caso contrario, *Dosepy* dispone de algunas funciones para cumplir con lo anterior.

**Comparación por índice gamma**<br/>
![Imagen_gamma](/assets/Image_gamma.png)<br/>
La comparación de dos distribuciones puede realizarse mediante la prueba del índice gamma 2-dimensional de acuerdo a la definición dada por [Low D. A.](https://doi.org/10.1118/1.598248) así como algunas recomendaciones del [TG-218]( https://doi.org/10.1002/mp.12810) de la AAPM:

* El criterio de aceptación para la diferencia en dosis puede ser seleccionado en modo absoluto (en Gy) o relativo.
  * En modo relativo, el porcentaje se interpreta con respecto al máximo de la distribución de dosis (global normalization), o con respecto a la dosis local (local normalization), según la selección del usuario.
* El umbral de dosis puede ser ajustado por el usuario.
* La distribución de referencia puede ser seleccionada por el usuario.
* Se permite definir un radio de búsqueda como proceso de optimización para el cálculo.
* Es posible utilizar el percentil 99.1 de la distribución de dosis como una aproximación del valor máximo. Esto permite evitar la posible inclusión de artefactos o errores en posiciones puntuales de la distribución (de utilidad por ejemplo cuando se utiliza película radiocrómica).
* No se realiza interpolación entre puntos.

**Comparación mediante perfiles**<br/>
![Imagen_perfil_1](/assets/Perfiles_1.png)<br/>
![Imagen_perfil_2](/assets/Perfiles_2.png)<br/>
También es posible comparar dos distribuciones de dosis mediante perfiles verticales y horizontales. La posición de cada perfil debe seleccionarse con ayuda de la interfaz gráfica.

## Instalación
**En Linux**<br/>
El método más sencillo para instalar *Dosepy* es escribiendo en una terminal:
```bash
pip install Dosepy
```
**En Windows**<br/>
Previo a la instalación de *Dosepy*, es necesario contar con un administrador de paquetes. Para quienes no estén familiarizados con los paquetes Python, se recomienda utilizar la plataforma [ANACONDA](https://www.anaconda.com/products/individual).<br/>
Una vez que se ha instalado ANACONDA, abrir el inicio de Windows y buscar *Anaconda Prompt*. Dentro de la terminal (ventana con fondo negro), seguir la indicación descrita para Linux (párrafo anterior).

**Requisitos**<br/>
Dosepy depende de otros paquetes Python (también de código abierto). Para su instalación escribir en una terminal:
```bash
pip install numpy, pydicom, matplotlib, tifffile, scipy, PyQt5
```

**Versión Beta**<br/>
Dosepy se encuentra en una versión beta, especificada por el formato 0.X.X. Lo anterior implica que en la práctica, un código que utiliza el paquete Dosepy en una versión, pudiera no ser ejecutado en una versión posterior.  La versión estable será publicada con el formato 1.X.X.<br/>
Para mantener actualizado el paquete Dosepy, utilizar [pip](https://pip.pypa.io/en/stable/):
```bash
pip install --upgrade Dosepy
```

**Ayuda**<br/>
Si tienes algún problema o duda respecto al uso del paquete Dosepy, permítenos saberlo.<br/>
Escribe a la dirección de correo electrónico: alfonso.cucei.udg@gmail.com

### Ejemplos

**Ejemplo con interfaz gráfica**

Para utilizar *Dosepy*, abrimos una terminal (o Anaconda Prompt en el caso de Windows) y escribimos el comando **python**:

```bash
python
```

Para abrir la interfaz gráfica, escribimos:

```python
import Dosepy.GUI
```

Dosepy.GUI viene pre-cargado con dos distribuciones de dosis con el objetivo de que el usuario pueda interactuar con las herramientas que se ofrecen para la comparación.<br/>

**Importación de archivo en formato csv**

La importación de la distribución de referencia puede realizarse sólo si el archivos se encuentra en formato .csv (valores separados por comas). Adicionalmente:
* El archivo deberá contener sólo los valores de dosis.
* Toda información adicional deberá estar precedida con el carácter "#". Ello indicará que todos los caracteres que se encuentren en la misma linea después de "#" debe de ser ignorados por Dosepy.
* La unidad para la dosis deberá ser el Gray (Gy).

**Importación de archivo en formato dcm**

La distribución a evaluar puede importarse en un archivo con formato .csv o en formato .dcm (archivo DICOM). Si el formato es DICOM:
* Deberá contener sólo un plano de dosis.
* La resolución espacial debe ser igual en cada dimensión.
* La unidad para la dosis deberá ser el Gray (Gy).

**Ejemplo utilizando una terminal**

En *Dosepy*, una distribución de dosis es representada como un objeto de la [clase](https://docs.python.org/es/3/tutorial/classes.html) **Dose** del paquete *Dosepy*. Para crear el objeto son necesarios dos argumentos: las dosis de la distribución en formato [ndarray](https://numpy.org/doc/stable/reference/index.html#module-numpy) y la resolución espacial dada por la distancia (en milímetros) entre dos puntos consecutivos.
Para utilizar *Dosepy*, abrimos una terminal (o Anaconda Prompt en el caso de Windows) y escribimos el comando *python*:

```bash
python
```

Dentro de Python, escribimos el siguiente código de prueba:

```python
import numpy as np
import Dosepy.dose as dp

a = np.zeros((10,10)) + 100   # Matrices de prueba
b = np.zeros((10,10)) + 96    # Diferencia en dosis de un 4 %

D_ref = dp.Dose(a, 1)   # Se crea la distribución de referencia
D_eval = dp.Dose(b, 1)  # Se crea la distribución a evaluar
```

La comparación gamma entre dos distribuciones de dosis se realiza mediante el método *gamma2D*. Como argumentos se requiere:
* La distribución de dosis de referencia
* El porcentaje para la diferencia en dosis de tolerancia
* La distancia de tolerancia o criterio DTA en mm.

```python
#   Llamamos al método gamma2D, con criterio 3 %, 1 mm.
gamma_distribution, pass_rate = D_eval.gamma2D(D_ref, 3, 1)
print(pass_rate)
```

**Datos en formato CSV, usando un umbral de dosis**

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
g, pass_rate = D_eval.gamma2D(D_ref, dose_t= 3, dist_t = 2, dose_tresh = 10)

#   Imprimimos el resultado
print(f'El índice de aprobación es: {pass_rate:.1f} %')
plt.imshow(g, vmax = 1.4)
plt.show()

#El índice de aprobación es: 98.9 %

```
**Datos en formato DICOM y modo de dosis absoluto**

Importación de un archivo de dosis en formato DICOM

*Consideraciones*

* La distribución de dosis en el archivo DICOM debe contener solo dos dimensiones (2D).
* El espacio entre dos puntos (pixeles) debe de ser igual en ambas dimensiones.
* No se hace uso de las coordenadas dadas en el archivo DICOM. Ver primera consideración en el apartado *Comparación por índice gamma*.

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

## Dosimetría con película

**Calibración**

Para obtener la curva de calibración se utilizan 10 películas de
4 cm x 5 cm, 9 de ellas irradiadas con dosis de 0.50, 1.00, 2.00,
4.00, 6.00, 8.00, 10.00, 12.0 y 14.00 Gy.

La digitalización de las películas antes y después de su irradiación,
deberá ser de tal modo que en la imagen se obtenga el acomodo mostrado
en la Figura 1, utilizando los siguientes parámetros:

* Resolución espacial: 	300 puntos por pulgada
* Composición: 	RGB
* Bits: 	16 por canal
* Formato: 	TIFF

![Cal_Peliculas](/assets/calibracion_t.png)<br/>
Figura 1. Arreglo para la digitalización de las películas.

Con ayuda del software [ImageJ](https://imagej.net/software/fiji/) (o cualquier otro programa), recortar la imagen hasta obtener un tamaño de 11 cm por 24 cm (1,300 por 2835 pixeles). Las siguientes ligas permiten descargar imágenes de muestra.

[Calib_Pre.tif](https://github.com/LuisOlivaresJ/Dosepy/blob/60aa1ccaa4155f19db3b063f8e782b47ffde6828/docs/film_dosimetry/Calib_Pre.tif)<br/>
[Calib_Post.tif](https://github.com/LuisOlivaresJ/Dosepy/blob/60aa1ccaa4155f19db3b063f8e782b47ffde6828/docs/film_dosimetry/Calib_Post.tif)<br/>

La calibración de la película se realiza ingresando a Dosepy dos imágenes del mismo tamaño, correspondientes a las plículas antes y después de su irraciación. Para ello, seguir los siguientes pasos:

   1. Abrir el software Dosepy.GUI
   2. En la pestaña *Herramientas*, haga clic en la opción *Dosimetría con película*.
   3. Al dar clic en el botón *Calib.*, seleccione la imagen en formato tiff correspondiente al arreglo de las 10 películas sin irradiar.
   4. Automáticamente se mostrará una nueva ventana. Seleccione la imagen tiff de las películas después de su irradiación.
   5. Se mostrará el ajuste y los correspondientes coeficientes de la curva.

![Curva_Calibracion](/assets/img_calib.png)<br/>
Figura 2. Curva de calibración. La línea azul representa un ajuste polinomial de grado 3. En color verde los 10 datos obtenidos de las imágenes tiff.

**Aplicar calibración a una imagen**

La curva de calibración previamente generada puede ser aplicada a un imagen en formato tif. (Los parámetros para la digitalización deben ser los mismos que los usados para la calibración). Para ello se requieren cargar la imagen de la película antes de la irradiación y una segunda imagen del mismo tamaño después de la irradiación.

[Imagen_ejemplo_PRE.tif](https://github.com/LuisOlivaresJ/Dosepy/blob/b6510eac7b65285b39d9b5c7fa6a24487f991db6/docs/film_dosimetry/QA_Pre.tif)<br/>
[Imagen_ejemplo_POST.tif](https://github.com/LuisOlivaresJ/Dosepy/blob/b6510eac7b65285b39d9b5c7fa6a24487f991db6/docs/film_dosimetry/QA_Post.tif)<br/>

1. Dar clic en el botón Dist.
2. Seleccionar la imagen tif de la película antes de su irradiación
3. En la ventana emergente, seleccionar la imagen tif de la película después de la irradiación.

![disutribucion](/assets/distribucion.png)<br/>

El número de filas y columnas de la distribución obtenida (distribución A) puede ser modificado con el objetivo de igualar al tamaño de otra distribución (B) de menor tamaño. Para ello se utiliza la resolución espacial de la distribución B.

1. En la opción Ref., ingresar la resolución espacial en mm/punto de la distribución B.
2. Dar clic en el botón *Reducir*
3. Automáticamente, se mostrará la distribución de dosis con un menor número de filas y columnas.

## Documentación del código
[dose.py](https://github.com/LuisOlivaresJ/Dosepy/blob/2ffb694187298b05c91b5489926897f17981c01c/src/Dosepy/dose.py)
```
Dosepy.dose.Dose(data, resolution)
  Clase para la representación de una distribución de dosis absorbida.
  Regresa un objeto Dose que contiene la distribución de dosis y la
  resolución espacial.

  Parameters:

     data : numpy.ndarray
         Arreglo o matriz de datos. Cada valor numérico representa la
         dosis absorbida en un punto en el espacio.

     resolution : float
         Resolución espacial dada como la distancia física (en milímetros)
         entre dos puntos consecutivos.

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
Se obtiene una matriz que representa los índices gamma en cada posición
de la distribución de dosis, así como el índice de aprobación definido
como el porcentaje de valores gamma que son menores o iguales a 1.

Consideraciones:
Se asume el registro de las distribuciones de dosis, es decir,
que la coordenada espacial de un punto en la distribución de referencia
es igual a la coordenada del mismo punto en la distribución a evaluar.

Parameters:
    D_reference : Objeto Dose
        Distribución de dosis de referencia contra la cual se
        realizará la comparación.
        El número de filas y columnas debe de ser igual a la
        distribución a evaluar.
        Lo anterior implica que las dimesiones espaciales de
        las distribuciones deben de ser iguales.

    dose_t : float, default = 3
        Tolerancia para la diferencia en dosis.
        Este valor puede interpretarse de 3 formas diferentes, según
        los parámetros dose_t_Gy y local_norm que se describen más adelante.

    dist_t : float, default = 3
        Tolerancia para la distancia, en milímetros (criterio DTA).

    dose_tresh : float, default = 10
        Umbral de dosis, en porcentaje (0 a 100). Todo punto en la
        distribución de dosis con un valor menor al umbral de dosis,
        es excluido del análisis.
        Por default, el porcentaje se interpreta con respecto al
        percentil 99.1 (aproximadamente el máximo) de la distribución
        a evaluar. Si el porcentaje se requiere con respecto al máximo,
        modificar el parámetro max_as_percentile = False (ver más adelante).

    dose_t_Gy : bool, default: False
        Si el argumento es True, entonces "dose_t" (la dosis de tolerancia)
        se interpreta como un valor fijo y absoluto en Gray [Gy].
        Si el argumento es False (default), "dose_t" se interpreta como
        un porcentaje.

    local_norm : bool, default: False
        Si el argumento es True (local normalization), el porcentaje
        de dosis de tolerancia "dose_t" se interpreta con respecto a
        la dosis local.
        Si el argumento es False (global normalization), el porcentaje
        de dosis de tolerancia "dose_t" se interpreta con respecto al
        máximo de la distribución a evaluar.
          Nota:
              1.- Los argumentos dose_t_Gy y local_norm NO deben ser
              seleccionados como True de forma simultánea.
              2.- Si se desea utilizar directamente el máximo de la
              distribución, utilizar el parámetro max_as_percentile = False
              (ver más adelante)

    mask_radius : float, default: 5
        Distancia física en milímetros que se utiliza para acotar el
        cálculo con posiciones que estén dentro de una vecindad dada
        por mask_radius.

        Para lo anterior, se genera un área de busqueda cuadrada o
        "máscara" aldrededor de cada punto o posición en la distribución
        de referencia.
        El uso de esta máscara permite reducir el tiempo de cálculo
        debido al siguiente proceso:
            Por cada punto en la distribución de referencia, el cálculo
            de la función Gamma se realiza solamente con aquellos puntos
            o posiciones de la distribución a evaluar que se encuentren
            a una distancia relativa menor o igual a mask_radius, es
            decir, con los puntos que están dentro de la vecindad dada
            por mask_radius.
            La longitud de uno de los lados de la máscara cuadrada es
            de 2*mask_radius + 1.
            Por otro lado, si se prefiere comparar con todos los puntos
            de la distribución a evaluar, es suficiente con ingresar
            una distancia mayor a las dimensiones de la distribución de
            dosis (por ejemplo mask_radius = 1000).

    max_as_percentile : bool, default: True
        -> Si el argumento es True, se utiliza el percentil 99.1 como una
           aproximación del valor máximo de la distribución de dosis. Lo
           anterior permite excluir artefactos o errores en posiciones
           puntuales (de utilidad por ejemplo cuando se utiliza película
               radiocrómica).
        -> Si el argumento es False, se utiliza directamente el valor
           máximo de la distribución.

Retorno:

    ndarray :
          Array, o matriz bidimensional con la distribución de índices
          gamma.

    float :
          Índice de aprobación. Se calcula como el porcentaje de valores
          gamma <= 1, sin incluir las posiciones en donde la dosis es
          menor al umbral de dosis.



```

**Funciones**

[dose.py](https://github.com/LuisOlivaresJ/Dosepy/blob/2ffb694187298b05c91b5489926897f17981c01c/src/Dosepy/dose.py)

```

Dosepy.dose.from_csv(file_name, PixelSpacing)

    Importación de un archivo de dosis en formato CSV (Comma separated values).
    Dentro del archivo .csv, utilizar el caracter # al inicio de una fila para
    que sea descartada (inicio de un comentario).

    Parameters
    -----------
    file_name : str
        Nombre del archivo en formato string

    PixelSpacing : float
        Distancia entre dos puntos consecutivos, en milímetros.

    Return
    --------
    Dosepy.dose.Dose
        Objeto Dose del paquete Dosepy que representa a la
        distribución de dosis.



Dosepy.dose.from_dicom(file_name)

    Importación de un archivo de dosis en formato DICOM

    Parameters
    -----------
    file_name : str
        Nombre del archivo en formato string

    Return
    --------
    Dosepy.dose.Dose
        Objeto Dose del paquete Dosepy que representa a la
        distribución de dosis

    Consideraciones
    ----------------
        La distribución de dosis en el archivo DICOM debe contener solo
        dos dimensiones.
        La resolución espacial debe de ser igual en ambas dimensiones.
        No se utilizan las coordenadas dadas en el archivo DICOM.
        Ver consideraciones en la nota del método gamma2D de la
        clase Dose.

```
[resol.py](https://github.com/LuisOlivaresJ/Dosepy/blob/2ffb694187298b05c91b5489926897f17981c01c/src/Dosepy/tools/resol.py)
```

Dosepy.tools.resol.equalize(array, resol_array, resol_ref)
    """
    Función que permite reducir el número de filas y columnas de una matriz
    (array) para igualar su resolución espacial (mm/punto) con respecto a una
    resolución de referencia.
    Para lo anterior, se calcula un promedio de varios puntos y se asigna a
    un nuevo punto con una mayor dimensión espacial.

    Parameters:
    -----------
    array: ndarray
        Matriz a la que se le requiere reducir el tamaño.

    resol_array: float
        Resolución espacial de la matriz, en milímetros por punto.

    resol_ref: float
        Resolución espacial de referencia, en milímetros por punto.

    Retorno:
  	--------
    array: ndarray
  			Matriz reducida en el número de filas y columnas.        

    Ejemplo:
    --------

        Sean A y B dos matrices de tamaño (2362 x 2362) y (256 x 256), con
        resolución espacial de 0.0847 mm/punto y 0.7812 mm/punto, respectivamente.

        La dimensión espacial de la matriz A es de 200.06 mm
        (2362 puntos * 0.0847 mm/punto = 200.06 mm)
        La dimensión espacial de la matriz B es de 199.99 mm.
        (256 puntos * 0.7812 mm/punto = 199.99 mm)

        Para reducir el tamaño de la matriz A e igualarla al tamaño de la
        matriz B, se utiliza la función equalize:

            import Dosepy.tools.resol as resol
            import numpy as np

            A = np.zeros( (2362, 2362) )

            C = resol.equalize(A, 0.0847, 0.7812)
            C.shape
            # (256, 256)

```

[film_to_dose.py](https://github.com/LuisOlivaresJ/Dosepy/blob/2ffb694187298b05c91b5489926897f17981c01c/src/Dosepy/tools/film_to_dose.py)

```
Dosepy.tools.film_to_dose.calibracion(img_pre, img_post)

    Función que permite generar una curva de calibración para transformar
    densidad óptica a dosis usando película radiocrómica.
    La calibración se genera a partir de dos imágenes de 10 películas
    antes y después de su exposición a diferentes dosis.

    Ambas imágenes deben de tener un tamaño de (1300, 2835, 3)
    en modo RGB.

    El centro de cada película deberá encontrarse en las siguientes
    posiciones (x,y) -> (fila, columna)

    1.- ( 200, 300)      2.- ( 200, 1000)
    3.- ( 800, 300)      4.- ( 800, 1000)
    5.- (1400, 300)      6.- (1400, 1000)
    7.- (2000, 300)      8.- (2000, 1000)
    9.- (2600, 300)     10.- (2600, 1000)

    Parámetros
    -----------
    img_pre : numpy.ndarray
        Arreglo matricial de datos 3-dimensional que representan
        a una imagen en modo RGB.
        La imagen debe de contener las 10 películas no irradiadas.

    img_post : numpy.ndarray
        Arreglo matricial de datos 3-dimensional que representan a
        una imagen en modo RGB.
        La imagen debe de contener las 10 películas irradiadas con
        los siguientes valores de dosis:

        1.-  0.00 Gy
        2.-  0.50 Gy
        3.-  1.00 Gy
        4.-  2.00 Gy
        5.-  4.00 Gy
        6.-  6.00 Gy
        7.-  8.00 Gy
        8.-  10.00 Gy
        9.-  12.00 Gy
        10.- 14.00 Gy


    Retorno
    --------
    popt : ndarray
        Coeficientes (a0, a1, a2 y a3) correspondientes a un polinómio
        de tercer grado (a0 + a1*x + a2*x^2 + a3*x^3).

    Dens_optica_vec : ndarray
        Densidad óptica de cada una de las 10 películas. Calculada como
        DO = - np.log10( I_post / (I_pre + 1)),
        en donde I_pre e I_post corresponden a la intensidad de pixel
        promedio en una ROI cuadrada de 70 pixeles de lado,
        para una película antes y después de su irradiación,
        respectivamente.

    Dosis_impartida : numpy.ndarray
        Valores de dosis impartida a cada película.

```


### Advertencia
El correcto funcionamiento del paquete se está evaluado y actualizado constantemente. Sin embargo, no se tiene garantía de que el código del paquete esté libre de errores o bugs. El usuario es el único responsable por utilizar *Dosepy*.

**Historia**<br/>
01-05-2019<br/>
  * *Dosepy* fue escrito por primera vez como parte de un desarrollo de [tesis](https://tesiunam.dgb.unam.mx/F/XL85Q4MGCIBS9NX72MY22SV9KYALL7VBBUFF2A5PCG96BP962B-26621?func=find-b&local_base=TES01&request=Luis+Alfonso+Olivares+Jimenez&find_code=WRD&adjacent=N&filter_code_2=WYR&filter_request_2=&filter_code_3=WYR&filter_request_3=) a nivel de Maestría en el año 2019, con el objetivo de comparar y evaluar distribuciones de dosis en radioterapia. Para ello se emplearon diferentes herramientas como perfiles, evaluación gamma e histogramas dosis volumen. La medición de las distribuciones de dosis se realizó con película radiocrómica EBT3.

28-06-2021  Versión 0.0.1<br/>
  * *Dosepy* se incorpora al índice de paquetes python [PyPi](https://pypi.org/)

01-07-2021  Versión 0.0.3<br/>
  * Se agregan las funciones from_csv y from_dicom para la lectura de datos.

16-07-2021  Versión 0.0.4<br/>
  * Se modifica el formato para el parámetro resolution. Se agregas indicaciones más detalladas para la instalación del paquete *Dosepy*

24-07-2021  Versión 0.0.8<br/>
  * Se agrega la posibilidad de usar una interfaz gráfica.

03-08-2021  Versión 0.1.0<br/>
  * Se agrega una página web con instrucciones y documentación para el uso del paquete Dosepy.

12-08-2021  Versión 0.1.1<br/>
  * Se agrega la carpeta tools junto con la función *equalize* del modulo resol, para modificar la resolución espacial de una distribución e igualarla a una de referencia.   

01-09-2021  Versión 0.2.1<br/>
  * Se agrega el menú "Herramientas" dentro de la interfaz gráfica para la dosimetría con película radiocrómica.
