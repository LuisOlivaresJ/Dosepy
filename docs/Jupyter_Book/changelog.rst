
=========
Changelog
=========

V 0.4.0 (AUG-2023)
-------------------

* On the GUI, dose objects are created after loading. 
* When a csv file is open, a new window shows to ask for resolution.
* Quality control tests for new versions or post installation acceptance.
* New tool for horizontal profile analysis, based on `relative_dose_1d. <https://github.com/LuisOlivaresJ/relative_dose_1d package>`_
.. image:: ../assets/Relative_dose_1d_incorporation.PNG
   :scale: 50 %

V 0.3.8 (MAY-2023)
-------------------

* Firsts steps for spanish to english documentation using Read The Docs Documentation.

V 0.3.7 (MARCH-2023)
--------------------

* Se resuelve `error <https://github.com/LuisOlivaresJ/Dosepy/issues/32>`_ de ejecución con archivos de prueba. Se define una vecindad de 2 cm x 2 cm para reducir el tiempo de cálculo para el índice gamma. En la `página principal <https://luisolivaresj.github.io/Dosepy/>`_ de Dosepy, se muestra la validación para la dosimetría con película al medir los factores de campo de un haz 6FFF.

V 0.3.6 (FEB-2023)
------------------

* Se modifica el algoritmo de dosimetría con película. Se agrega Notebook para dosimetría con película. En la evaluación gamma, se habilita la opción para definir la dosis máxima como el percentil 99.1 de la distribución de dosis a evaluar. Se agrega información referente al uso no clínico del software Dosepy.

V 0.3.3-5 (JAN-2023)
--------------------

* Se modifican los nombres de los archivos y las clases para facilitar el mantenimiento del paquete. Se agrega guía de uso con Jupyter-Notebook.

V 0.3.2 (SEP-2022)
------------------

* Se inhabilita como primera opción el botón para abrir la distribución de dosis a evaluar. Se inhabilita el botón para calcular la distribución gamma si los parámetros ingresados por el usuario no son valores numéricos. Lo anterior para evitar un error de ejecución. 

V 0.3.1 (OCT-2022)
------------------

* Se actualiza la licencia. Se facilita la instalación al agregarse automáticamente las dependencias como numpy, matplotlib, etc. Se agrega información del tamaño de las distribuciones de dosis cuando se comparan matrices con diferentes dimensiones. Al guardar una distribución de dosis, se resuelve el error de generarse el nombre del archivo con doble formato (por ejemplo file.csv.csv). Se mejora el ingreso del parámetro "Ref." para ejecutar cambio de resolución solo cuando el valor ingresado por el usuario es un número flotante.

V 0.3.0 (JUL-2022)
------------------

* Se agrega un resumen del trabajo de validación del software. Se requiere de un password para utilizar el software.

Versión 0.2.3 (OCT-2021)
------------------------

* Se modifica la LICENCIA por derechos de autor. Se agrega video de presentación en congreso.

V 0.2.2 (OCT-2021)
------------------

* Se agrega el menú "Ayuda" para mostrar la versión y un link para la Documentación

V 0.2.1 (SEP-2021)
------------------

* Se agrega el menú "Herramientas" dentro de la interfaz gráfica para la dosimetría con película radiocrómica.

V 0.1.1 (AUG-2021)
------------------

* Se agrega la carpeta tools junto con la función *equalize* del modulo resol, para modificar la resolución espacial de una distribución e igualarla a una de referencia.

V 0.1.0 (AUG-2021)
------------------

* Se agrega una página web con instrucciones y documentación para el uso del paquete Dosepy.

V 0.0.8 (JUL-2021)
------------------

* Se agrega la posibilidad de usar una interfaz gráfica

V 0.0.4 (JUL)
-------------

* Se modifica el formato para el parámetro resolution. Se agregas indicaciones más detalladas para la instalación del paquete *Dosepy*

V 0.0.3 (JUL-2021)
------------------

* Se agregan las funciones from_csv y from_dicom para la lectura de datos.

V 0.0.1 (JUN-2021)
------------------

* *Dosepy* se incorpora al índice de paquetes python `PyPi <https://pypi.org/>`_.

01-MAY-2019
-----------

* *Dosepy* fue escrito por primera vez como parte de un desarrollo de `tesis <https://tesiunam.dgb.unam.mx/F/8V8RPCG2P1P85AN4XJ33LCS6CRT3NEL72J8IQQYUAKMESPGRGS-06398?func=find-b&local_base=TES01&request=Luis+Alfonso+Olivares+Jimenez&find_code=WRD&adjacent=N&filter_code_2=WYR&filter_request_2=&filter_code_3=WYR&filter_request_3=>`_ a nivel de Maestría en el año 2019, con el objetivo de comparar y evaluar distribuciones de dosis en radioterapia. Para ello se emplearon diferentes herramientas como perfiles, evaluación gamma e histogramas dosis volumen. La medición de las distribuciones de dosis se realizó con película radiocrómica EBT3.
