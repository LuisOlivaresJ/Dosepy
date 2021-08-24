# -*- coding: utf-8 -*-
"""

Última modificación: 09 Agosto 2021
@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico en Radioterapia, La Paz, Baja California Sur, México.

"""

#	Código que permite igualar el número de filas y columnas entre dos matrices
#	de tamaños diferentes.
#	Para lo anterior, se calcula un promedio de varios puntos y se asigna a un nuevo punto con
# 	una mayor dimensión espacial.
#

import numpy as np

def lista_puntos_prom(res_A, res_B, n_puntos_total):
	"""
	Función que permite generar una lista, en donde cada elemento representa
	el número de puntos que se deben de promediar para igualar las resoluciones.

	Parámetros:
	-----------
		res_A: float
			Resolución espacial dada como la distancia en mm, entre dos puntos.

		res_B: float
			Resolución espacial dada como la distancia en mm, entre dos puntos.

		n_pix_total: int
			Número de puntos del vector con mayor tamaño.

	Return:
	-------
		array: ndarray
			Matriz de datos con resolución espacial aumentada.

	"""
	if res_A > res_B:
		res_mayor = res_A
		res_menor = res_B

	else:
		res_mayor = res_B
		res_menor = res_A

	N_puntos_restantes = n_puntos_total
	n_puntos_prom = int( res_mayor // res_menor)	# Valor entero de la división
	residuo = res_mayor % res_menor
	n_puntos_prom_lista = []
	residuo_acumulado = residuo

	while N_puntos_restantes >= n_puntos_prom:
		if residuo_acumulado > res_menor/2:
			n_puntos_prom_lista.append(n_puntos_prom + 1)
			N_puntos_restantes -= (n_puntos_prom + 1)
			residuo_acumulado = residuo_acumulado - ( (n_puntos_prom + 1)*res_menor - res_mayor )
		else:
			n_puntos_prom_lista.append(n_puntos_prom)
			N_puntos_restantes -= n_puntos_prom
			residuo_acumulado += residuo

	if N_puntos_restantes > 0:
		n_puntos_prom_lista.append(N_puntos_restantes)
	return n_puntos_prom_lista

def equalize(array, resol_array, resol_ref):
	"""
	Función que permite reducir el número de filas y columnas de una matriz (array)
	para igualar su resolución espacial (mm/punto) con respecto a una resolución de referencia.

	Parámetros:
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
            Matriz reducida en su número de filas y columnas.

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

	"""

	lista_pix_colum = lista_puntos_prom(resol_array, resol_ref, array.shape[1])
	lista_pix_filas = lista_puntos_prom(resol_array, resol_ref, array.shape[0])
	array_nuevo_prom = np.zeros( (len(lista_pix_filas), len(lista_pix_colum) ) )
	f = 0		#Contador para número de fila
	for i in np.arange( len(lista_pix_filas) ):
		c = 0	#Contador para número de columna
		for j in np.arange( len(lista_pix_colum) ):
			DUMMY = array[ f : f+ lista_pix_filas[i], c : c + lista_pix_colum[j] ]
			array_nuevo_prom[i,j] = np.mean(DUMMY)
			c = c + lista_pix_colum[j]
		f = f + lista_pix_filas[i]
	return array_nuevo_prom



def main():         # Función para realizar pruebas
	a = lista_puntos_prom(6, 20, 13)
	print(a)

	import tifffile as tiff
	import matplotlib.pyplot as plt

	file_name = "Test_resolucion.tif"
	film = tiff.imread(file_name)
	film_prom = np.mean(film, axis = 2)

	tif_info = tiff.TiffFile(file_name)
	with tif_info as tif:
		tag = tif.pages[0].tags['XResolution']
	resol = int( tag.value[0] / tag.value[1] )

	res_ref = 200 / 256		# Resolución de referencia
	res_film = 25.4 / resol	# Resolución de la película en milímetros por punto

	Dosis_peli_prom = equalize( film_prom, res_film, res_ref )

	plt.imshow(Dosis_peli_prom)
	plt.show()

if __name__ == "__main__":
	main()
