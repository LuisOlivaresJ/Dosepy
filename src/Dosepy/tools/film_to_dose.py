# -*- coding: utf-8 -*-
"""
@author:
    Luis Alfonso Olivares Jimenez

    Script for film dosimetry.
"""

import numpy as np
from scipy.optimize import curve_fit
import tifffile as tif
import matplotlib.pyplot as plt

def cubico(x,a,b,c,d):
    """
    Función polinomial de tercer grado usada para el ajuste de la curva de calibración.
    """
    return a + b*x + c*x**2 + d*x**3

def calibracion(img_pre, img_post):
    """
    Función que permite generar una curva de calibración para transformar densidad óptica a dosis usando película radiocrómica.
    La calibración se genera a partir de dos imágenes de 10 películas antes y después de su exposición a diferentes dosis.

    Ambas imágenes deben de tener un tamaño de (1300, 2835, 3) en modo RGB.

    El centro de cada película deberá encontrarse en las siguientes posiciones (x,y) -> (fila, columna)

    1.- ( 200, 300)      2.- ( 200, 1000)
    3.- ( 800, 300)      4.- ( 800, 1000)
    5.- (1400, 300)      6.- (1400, 1000)
    7.- (2000, 300)      8.- (2000, 1000)
    9.- (2600, 300)     10.- (2600, 1000)

    Parámetros
    -----------
    img_pre : numpy.ndarray
        Arreglo matricial de datos 3-dimensional que representan a una imagen en modo RGB.
        La imagen debe de contener las 10 películas no irradiadas.

    img_post : numpy.ndarray
        Arreglo matricial de datos 3-dimensional que representan a una imagen en modo RGB.
        La imagen debe de contener las 10 películas irradiadas con los siguientes valores de dosis:

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
        Coeficientes (a0, a1, a2 y a3) correspondientes a un polinómio de tercer grado (a0 + a1*x + a2*x^2 + a3*x^3).

    Dens_optica_vec : ndarray
        Densidad óptica de cada una de las 10 películas. Calculada como DO = - np.log10( I_post / (I_pre + 1)),
        en donde I_pre e I_post corresponden a la intensidad de pixel promedio en una ROI cuadrada de 70 pixeles de lado,
        para una película antes y después de su irradiación, respectivamente.

    Dosis_impartida : numpy.ndarray
        Valores de dosis impartida a cada película.

    """
    
    film_pre_prom = np.mean(img_pre, axis = 2)  # Promedio de los tres canales de color RGB
    film_post_prom = np.mean(img_post, axis = 2)

    Dosis_impartida = np.array([0, 0.50, 1.00, 2.00, 4.00, 6.00, 8.00, 10.00, 12.00, 14.00])

    Pix_mean_pre = np.zeros((5,2))      #   Para almacenar el valor promedio de una ROI por cada película
    Pix_mean_post = np.zeros((5,2))
    Dens_optica = np.zeros((5,2))
    colm = [300, 1000] #   Posición del centro de las peliculas, COLUMNAS
    spac = 600          #   Espacio entre filas para el centro entre películas

    for i in np.arange(5):
        for j in np.arange(2):
            Pix_mean_pre[i,j] = np.mean( film_pre_prom[ i*600 + (200 - 35): i*600 + (200 + 35), j*700 +  (300 - 35): j*700 + ( 300 + 35)] )
            Pix_mean_post[i,j] = np.mean( film_post_prom[ i*600 + (200 - 35): i*600 +  (200 + 35), j*700 + (300 - 35): j*700 + ( 300 + 35)] )
            Dens_optica[i,j] = -np.log10(Pix_mean_post[i,j] / Pix_mean_pre[i,j])

    Dens_optica_vec = np.matrix.flatten(Dens_optica)
    popt, pcov = curve_fit(cubico, Dens_optica_vec, Dosis_impartida)
    return popt, Dens_optica_vec, Dosis_impartida

def a_dosis(Pre, Post, popt):
    film_pre_prom = np.mean(Pre, axis = 2)  # Promedio de los tres canales de color
    film_post_prom = np.mean(Post, axis = 2)
    film_DO = -np.log10(film_post_prom / film_pre_prom )
    Dosis_FILM_full = cubico(film_DO, popt[0], popt[1], popt[2], popt[3])

    return Dosis_FILM_full

def main():
    Img_pre = tif.imread('C_Pre.tif')
    Img_post = tif.imread('C_Post.tif')
    popt, Dens_optica, Dosis_imaprtida = calibracion(Img_pre, Img_post)
    x = np.linspace(0, 0.35)
    y = cubico(x, popt[0], popt[1], popt[2], popt[3])
    plt.plot(Dens_optica, Dosis_imaprtida,'o')
    plt.plot(x,y)
    plt.xlabel('Dens. Óptica')
    plt.ylabel('Dosis [Gy]')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
