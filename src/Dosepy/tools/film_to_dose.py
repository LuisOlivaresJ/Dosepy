# -*- coding: utf-8 -*-
"""

Última modificación: 22 Agosto 2021
@author:
    Luis Alfonso Olivares Jimenez
    Maestro en Ciencias (Física Médica)
    Físico Médico en Radioterapia, La Paz, Baja California Sur, México.

"""

import numpy as np
from scipy.optimize import curve_fit
import tifffile as tif
import matplotlib.pyplot as plt

def cubico(x,a,b,c,d):
    return a + b*x + c*x**2 + d*x**3

def calibracion(img_pre, img_post):
    film_pre_prom = np.mean(img_pre, axis = 2)  # Promedio de los tres canales de color RGB
    film_post_prom = np.mean(img_post, axis = 2)

    #Dosis_impartida = np.array([0, 0.5, 1, 2, 3, 5, 7, 10])
    Dosis_impartida = np.array([0, 0.502, 1.004, 2.000, 4.000, 6.000, 7.999, 9.999, 11.999, 13.999])

    Pix_mean_pre = np.zeros((5,2))
    Pix_mean_post = np.zeros((5,2))
    Dens_optica = np.zeros((5,2))
    colm = [300, 1000] #   Posición del centro de las peliculas, COLUMNAS
    spac = 600          #   Espacio entre filas para el centro entre películas
#    Dens_optica[0,1] = film_DO[110 - 35: 110 + 35, 110 - 35: 110 + 35]
    for i in np.arange(5):
        for j in np.arange(2):
            Pix_mean_pre[i,j] = np.mean( film_pre_prom[ i*600 + (200 - 35): i*600 + (200 + 35), j*700 +  (300 - 35): j*700 + ( 300 + 35)] )
            Pix_mean_post[i,j] = np.mean( film_post_prom[ i*600 + (200 - 35): i*600 +  (200 + 35), j*700 + (300 - 35): j*700 + ( 300 + 35)] )
            Dens_optica[i,j] = -np.log10(Pix_mean_post[i,j] / (Pix_mean_pre[i,j] + 1))
#    temporal = np.zeros((5,2))
#    temporal[:,0] = Dens_optica[:,1]
#    temporal[:,1] = Dens_optica[:,0]
    Dens_optica_vec = np.matrix.flatten(Dens_optica)
    popt, pcov = curve_fit(cubico, Dens_optica_vec, Dosis_impartida)
    return popt, Dens_optica_vec, Dosis_impartida

def a_dosis(Pre, Post, popt):
    film_pre_prom = np.mean(Pre, axis = 2)  # Promedio de los tres canales de color
    film_post_prom = np.mean(Post, axis = 2)
    film_DO = -np.log10(film_post_prom / (film_pre_prom + 1))   #La suma de uno se utiliza para evitar la división por cero
    Dosis_FILM_full = cubico(film_DO, popt[0], popt[1], popt[2], popt[3])

    return Dosis_FILM_full

def main():
    Img_pre = tif.imread('C_Pre.tif')
    Img_post = tif.imread('C_Post.tif')
    popt, Dens_optica, Dosis_imaprtida = calibracion(Img_pre,Img_post)
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
