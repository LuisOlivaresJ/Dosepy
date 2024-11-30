# Functions for curve fitting and film response
# =============================================

import numpy as np

# Functions used for curve fitting

def polynomial_g3(x, a, b, c, d):
    """
    Polynomial function of degree 3.
    """
    return a + b*x + c*x**2 + d*x**3


def polynomial_n(x, a, b, n):
    """
    Polynomial function of degree n.
    """
    return a*x + b*x**n


def rational_function(x, a, b, c):
    """
    Rational function.
    """
    return -c + b/(x-a)


# Functions used for film response

def optical_density(i, i_0):
    """
    Compute the optical density.

    Parameters
    ----------
    i : ndarray
        Intensity (irradiated film).
    i_0 : float
        Reference intensity (unirradiated film).
    """

    return -np.log10(i/i_0)


def uncertainty_optical_density(i, std_i, i_0, std_i_0):
    """
    Compute the uncertainty of the optical density of the film.

    Parameters
    ----------
    i : ndarray
        Intensity (irradiated film).
    i_0 : float
        Reference intensity (unirradiated film).
    std_i : float
        The standard deviation of the intensity.
    std_i_0 : float
        The standard deviation of the reference intensity.
    """

    return ( 1/np.log(10))*np.sqrt( (std_i/i)**2 + (std_i_0/i_0)**2 )


def ratio(i, i_0):
    """
    Compute the ratio of the pixel values.

    Parameters
    ----------
    i : ndarray
        Intensity (irradiated film).
    i_0 : float
        Reference intensity (unirradiated film).
    """
    return i/i_0


def uncertainty_ratio(i, std_i, i_0, std_i_0):
    """
    Compute the uncertainty of the ratio of the pixel values.

    Parameters
    ----------
    i : ndarray
        Intensity (irradiated film).
    i_0 : float
        Reference intensity (unirradiated film).
    std_i : float
        The standard deviation of the intensity.
    std_i_0 : float
        The standard deviation of the reference intensity.
    """
    return (i/i_0)*np.sqrt( (std_i/i)**2 + (std_i_0/i_0)**2 )