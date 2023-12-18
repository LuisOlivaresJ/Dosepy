"""
Module for the management of the calibration curve.
The calibration, measurement, and unirradiated background films should be of the same model and 
production lot and the readout system and data acquisition procedures should be consistent across all films.
A 16-bits scanner is recomended. It measures the intensity of the transmitted or reflected light and scales PVs from zero 
to 65535 (= 2^16 - 1) wherethe limits are represented by complete darkness and the intensity of the 
unattenuated light source.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


"""Functions used for film calibration."""
def polynomial_g3(x,a,b,c,d):
    """
    Polynomial function of degree 3.
    """
    return a + b*x + c*x**2 + d*x**3

def rational_func(x, a, b, c):
    """
    Rational function.
    """
    return -c + b/(x-a)

class Calibration:
    """Class used to represent a calibration curve.
        
        Attributes
        ----------
        doses : list
            The doses values that were used to expose films for calibration.
        optical_density : list
            Optical density used for calibration.
        func : str
            The model function, f(x, …) used for intesity-dose relationship. 
            "P3": Polynomial function of degree 3.
            "RA": Rational function.
        channel : str
            Color channel. "R": Red, "G": Green and "B": Blue. 
        popt : array
            Parameters used by the function.
        pcov : 2-D array
            The estimated approximate covariance of popt. The diagonals provide the variance 
            of the parameter estimate. To compute one standard deviation errors on the parameters, 
            use perr = np.sqrt(np.diag(pcov)).
        """
    
    def __init__(self, doses: list, optical_density: list, func: str = "P3", channel: str = "R"):
        
        self.doses = sorted(doses)
        self.optical_density = sorted(optical_density)
        self.func = func
        if self.func == "P3":
            self.popt, self.pcov = curve_fit(polynomial_g3, self.optical_density, self.doses)
        elif self.func == "RA":
            self.popt, self.pcov = curve_fit(rational_func, self.optical_density, self.doses, p0=[0.1, 200, 500])
        else:
            raise Exception("Invalid function.")
        self.channel = channel

    def plot(self, ax: plt.Axes = None, show: bool = True, **kwargs) -> plt.Axes:
        """Plot the calibration curve.

        Parameters
        ----------
        ax : matplotlib.Axes instance
            The axis to plot the image to. If None, creates a new figure.
            Parameters
        show : bool
            Whether to actually show the image. Set to false when plotting multiple items.
        kwargs
            kwargs passed to plt.plot()
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        x = np.linspace(self.optical_density[0], self.optical_density[-1], 100)
        if self.func == "P3":
            y = polynomial_g3(x, *self.popt)
        elif self.func == "RA":
            y = rational_func(x, *self.popt)
        ax.plot(x, y, **kwargs)
        ax.plot(self.optical_density, self.doses, '*', **kwargs)
        ax.set_xlabel("Optical density")
        ax.set_ylabel("Dose")
        if show:
            plt.show()
        return ax
    