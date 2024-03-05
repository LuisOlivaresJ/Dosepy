"""
NAME
    Calibration module

DESCRIPTION
    Module for the management of the calibration curve. Here are the functions
    to be used for fitting. See Calibration class for details.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


"""Functions used for film calibration."""


def polynomial_g3(x, a, b, c, d):
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
        y : list
            The doses values that were used to expose films for calibration.
        x : list
            Optical density if "P3" fit function is used, or normalized pixel value
            for "RF" fit function.
        func : str
            The model function used for dose-film response relationship.
            "P3": Polynomial function of degree 3.
            "RF": Rational function.
        channel : str
            Color channel. "R": Red, "G": Green and "B": Blue.
        popt : array
            Parameters of the function.
        pcov : 2-D array
            The estimated approximate covariance of popt. The diagonals provide
            the variance of the parameter estimate. To compute one standard
            deviation errors on the parameters, use perr = np.sqrt(np.diag(pcov)).
        """

    def __init__(self, y: list, x: list, func: str = "P3", channel: str = "R"):

        self.doses = sorted(y)

        if func in ["P3", "Polynomial"]:
            self.x = sorted(x)  # Film response.
        elif func in ["RF", "Rational"]:
            self.x = sorted(x, reverse=True)

        self.func = func

        if self.func in ["P3", "Polynomial"]:
            self.popt, self.pcov = curve_fit(polynomial_g3, self.x, self.doses)
        elif self.func in ["RF", "Rational"]:
            self.popt, self.pcov = curve_fit(
                                            rational_func,
                                            self.x,
                                            self.doses,
                                            p0=[0.1, 200, 500]
                                            )
        else:
            raise Exception("Invalid fit function.")
        self.channel = channel

    def plot(self, ax: plt.Axes = None, show: bool = True, **kwargs) -> plt.Axes:
        """Plot the calibration curve.

        Parameters
        ----------
        ax : matplotlib.Axes instance
            The axis to plot the image to. If None, creates a new figure.
        show : bool
            Whether to actually show the image. Set to false when plotting
            multiple items.
        kwargs
            kwargs passed to plt.plot()
        """
        if ax is None:
            fig, ax = plt.subplots()

        x = np.linspace(self.x[0], self.x[-1], 100)
        if self.func in ["P3", "Polynomial"]:
            y = polynomial_g3(x, *self.popt)
            ax.set_xlabel("Optical density")
        elif self.func in ["RF", "Rational"]:
            y = rational_func(x, *self.popt)
            ax.set_xlabel("Normalized pixel value")

        if self.channel in ["R", "Red", "r", "red"]:
            color = "red"
        elif self.channel in ["G", "Green", "g", "green"]:
            color = "green"
        elif self.channel in ["B", "Blue", "b", "blue"]:
            color = "blue"
        elif self.channel in ["M", "Mean", "m", "mean"]:
            color = "black"
        
        ax.plot(x, y, color = color, **kwargs)
        ax.plot(
            self.x,
            self.doses,
            color = color,
            marker = '*',
            linestyle = 'None',
            **kwargs
            )
        ax.set_ylabel("Dose [Gy]")
        if show:
            plt.show()
        return ax
