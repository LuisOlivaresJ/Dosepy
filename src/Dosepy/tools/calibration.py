import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


"""Fit functions."""
def polymonial_g3(x,a,b,c,d):
    """
    Polynomial function of degree 3.
    """
    return a + b*x + c*x**2 + d*x**3

class Calibration:
    """Class used to represent a calibration curve.
        
        Attributes
        ----------
        doses : list
            The doses values that were used to expose films for calibration.
        intensity : list
            Intensity values used for calibration.
        func : str
            The model function, f(x, …) used for intesity-dose relationship. 
            "P3": Polynomial function of degree 3. 
        popt : array
            Parameters used by the function.
        pcov : 2-D array
            The estimated approximate covariance of popt. The diagonals provide the variance 
            of the parameter estimate. To compute one standard deviation errors on the parameters, 
            use perr = np.sqrt(np.diag(pcov)).
        """
    
    def __init__(self, doses: list, intensity: list, func: str):
        
        self.doses = sorted(doses)
        self.intensity = sorted(intensity)
        self.func = func
        if self.func == "P3":
            self.popt, self.pcov = curve_fit(polymonial_g3, self.intensity, self.doses)
        else:
            raise Exception("Invalid function.")

    def plot(self, ax: plt.Axes = None, show: bool = True, **kwargs) -> plt.Axes:
        """Plot the calibration curve.

        Parameters
        ----------
        ax : matplotlib.Axes instance
            The axis to plot the image to. If None, creates a new figure.
            Parameters
        show : bool
            Whether to actually show the image. Set to false when plotting multiple items.
        clear_fig : bool
            Whether to clear the prior items on the figure before plotting.
        kwargs
            kwargs passed to plt.plot()
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        x = np.linspace(self.intensity[0], self.intensity[-1], 100)
        y = polymonial_g3(x, *self.popt)
        ax.plot(x, y, **kwargs)
        if show:
            plt.show()
        return ax
    