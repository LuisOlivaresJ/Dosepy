from scipy import ndimage
import numpy as np

def filter_array(
    array: np.ndarray, size: float | int = 0.05, kind: str = "median"
) -> np.ndarray:
    """Filter the array.

    Parameters
    ----------
    array: np.ndarray
        The array to filter.
    size : float, int
        Size of the median filter to apply.
        If a float, the size is the ratio of the length. Must be in the range 0-1.
        E.g. if size=0.1 for a 1000-element array, the filter will be 100 elements.
        If an int, the filter is the size passed.
    kind : {'median', 'gaussian'}
        The kind of filter to apply. If gaussian, `size` is the sigma value.

    Notes
    -----
    This function was obtained from the `pylinac` library.
    """
    if isinstance(size, float):
        if 0 < size < 1:
            size = int(round(len(array) * size))
            size = max(size, 1)
        else:
            raise ValueError("Float was passed but was not between 0 and 1")

    if kind == "median":
        filtered_array = ndimage.median_filter(array, size=size)
    elif kind == "gaussian":
        filtered_array = ndimage.gaussian_filter(array, sigma=size)
    else:
        raise ValueError(
            f"Filter type {kind} unsupported. Use one of 'median', 'gaussian'"
        )
    return filtered_array