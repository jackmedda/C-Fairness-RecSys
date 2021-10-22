import numpy as np
import matplotlib.pyplot as plt


def gini(x, w=None):
    x = np.asarray(x)

    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (cumxw[-1] * cumw[-1])
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def show(to_plot):
    def wrapper(*args, **kwargs):
        plt.figure()
        axes = to_plot(*args, **kwargs)
        plt.show()
        return axes
    return wrapper
