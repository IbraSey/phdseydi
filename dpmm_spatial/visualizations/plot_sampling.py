import matplotlib.pyplot as plt
import numpy as np
import openturns as ot


def plot_sampling(samples, title='Échantillons', s=5, alpha=0.5, xlim=(0, 2), ylim=(0, 2), ax=None):
    """
    Affiche un nuage de points représentant un échantillon 2D.

    Paramètres :
        - samples : ndarray, tableau (N, 2) des points (x, y)
        - title : str, titre de l'axe
        - s : float, taille des points
        - alpha : float, transparence
        - xlim, ylim : tuple, bornes des axes
        - ax : matplotlib.axes.Axes, subplot cible (optionnel)
    """
    samples = np.asarray(samples)
    if ax is None:
        ax = plt.gca()

    ax.scatter(samples[:, 0], samples[:, 1], s=s, alpha=alpha)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")



