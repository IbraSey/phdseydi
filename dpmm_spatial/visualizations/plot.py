import matplotlib.pyplot as plt
import numpy as np


def plot_density_heatmap(Z, title='Heatmap', extent=(0, 2, 0, 2), cmap='grey_r', ax=None):
    """
    Affiche une heatmap 2D de la densité.

    Paramètres :
        - Z : ndarray 2D, les valeurs de densité
        - title : str, titre de la figure
        - extent : tuple, domaine des axes
        - cmap : str, colormap utilisée
        - ax : matplotlib.axes.Axes, axe cible (optionnel). Si None, utilise plt.gca()
    """
    
    if ax is None:
        ax = plt.gca()

    im = ax.imshow(Z, extent=extent, origin='lower', cmap=cmap)
    plt.colorbar(im, ax=ax, label="Densité")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal')


def plot_contour_levels(X, Y, Z, levels=20, title="Lignes de niveaux", cmap='viridis', ax=None):
    """
    EN CHANTIER 
    """
    
    if ax is None:
        ax = plt.gca()
    contour = ax.contour(X, Y, Z, levels=levels, cmap=cmap)
    plt.colorbar(contour, ax=ax, label="Densité")
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal')


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



    



