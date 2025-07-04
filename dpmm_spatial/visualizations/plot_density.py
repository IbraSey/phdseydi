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


def plot_contour_levels(X, Y, Z, levels=20, title="Lignes de niveaux", cmap='viridis'):
    """
    En chantier
    """

    plt.figure(figsize=(7, 6))
    contour = plt.contour(X, Y, Z, levels=levels, cmap=cmap)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.colorbar(contour, label="Valeur")
    plt.tight_layout()
    plt.show()


def plot_density_front(Z, x_vals=np.linspace(0, 2, 100), y_vals=np.linspace(0, 2, 100), fixed_axis='y', fixed_value=0.5, title="Densité d'une face"):
    """
    En chantier
    """

    plt.figure(figsize=(7, 6))

    if fixed_axis == 'y':
        idx = np.abs(y_vals - fixed_value).argmin()
        marginal_vals = Z[idx, :]
        plt.plot(x_vals, marginal_vals)
        plt.xlabel("x")
    else:
        idx = np.abs(x_vals - fixed_value).argmin()
        marginal_vals = Z[:, idx]
        plt.plot(y_vals, marginal_vals)
        plt.xlabel("y")

    plt.title(f"{title} (fixé {fixed_axis} = {fixed_value})")
    plt.ylabel("Densité")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




