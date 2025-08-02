import matplotlib.pyplot as plt
import numpy as np


def plot_density_heatmap(Z, title='Heatmap', extent=(0, 2, 0, 2), cmap='grey_r', ax=None):
    """
    Display a heatmap of a 2D density matrix.

    Parameters
    ----------
    Z : ndarray of shape (n_y, n_x)
        2D array representing the evaluated density over a grid.

    title : str, default='Heatmap'
        Title of the plot.

    extent : tuple of float, default=(0, 2, 0, 2)
        Boundaries of the heatmap image (x_min, x_max, y_min, y_max).

    cmap : str, default='grey_r'
        Colormap used to display the heatmap.

    ax : matplotlib.axes.Axes or None, default=None
        Axis to plot on. If None, uses the current axis.

    Returns
    -------
    None
    """
    
    if ax is None:
        ax = plt.gca()

    im = ax.imshow(Z, extent=extent, origin='lower', cmap=cmap)
    plt.colorbar(im, ax=ax, label="Densité")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal')


def plot_contour_levels(X, Y, Z, levels=20, title="Lignes de niveaux", cmap='viridis', ax=None, inline=True):
    """
    Plot contour lines for a 2D density function.

    Parameters
    ----------
    X : ndarray of shape (n_y, n_x)
        Meshgrid for the x-coordinates.

    Y : ndarray of shape (n_y, n_x)
        Meshgrid for the y-coordinates.

    Z : ndarray of shape (n_y, n_x)
        Scalar field values evaluated at each (X, Y) point.

    levels : int or list of float, default=20
        Number or explicit values of contour levels.

    title : str, default='Lignes de niveaux'
        Title of the plot.

    cmap : str, default='viridis'
        Colormap used for contour lines.

    ax : matplotlib.axes.Axes or None, default=None
        Axis to plot on. If None, uses the current axis.

    Returns
    -------
    None
    """
    
    if ax is None:
        ax = plt.gca()
    contour = ax.contour(X, Y, Z, levels=levels, cmap=cmap)
    plt.colorbar(contour, ax=ax, label="Densité")
    ax.clabel(contour, inline=inline, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal')


def plot_sampling(samples, title='Échantillons', s=5, alpha=0.5, xlim=(0, 2), ylim=(0, 2), ax=None):
    """
    Scatter plot of 2D samples, e.g. from a density or generative model.

    Parameters
    ----------
    samples : array-like of shape (n_samples, 2)
        2D coordinates of the sampled points.

    title : str, default='Échantillons'
        Title of the plot.

    s : float, default=5
        Marker size.

    alpha : float, default=0.5
        Opacity level for points (0 = transparent, 1 = opaque).

    xlim : tuple of float, default=(0, 2)
        Limits of the x-axis.

    ylim : tuple of float, default=(0, 2)
        Limits of the y-axis.

    ax : matplotlib.axes.Axes or None, default=None
        Axis to plot on. If None, uses the current axis.

    Returns
    -------
    None
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



    



