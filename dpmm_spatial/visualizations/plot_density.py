import matplotlib.pyplot as plt
import numpy as np


def plot_sampling(samples, title='Échantillons', s=5, alpha=0.5, xlim=(0, 2), ylim=(0, 2)):
    """
    En chantier
    """

    samples = np.asarray(samples)
    plt.figure(figsize=(7, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=s, alpha=alpha)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_density_heatmap(Z, title='Heatmap', extent=(0, 2, 0, 2), cmap='grey_r'):
    """
    En chantier
    """

    plt.figure(figsize=(7, 6))
    plt.imshow(Z, extent=extent, origin='lower', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()


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


def plot_density_marginal(Z, x_vals=np.linspace(0, 2, 100), y_vals=np.linspace(0, 2, 100), fixed_axis='y', fixed_value=0.5, title='Margiale'):
    """
    En chantier
    """

    plt.figure(figsize=(7, 6))

    if fixed_axis == 'y':
        idx = np.abs(y_vals - fixed_value).argmin()
        marginal_vals = Z[idx, :]
        plt.plot(x_vals, marginal_vals)
        plt.xlabel("x")
    elif fixed_axis == 'x':
        idx = np.abs(x_vals - fixed_value).argmin()
        marginal_vals = Z[:, idx]
        plt.plot(y_vals, marginal_vals)
        plt.xlabel("y")
    else:
        raise ValueError("On doit fixer 'x' ou 'y'")

    plt.title(f"{title} (fixé {fixed_axis} = {fixed_value})")
    plt.ylabel("Densité marginale")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




