import matplotlib.pyplot as plt
import numpy as np

def plot_density_heatmap(Z, title, extent=(0, 2, 0, 2), cmap='jet'):
    plt.imshow(Z, extent=extent, origin='lower', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()
