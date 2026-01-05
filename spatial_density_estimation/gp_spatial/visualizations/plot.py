#%%
# =================================================================================================
# -------------------------------------------- IMPORTS --------------------------------------------
# =================================================================================================

from pathlib import Path
import os, sys
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
import openturns as ot
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Rectangle
ot.RandomGenerator.SetSeed(42)


# %%
# =================================================================================================
# -------------------------------------- FONCTION PLOT FIELD --------------------------------------
# =================================================================================================

def plot_field(field, mode="plot", ax=None, title=None, vmin=None, vmax=None, add_colorbar=True, 
               savefigure=False, title_savefig='figure_output.pdf'):
    """

    """
    mesh = field.getMesh()

    x = mesh.getVertices().getMarginal(0)
    y = mesh.getVertices().getMarginal(1)
    z = field.getValues()

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx = len(x_unique)
    ny = len(y_unique)

    X = np.array(x).reshape(ny, nx)
    Y = np.array(y).reshape(ny, nx)
    Z = np.array(z).reshape(ny, nx)

    if mode == "plot":
        fig, ax_local = plt.subplots(figsize=(6, 4))
    elif mode == "subplot":
        if ax is None:
            raise ValueError("En mode 'subplot', fournir un axe via le paramètre ax.")
        fig, ax_local = ax.figure, ax
    else:
        raise ValueError("mode doit être 'plot' ou 'subplot'.")

    contour = ax_local.contourf(X, Y, Z, levels=15, vmin=vmin, vmax=vmax)
    if add_colorbar:
        fig.colorbar(contour, ax=ax_local)
    if title:
        ax_local.set_title(title)

    if savefigure:
        ROOT = Path(__file__).resolve().parent.parent
        FIGURES_DIR = ROOT / "visualizations" / "figures"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / title_savefig, dpi=300, bbox_inches="tight")

    return fig, ax_local, contour


# %%
# =======================================================================================================
# ---------------------------------------- FONCTION PLOT DONNÉES ----------------------------------------
# =======================================================================================================

def plot_poisson_zones_data(X, zones, mus_vec, X_bounds, Y_bounds, 
                            title_prefix="Processus de Poisson homogène par zones", cmap=plt.cm.viridis,
                            show_time_hist=True, savefigure=False, title_savefig='figure_output.pdf'):
    """

    """
    X_array = np.asarray(X, dtype=float)
    mus_vec = np.asarray(mus_vec, dtype=float)
    N = X_array.shape[0]
    J = len(zones)
    mu_max = max(mus_vec)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    ax_points, ax_zones = axes

    # --------------------
    # Points (sans zones)
    # --------------------
    if N > 0:
        ax_points.scatter(
            X_array[:, 0],
            X_array[:, 1],
            s=20,
            c="red",
            alpha=0.6,
            edgecolors="darkred",
            linewidth=0.5,
            label="Événements",
        )
    ax_points.set_title("Événements (sans zones)", fontsize=12)

    # --------------------
    # Points (avec zones)
    # --------------------
    for zone, mu in zip(zones, mus_vec):
        xmin, ymin, xmax, ymax = zone.bounds

        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor=cmap(float(mu) / mu_max),
            edgecolor="black",
            linewidth=2,
            alpha=0.35,
        )
        ax_zones.add_patch(rect)

        # Label en haut-gauche de la zone
        tx = xmin + 0.05 * (xmax - xmin)
        ty = ymax - 0.05 * (ymax - ymin)
        ax_zones.text(
            tx,
            ty,
            rf"$\mu={float(mu):.2f}$",
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    if N > 0:
        ax_zones.scatter(
            X_array[:, 0],
            X_array[:, 1],
            s=20,
            c="red",
            alpha=0.6,
            edgecolors="darkred",
            linewidth=0.5,
            label="Événements",
        )
    ax_zones.set_title("Événements (avec zones)", fontsize=12)

    # ----------------
    # Mise en forme
    # ----------------
    for ax in (ax_points, ax_zones):
        ax.set_xlim(X_bounds)
        ax.set_ylim(Y_bounds)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle(
        f"{title_prefix} : {N} événements sur {J} zones\n",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

    if savefigure:
        ROOT = Path(__file__).resolve().parent.parent
        FIGURES_DIR = ROOT / "visualizations" / "figures"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / title_savefig, dpi=300, bbox_inches="tight")

    # ============================
    # Histogramme temporel
    # ============================
    if show_time_hist:
        fig2, ax2 = plt.subplots(figsize=(12, 4))

        if N > 0:
            ax2.hist(
                X_array[:, 2],
                bins=30,
                alpha=0.7,
                edgecolor="black",
            )

        ax2.set_xlabel("Temps", fontsize=12)
        ax2.set_ylabel("Nb d'événements", fontsize=12)
        ax2.set_title("Distribution temporelle des events\n", fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        if savefigure :
            ROOT = Path(__file__).resolve().parent.parent
            FIGURES_DIR = ROOT / "visualizations" / "figures"
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig2.savefig(FIGURES_DIR / ("time_" + title_savefig), dpi=300, bbox_inches="tight")


# %%








# %%








# %%






