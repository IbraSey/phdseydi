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
# =======================================================================================================
# ----------------------------------------- FONCTION PLOT FIELD -----------------------------------------
# =======================================================================================================

def plot_field(field, mode="plot", ax=None, title=None, vmin=None, vmax=None, add_colorbar=True):
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
            raise ValueError("En mode 'subplot', fournir un axe via le paramètre ax")
        fig, ax_local = ax.figure, ax
    else:
        raise ValueError("mode doit être 'plot' ou 'subplot' !")

    contour = ax_local.contourf(X, Y, Z, levels=15, vmin=vmin, vmax=vmax)
    if add_colorbar:
        fig.colorbar(contour, ax=ax_local)
    if title:
        ax_local.set_title(title)

    return fig, ax_local, contour





# %%
# ==========================================================================================================
# ------------------------------------------- FONCTION PLOT DATA -------------------------------------------
# ==========================================================================================================

def plot_poisson_zones_data(X, zones, X_bounds, Y_bounds, 
                            mus_vec=None, 
                            title="Réalisations du processus spatial", 
                            show_time_hist=False, 
                            savefigure=False, 
                            title_savefig='figure_output.pdf',
                            cmap=plt.cm.viridis):
    """

    """
    # -----------------------------------------------------------
    # 1. Préparation des données
    # -----------------------------------------------------------
    X_array = np.asarray(X, dtype=float)
    N = X_array.shape[0] if X_array.size > 0 else 0
    J = len(zones)
    
    # Gestion des intensités pour la coloration (si donné)
    has_mus = mus_vec is not None
    if has_mus:
        mus_np = np.asarray(mus_vec, dtype=float)
        mu_max = np.max(mus_np) if len(mus_np) > 0 else 1.0
    
    # -----------------------------------------------------------
    # 2. Création figure 
    # -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    ax_points, ax_zones = axes

    # --- Subplot 1 : Points seuls ---
    if N > 0:
        ax_points.scatter(
            X_array[:, 0], X_array[:, 1], s=5, c="red", marker='o', alpha=0.6, 
            edgecolors="darkred", linewidth=0.5, label="Événements")
    ax_points.set_title("Événements (sans zones)", fontsize=11)

    # --- Subplot 2 : Points + Zones ---
    for i, zone in enumerate(zones):
        xmin, ymin, xmax, ymax = zone.bounds
        
        # Style du rectangle selon la présence de mus_vec
        if has_mus:
            mu_val = mus_np[i]
            color = cmap(float(mu_val) / mu_max)
            facecolor = color
            alpha_rect = 0.35
            
            # Ajout du texte d'intensité
            tx = xmin + 0.05 * (xmax - xmin)
            ty = ymax - 0.05 * (ymax - ymin)
            ax_zones.text(
                tx, ty, rf"$\lambda={mu_val:.2f}$",
                ha="left", va="top", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )
        else:
            facecolor = "none" # Transparent si pas d'intensité
            alpha_rect = 0.1

        rect = Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, 
            facecolor=facecolor,
            edgecolor="black", 
            linewidth=1.0 if has_mus else 0.8, 
            linestyle="-" if has_mus else "--", 
            alpha=alpha_rect if has_mus else 0.5
        )
        ax_zones.add_patch(rect)

    # Ré-affichage des points par dessus les zones
    if N > 0:
        ax_zones.scatter(
            X_array[:, 0], X_array[:, 1],
            s=5, c="red", marker='o', alpha=0.6, 
            edgecolors="darkred", linewidth=0.5, label="Événements"
        )
    ax_zones.set_title("Événements (avec zones)", fontsize=11)

    # --- Mise en forme globale ---
    for ax in axes:
        ax.set_xlim(X_bounds)
        ax.set_ylim(Y_bounds)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.6)
        if N > 0: ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f"{title} (N={N} événements, J={J} zone(s))", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    # -----------------------------------------------------------
    # 3. Histogramme Temporel
    # -----------------------------------------------------------
    if show_time_hist and N > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.hist(X_array[:, 2], bins=30, color='gray', edgecolor='black', alpha=0.7)
        ax2.set_xlabel("Temps (t)")
        ax2.set_ylabel("Nombre d'événements")
        ax2.set_title("Distribution temporelle")
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------
    # 4. Sauvegarde figure 
    # -----------------------------------------------------------
    if savefigure :
            ROOT = Path(__file__).resolve().parent.parent
            FIGURES_DIR = ROOT / "visualizations" / "figures"
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig2.savefig(FIGURES_DIR / ("time_" + title_savefig), dpi=200, bbox_inches="tight")





# %%









# %%








# %%






