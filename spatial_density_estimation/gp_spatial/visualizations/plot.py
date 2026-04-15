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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi
from shapely.geometry import box, Polygon as ShapelyPolygon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from polyagamma import random_polyagamma
ot.RandomGenerator.SetSeed(42)


# %%
# =======================================================================================================
# ------------------------------------------ FONCTIONS DE PLOT ------------------------------------------
# =======================================================================================================

def plot_field(
    field,
    mode="plot",
    ax=None,
    title=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
    add_colorbar=True,
    savefigure=True,
    title_savefig="field_output.pdf",
):
    """

    """
    mesh = field.getMesh()
    x    = mesh.getVertices().getMarginal(0)
    y    = mesh.getVertices().getMarginal(1)
    z    = field.getValues()

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

    contour = ax_local.contourf(X, Y, Z, levels=15, vmin=vmin, vmax=vmax, cmap=cmap)

    if add_colorbar:
        fig.colorbar(contour, ax=ax_local)

    if title:
        ax_local.set_title(title)

    if savefigure:
        if mode != "plot":
            print("[plot_field] Sauvegarde ignorée : disponible uniquement en mode 'plot'.")
        else:
            try:
                try:
                    ROOT = Path(__file__).resolve().parent.parent
                except NameError:
                    ROOT = Path(".").resolve()
                FIGURES_DIR = ROOT / "visualizations" / "figures"
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                fig.savefig(FIGURES_DIR / title_savefig,
                            dpi=200, bbox_inches="tight")
                print(f"Figure sauvegardée : {FIGURES_DIR / title_savefig}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde : {e}")

    return fig, ax_local, contour


# LE TABLEAU DE BORD 
def plot_process_dashboard(sim_data, grids, cmap="viridis", 
                           title="Réalisations du processus spatial",
                           savefigure=False, title_savefig="figure_output.pdf"):
    X_array = np.asarray(sim_data["X"])
    N = X_array.shape[0] if X_array.size > 0 else 0
    X_bounds, Y_bounds, T = sim_data["bounds"]
    mus_np = np.asarray(sim_data["mus_vec"])
    GX, GY = grids["GX"], grids["GY"]
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)

    # ---------------------------------------------------------
    # EN HAUT À GAUCHE : Champ latent f*
    # ---------------------------------------------------------
    ax_f = fig.add_subplot(gs[0, 0])
    im_f = ax_f.contourf(GX, GY, grids["f_star"], levels=50, cmap="coolwarm")
    divider_f = make_axes_locatable(ax_f)
    cax_f = divider_f.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im_f, cax=cax_f, label=r"$f^\star(x,y)$")
    ax_f.set_title(r"Latent field $f^\star$" + "\n")

    # ---------------------------------------------------------
    # EN HAUT À DROITE : Intensité vraie mu*
    # ---------------------------------------------------------
    ax_mu = fig.add_subplot(gs[0, 1])
    im_mu = ax_mu.contourf(GX, GY, grids["mu_star"], levels=50, cmap=cmap)
    divider_mu = make_axes_locatable(ax_mu)
    cax_mu = divider_mu.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im_mu, cax=cax_mu, label=r"$\mu^\star(x,y)$")
    ax_mu.set_title(r"True intensity $\mu^\star = \tilde{\mu} \cdot \sigma(f^\star)$" + "\n")

    # ---------------------------------------------------------
    # EN BAS À GAUCHE : Événements seuls (Points)
    # ---------------------------------------------------------
    ax_pts = fig.add_subplot(gs[1, 0])
    if N > 0:
        ax_pts.scatter(X_array[:, 0], X_array[:, 1], s=4, c="crimson", alpha=0.7, lw=0.4, zorder=3)
    ax_pts.set_title("\n" + f"Simulated events (N={N})" + "\n")
    # divider_pts = make_axes_locatable(ax_pts)
    # cax_pts = divider_pts.append_axes("right", size="5%", pad=0.08)
    # cax_pts.set_visible(False)

    # ---------------------------------------------------------
    # EN BAS À DROITE : Données + Zones de fond
    # ---------------------------------------------------------
    ax_zon = fig.add_subplot(gs[1, 1])
    norm = plt.Normalize(vmin=mus_np.min()-0.1, vmax=mus_np.max()+0.1)
    
    for i, zone in enumerate(sim_data["zones"]):
        xs, ys = zone.exterior.xy
        ax_zon.fill(xs, ys, facecolor=cmap_obj(norm(mus_np[i])), alpha=0.35, linewidth=1.0, zorder=1)
        ax_zon.plot(xs, ys, lw=0.8, zorder=2)
        ax_zon.text(zone.centroid.x, zone.centroid.y, rf"$\tilde\mu={mus_np[i]:.1f}$", ha="center", va="center", 
                     fontsize=9, fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    if N > 0:
       ax_zon.scatter(X_array[:, 0], X_array[:, 1], s=3, c="crimson", alpha=0.4, zorder=3)
    ax_zon.set_title("\n" + "Areas" + "\n", fontsize=11)
    # divider_zon = make_axes_locatable(ax_zon)
    # cax_zon = divider_zon.append_axes("right", size="5%", pad=0.08)
    # fig.colorbar(im_mu, cax=cax_zon)

    # ----------------------------------
    # Mise en forme commune
    # ----------------------------------
    for ax in [ax_f, ax_mu, ax_pts, ax_zon]:
        ax.set_xlim(X_bounds)
        ax.set_ylim(Y_bounds)
        #ax.set_xlabel("x", fontsize=10)
        #ax.set_ylabel("y", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # ------------------
    # Sauvegarde
    # ------------------
    if savefigure:
        try:
            try:
                ROOT = Path(__file__).resolve().parent.parent
            except NameError:
                ROOT = Path(".").resolve()
            FIGURES_DIR = ROOT / "visualizations" / "figures"
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            save_path = FIGURES_DIR / Path(title_savefig).with_suffix(".pdf")
            fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
            print(f"Figure sauvegardée : {save_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")

    plt.show()


# FONCTION PLOT CELLULES DE VORONOÏ
def plot_voronoi_cells(
    cells,
    germs,
    X_bounds=(0.0, 2.0),
    Y_bounds=(0.0, 2.0),
    cmap_name="cividis",
    annotate=True,
    title="Pavage de Voronoi",
    figsize=(6, 6),
    savefigure=False,
    title_savefig="voronoi.pdf",
):
    """

    """
    n = len(germs)
    cmap = plt.get_cmap(cmap_name, n)

    fig, ax = plt.subplots(figsize=figsize)

    for i, cell in enumerate(cells):
        if cell is None or cell.is_empty:
            continue
        xs, ys = cell.exterior.xy
        ax.fill(xs, ys, alpha=0.4, color=cmap(i), zorder=1)
        ax.plot(xs, ys, color=cmap(i), linewidth=1.2, zorder=2)

    ax.scatter(
        germs[:, 0], germs[:, 1],
        c=np.arange(n), cmap=cmap_name,
        s=20, zorder=4, edgecolors="black", linewidths=0.8,
    )

    if annotate:
        for i, (x, y) in enumerate(germs):
            ax.annotate(
                f"$s_{{{i}}}$", (x, y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, zorder=5,
            )

    ax.set_xlim(X_bounds)
    ax.set_ylim(Y_bounds)
    ax.set_aspect("equal")
    ax.set_title(title + "\n", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if savefigure:
        try:
            try:
                ROOT = Path(__file__).resolve().parent.parent
            except NameError:
                ROOT = Path(".").resolve()
            FIGURES_DIR = ROOT / "visualizations" / "figures"
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            save_path = FIGURES_DIR / Path(title_savefig).with_suffix(".pdf")
            fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
            print(f"Figure sauvegardée : {save_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")

    plt.show()



# %%







# %%





# %%






