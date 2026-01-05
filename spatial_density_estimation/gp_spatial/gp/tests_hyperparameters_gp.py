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
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from matplotlib.colors import Normalize
from visualizations.plot import plot_field
ot.RandomGenerator.SetSeed(42)


#%%
# =========================================================================================================
# ----------------------------------------- Tests hyperparameters -----------------------------------------
# =========================================================================================================

# ======================== Maillage & Transformation par sigmoïde ========================
mesher = ot.IntervalMesher([50, 50])
mesh = mesher.build(ot.Interval([0.0, 0.0], [2.0, 2.0]))
sigmoid = ot.SymbolicFunction(['z'], ['1/(1+exp(-z))'])
field_function = ot.PythonFieldFunction(mesh, 1, mesh, 1, sigmoid)

# Grille d'hyperparamètres du noyau SE : ([lx, ly], amplitude)
param_grid = [
    ([0.3, 0.3], 0.8),    # référence pour comparaison
    ([0.1, 0.1], 0.8),    # plus rugeux
    ([0.8, 0.8], 0.8),    # plus lisse
    ([0.3, 0.3], 0.3),    # amplitude plus faible
    ([0.3, 0.3], 2.0),    # amplitude plus forte
    ([0.1, 0.7], 0.8),    # anisotrope
]

fields = []
for scales, amp in param_grid:
    cov = ot.SquaredExponential(scales, [amp])
    gp = ot.GaussianProcess(cov, mesh)
    process = ot.CompositeProcess(field_function, gp)
    fields.append((process.getRealization(), scales, amp))

# Bornes de couleurs communes pour comparaison cohérente
all_vals = np.concatenate([np.array(f[0].getValues()).ravel() for f in fields])
zmin, zmax = float(np.min(all_vals)), float(np.max(all_vals))

# Grille de subplots
n = len(fields)
ncols = min(3, n)
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5.2*ncols, 4.3*nrows), squeeze=False)
axes_flat = axes.ravel()

for i, (field_f, scales, amp) in enumerate(fields):
    fig, ax_i, _ = plot_field(
        field_f,
        mode="subplot",
        ax=axes_flat[i],
        title=f"SE[({scales[0]}, {scales[1]}), {amp}]",
        vmin=zmin, vmax=zmax,
        add_colorbar=True
    )

fig.suptitle(r"Réalisations de $\sigma(\mathrm{GP})$", y=0.995)
plt.tight_layout()
plt.show()
#ROOT = Path(__file__).resolve().parent.parent
#FIGURES_DIR = ROOT / "visualizations" / "figures"
#FIGURES_DIR.mkdir(parents=True, exist_ok=True)
#fig.savefig(FIGURES_DIR / "figure_test_hyperparameters_gp.pdf", dpi=300, bbox_inches="tight")


# %%
# ========================================================================================================
# ----------------------------------------- Tests add mean in GP -----------------------------------------
# ========================================================================================================
def tend(X):
    return [3*(X[0]**2 + X[1]**2 <= 2.0)]

f = ot.PythonFunction(2, 1, tend)
fTrend = ot.TrendTransform(f, mesh)
cov = ot.SquaredExponential([0.35, 0.12], [1.0])
X = ot.GaussianProcess(fTrend ,cov, mesh)

sigmoid = ot.SymbolicFunction(['z'], ['1/(1+exp(-z))'])
field_function = ot.PythonFieldFunction(mesh, 1, mesh, 1, sigmoid)
process = ot.CompositeProcess(field_function, X)

# Plot a realization of the composite process
field_f = process.getRealization()
plot_field(field_f, title='Realization GP with trend')
#plot_field(field_f, title='Realization GP with trend', 
#           savefigure=True, title_savefig='figure_gp_moon.pdf')


# %%
# =======================================================================================================
# -------------------------- Construction zonage cas jouet & Ajout comme trend --------------------------
# =======================================================================================================
# Polygones dans [0,2]²
polygons = [
    Polygon([(0.5,0.2), (0.55,0.2), (0.51,0.22), (0.39,0.48), (0.25,0.57), (0.13,0.68), (0.15,0.4), (0.35,0.28)]),
    Polygon([(0.4,0.1), (1.15,0.1), (1.25,0.3), (1.7,0.4), (1.7,0.6), (1.2,0.5), (1.1,0.47), (0.9,0.4), (0.55,0.2)]),
    Polygon([(0.55,0.2), (0.9,0.4), (0.85,0.58), (0.62,0.6), (0.5,0.6), (0.39,0.48), (0.51,0.22), (0.55,0.2)]),
    Polygon([(0.13,0.68), (0.25,0.57), (0.39,0.48), (0.5,0.6), (0.62,0.6), (0.58,1.0), (0.4,1.25), (0.37,1.4), (0.55,1.6), 
             (0.75,1.59), (1.0,1.05), (1.35,1.2), (1.55,1.5), (1.35,1.6), (1.1, 1.7), (0.6,1.7), (0.23,1.4), (0.05,1.0)]),
    Polygon([(0.62,0.6), (0.85,0.58), (0.9,0.4), (1.1,0.47), (1.2,0.5), (1.5,1.25), (1.35,1.2), (1.0,1.05), (0.58,1.0)]),
    Polygon([(0.58,1.0), (1.0,1.05), (0.75,1.59), (0.55,1.6), (0.37,1.4), (0.4,1.25)]),
    Polygon([(0.23,1.4), (0.35,1.9), (1.45,1.85), (1.35,1.6), (1.1, 1.7), (0.6,1.7), (0.23,1.4)]),
    Polygon([(1.45,1.85), (1.35,1.6), (1.55,1.5), (1.35,1.2), (1.5,1.25), (1.75,1.4), (1.75,1.6), (1.7,1.77), (1.65,1.80)]),
    Polygon([(1.5,1.25), (1.2,0.5), (1.7,0.6), (1.7,0.4), (1.78,0.5), (1.82,0.75), (1.83,1.0), (1.82,1.15), (1.75,1.4)])
]

# Intensités (poids de zone)
#zone_weights = np.array([0.05, 0.05, 0.25, 0.01, 0.1, 0.14, 0.14, 0.01, 0.25]) 
zone_weights = np.random.normal(loc=0.0, scale=1.0, size=9)
cmap = plt.cm.viridis
norm = Normalize(vmin=zone_weights.min(), vmax=zone_weights.max())
Zones = [(prep(p), float(w)) for p, w in zip(polygons, zone_weights)]

# Calcul des aires
areas = np.array([poly.area for poly in polygons])
for i, area in enumerate(areas, 1):
    print(f"Aiire du polygone P{i} : {area:.4f}")

# =========================================== Affichage ===========================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_aspect('equal')

for i, (poly, intensity) in enumerate(zip(polygons, zone_weights), start=1):
    color = cmap(norm(intensity))
    patch = plt.Polygon(list(poly.exterior.coords), facecolor=color, edgecolor='black')
    ax.add_patch(patch)

    # Position du label
    if i == 4:
        label_x, label_y = 0.32, 0.9  # Coordonnées choisies manuellement pour P4
    else:
        centroid = poly.centroid
        label_x, label_y = centroid.x, centroid.y

    ax.text(label_x, label_y, f'P{i}', color='black', weight='bold',
            ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))

# Colorbar et finalisation
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Intensity")
plt.title("Zonage sismotectonique - Cas jouet")
plt.grid(True)
plt.show()
#ROOT = Path(__file__).resolve().parent.parent
#FIGURES_DIR = ROOT / "visualizations" / "figures"
#FIGURES_DIR.mkdir(parents=True, exist_ok=True)
#fig.savefig(FIGURES_DIR / "figure_zonage_jouet.pdf", dpi=300, bbox_inches="tight")

def tend(X):
    x, y = float(X[0]), float(X[1])
    pt = Point(x, y)
    for P, w in Zones:
        if P.covers(pt):
            #return [-np.log( (max(abs(zone_weights)) /abs(w))-1 )]  
            return [w]           
    return [-np.inf]

f = ot.PythonFunction(2, 1, tend)
fTrend = ot.TrendTransform(f, mesh)
cov = ot.SquaredExponential([0.15, 0.15], [1.0])
X = ot.GaussianProcess(fTrend, cov, mesh)

sigmoid = ot.SymbolicFunction(['z'], ['1/(1+exp(-z))'])
field_function = ot.PythonFieldFunction(mesh, 1, mesh, 1, sigmoid)
process = ot.CompositeProcess(field_function, X)

field_f = process.getRealization()
def tend(X):
    x, y = float(X[0]), float(X[1])
    pt = Point(x, y)
    for P, w in Zones:
        if P.covers(pt):
            #return [-np.log( (max(abs(zone_weights)) /abs(w))-1 )]  
            return [w]           
    return [-np.inf]

f = ot.PythonFunction(2, 1, tend)
fTrend = ot.TrendTransform(f, mesh)
cov = ot.SquaredExponential([0.15, 0.15], [1.0])
X = ot.GaussianProcess(fTrend, cov, mesh)

sigmoid = ot.SymbolicFunction(['z'], ['1/(1+exp(-z))'])
field_function = ot.PythonFieldFunction(mesh, 1, mesh, 1, sigmoid)
process = ot.CompositeProcess(field_function, X)

field_f = process.getRealization()
plot_field(field_f, title='Realization GP with zonage trend')
#plot_field(field_f, title='Realization GP with zonage trend', 
#           savefigure=True, title_savefig='figure_gp_with_zonage_jouet.pdf')


# %%




# %%
