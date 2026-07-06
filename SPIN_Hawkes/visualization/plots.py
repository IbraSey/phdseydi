"""Plotting and figure-saving helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..simulation import SpatialProcessSimulation

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
VECTOR_FIGURE_FORMAT = "pdf"
RASTER_FIGURE_FORMAT = "png"
RASTER_FIGURE_DPI = 600
FIGURE_DPI = RASTER_FIGURE_DPI


def save_figure(
    fig,
    filename: str | Path,
    output_dir: str | Path | None = None,
    figure_type: str = "vector",
    dpi: int | None = None,
) -> Path:
    """Save a figure below the package ``figures`` directory.

    ``figure_type='vector'`` is for curves, graphs, diagrams and histograms:
    figures are saved as PDF and no explicit DPI is passed. ``figure_type='raster'``
    is for heatmaps, intensity maps, simulation images and other matrix-like
    displays: figures are saved as PNG at 600 dpi by default.
    """
    figure_type = str(figure_type).lower()
    if figure_type not in {"vector", "raster"}:
        raise ValueError("figure_type must be 'vector' or 'raster'.")

    path = Path(filename)
    suffix = ".png" if figure_type == "raster" else ".pdf"
    if path.suffix.lower() != suffix:
        path = path.with_suffix(suffix)

    directory = DEFAULT_FIGURES_DIR if output_dir is None else Path(output_dir)
    destination = directory / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"}
    if figure_type == "raster":
        save_kwargs["dpi"] = RASTER_FIGURE_DPI if dpi is None else int(dpi)
    fig.savefig(destination, **save_kwargs)
    return destination


def plot_field(
    field,
    mode: str = "plot",
    ax=None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap="viridis",
    add_colorbar: bool = True,
    savefigure: bool = False,
    title_savefig: str = "field_output",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    """Plot a scalar OpenTURNS field on a regular two-dimensional mesh."""
    mesh = field.getMesh()
    vertices = np.asarray(mesh.getVertices(), dtype=float)
    values = np.asarray(field.getValues(), dtype=float).reshape(-1)
    x_unique = np.unique(vertices[:, 0])
    y_unique = np.unique(vertices[:, 1])
    expected_size = x_unique.size * y_unique.size
    if vertices.shape[0] != expected_size or values.size != expected_size:
        raise ValueError("plot_field requires a complete regular two-dimensional mesh.")

    x_grid = vertices[:, 0].reshape(y_unique.size, x_unique.size)
    y_grid = vertices[:, 1].reshape(y_unique.size, x_unique.size)
    value_grid = values.reshape(y_unique.size, x_unique.size)

    if mode == "plot":
        fig, local_ax = plt.subplots(figsize=(6, 4))
    elif mode == "subplot":
        if ax is None:
            raise ValueError("ax is required when mode='subplot'.")
        fig, local_ax = ax.figure, ax
    else:
        raise ValueError("mode must be either 'plot' or 'subplot'.")

    contour = local_ax.contourf(
        x_grid, y_grid, value_grid, levels=15, vmin=vmin, vmax=vmax, cmap=cmap
    )
    if add_colorbar:
        fig.colorbar(contour, ax=local_ax)
    if title:
        local_ax.set_title(title)
    if savefigure:
        if mode != "plot":
            raise ValueError("A subplot must be saved through its parent figure.")
        save_figure(fig, title_savefig, output_dir, figure_type="raster")
    if show and mode == "plot":
        plt.show()
    return fig, local_ax, contour


def plot_process_dashboard(
    simulation,
    cmap="viridis",
    latent_cmap="coolwarm",
    title: str = "Spatial process simulation",
    savefigure: bool = False,
    title_savefig: str = "process_dashboard",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    """Plot latent field, intensity, events and piecewise baseline domains."""
    if not isinstance(simulation, SpatialProcessSimulation):
        raise TypeError("simulation must be a SpatialProcessSimulation instance.")
    events = np.asarray(simulation.sample, dtype=float)
    n_events = events.shape[0]
    x_bounds, y_bounds = simulation.x_bounds, simulation.y_bounds
    baseline = simulation.baseline_intensities
    domains = simulation.domains.polygons
    grid = simulation.grid
    grid_x, grid_y = grid.x, grid.y
    color_map = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    latent_ax = axes[0, 0]
    latent_image = latent_ax.contourf(
        grid_x, grid_y, grid.latent, levels=50, cmap=latent_cmap
    )
    latent_divider = make_axes_locatable(latent_ax)
    fig.colorbar(
        latent_image,
        cax=latent_divider.append_axes("right", size="5%", pad=0.08),
        label=r"$f^\star(x,y)$",
    )
    latent_ax.set_title(r"Latent field $f^\star$")

    intensity_ax = axes[0, 1]
    intensity_image = intensity_ax.contourf(
        grid_x, grid_y, grid.intensity, levels=50, cmap=cmap
    )
    intensity_divider = make_axes_locatable(intensity_ax)
    fig.colorbar(
        intensity_image,
        cax=intensity_divider.append_axes("right", size="5%", pad=0.08),
        label=r"$\mu^\star(x,y)$",
    )
    intensity_ax.set_title(r"True intensity $\mu^\star=	ilde\mu\,\sigma(f^\star)$")

    event_ax = axes[1, 0]
    if n_events:
        event_ax.scatter(events[:, 0], events[:, 1], s=5, c="crimson", alpha=0.7)
    event_ax.set_title(f"Simulated events (N={n_events})")

    domain_ax = axes[1, 1]
    normalizer = plt.Normalize(vmin=baseline.min(), vmax=baseline.max())
    if np.isclose(baseline.min(), baseline.max()):
        normalizer = plt.Normalize(vmin=baseline.min() - 0.5, vmax=baseline.max() + 0.5)
    for index, domain in enumerate(domains):
        xs, ys = domain.exterior.xy
        domain_ax.fill(
            xs, ys, facecolor=color_map(normalizer(baseline[index])), alpha=0.35
        )
        domain_ax.plot(xs, ys, linewidth=0.8)
        domain_ax.text(
            domain.centroid.x,
            domain.centroid.y,
            rf"$	ilde\mu={baseline[index]:.2g}$",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )
    if n_events:
        domain_ax.scatter(events[:, 0], events[:, 1], s=4, c="crimson", alpha=0.4)
    domain_ax.set_title("Spatial domains")

    for axis in axes.flat:
        axis.set_xlim(x_bounds)
        axis.set_ylim(y_bounds)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    if savefigure:
        save_figure(fig, title_savefig, output_dir, figure_type="raster")
    if show:
        plt.show()
    return fig, axes


def plot_voronoi_cells(
    cells,
    germs,
    X_bounds=(0.0, 2.0),
    Y_bounds=(0.0, 2.0),
    cmap_name="cividis",
    annotate: bool = True,
    title: str = "Voronoi tessellation",
    figsize=(6, 6),
    savefigure: bool = False,
    title_savefig: str = "voronoi",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    """Plot bounded Voronoi cells and their germs."""
    germs = np.asarray(germs, dtype=float)
    if germs.ndim != 2 or germs.shape[1] != 2 or len(cells) != len(germs):
        raise ValueError("cells and germs must describe the same 2D tessellation.")
    color_map = plt.get_cmap(cmap_name, len(germs))
    fig, ax = plt.subplots(figsize=figsize)
    for index, cell in enumerate(cells):
        if cell is None or cell.is_empty:
            continue
        xs, ys = cell.exterior.xy
        ax.fill(xs, ys, alpha=0.4, color=color_map(index), zorder=1)
        ax.plot(xs, ys, color=color_map(index), linewidth=1.2, zorder=2)
    ax.scatter(
        germs[:, 0], germs[:, 1], c=np.arange(len(germs)), cmap=cmap_name,
        s=20, zorder=4, edgecolors="black", linewidths=0.8,
    )
    if annotate:
        for index, (x_coord, y_coord) in enumerate(germs):
            ax.annotate(
                rf"$s_{{{index}}}$", (x_coord, y_coord),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )
    ax.set_xlim(X_bounds)
    ax.set_ylim(Y_bounds)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savefigure:
        save_figure(fig, title_savefig, output_dir)
    if show:
        plt.show()
    return fig, ax
