"""Diagnostic plots for simulations, partitions and posterior outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..simulation import SpatialProcessSimulation
from .fields import save_figure


def _simulation_payload(simulation, grids=None):
    if isinstance(simulation, SpatialProcessSimulation):
        sim_data, grid_data = simulation.as_mapping()
        return sim_data, grid_data
    if grids is None:
        raise ValueError("grids is required with a simulation-data dictionary.")
    return simulation, grids


def plot_process_dashboard(
    simulation,
    grids=None,
    cmap="viridis",
    latent_cmap="coolwarm",
    title: str = "Spatial process simulation",
    savefigure: bool = False,
    title_savefig: str = "process_dashboard.pdf",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    """Plot latent field, intensity, events and piecewise baseline domains."""
    sim_data, grid_data = _simulation_payload(simulation, grids)
    events = np.asarray(sim_data["X"], dtype=float)
    n_events = events.shape[0]
    x_bounds, y_bounds, _ = sim_data["bounds"]
    baseline = np.asarray(sim_data["mus_vec"], dtype=float)
    domains = sim_data["domains"]
    grid_x, grid_y = grid_data["GX"], grid_data["GY"]
    color_map = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    latent_ax = axes[0, 0]
    latent_image = latent_ax.contourf(
        grid_x, grid_y, grid_data["f_star"], levels=50, cmap=latent_cmap
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
        grid_x, grid_y, grid_data["mu_star"], levels=50, cmap=cmap
    )
    intensity_divider = make_axes_locatable(intensity_ax)
    fig.colorbar(
        intensity_image,
        cax=intensity_divider.append_axes("right", size="5%", pad=0.08),
        label=r"$\mu^\star(x,y)$",
    )
    intensity_ax.set_title(r"True intensity $\mu^\star=\tilde\mu\,\sigma(f^\star)$")

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
            rf"$\tilde\mu={baseline[index]:.2g}$",
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
        save_figure(fig, title_savefig, output_dir)
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
    title_savefig: str = "voronoi.pdf",
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
