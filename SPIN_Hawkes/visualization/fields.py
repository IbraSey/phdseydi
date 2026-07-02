"""Visualisation helpers for OpenTURNS fields."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "figures"
FIGURE_DPI = 50


def save_figure(
    fig,
    filename: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = FIGURE_DPI,
) -> Path:
    """Save a figure below ``spin_h/artifacts/figures`` at 50 dpi by default.

    A ``.pdf`` extension is added when ``filename`` has no suffix. Nested
    paths are supported, which lets each experiment own a dedicated folder.
    """
    path = Path(filename)
    if not path.suffix:
        path = path.with_suffix(".pdf")
    directory = DEFAULT_FIGURES_DIR if output_dir is None else Path(output_dir)
    destination = directory / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=int(dpi), bbox_inches="tight")
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
    title_savefig: str = "field_output.pdf",
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
        save_figure(fig, title_savefig, output_dir)
    if show and mode == "plot":
        plt.show()
    return fig, local_ax, contour
