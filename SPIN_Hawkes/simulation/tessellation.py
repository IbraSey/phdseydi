"""Spatial tessellation helpers used by simulations and examples."""

from numbers import Integral

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box


def _finite_voronoi_regions(vor: Voronoi, radius: float | None = None):
    """Reconstruct finite regions of a two-dimensional Voronoi diagram."""
    if vor.points.shape[1] != 2:
        raise ValueError("Only two-dimensional Voronoi diagrams are supported.")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max() * 4.0)

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(vertex >= 0 for vertex in vertices):
            new_regions.append(vertices)
            continue

        new_region = [vertex for vertex in vertices if vertex >= 0]
        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        points = np.asarray([new_vertices[vertex] for vertex in new_region])
        centroid = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        new_regions.append([vertex for _, vertex in sorted(zip(angles, new_region))])

    return new_regions, np.asarray(new_vertices)


def generate_voronoi_cells(
    n_germs: int = 5,
    X_bounds: tuple[float, float] = (0.0, 2.0),
    Y_bounds: tuple[float, float] = (0.0, 2.0),
    rng_seed: int | None = None,
):
    """Generate bounded Voronoi cells clipped to a rectangular domain.

    Returns one cell per germ, in the same order as the returned germ array.
    """
    if isinstance(n_germs, bool) or not isinstance(n_germs, Integral):
        raise ValueError("n_germs must be an integer.")
    n_germs = int(n_germs)
    if n_germs < 3:
        raise ValueError("n_germs must be at least 3 for a 2D Voronoi diagram.")
    xmin, xmax = map(float, X_bounds)
    ymin, ymax = map(float, Y_bounds)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("Spatial bounds must be strictly increasing.")

    rng = np.random.default_rng(rng_seed)
    germs = np.column_stack(
        [rng.uniform(xmin, xmax, n_germs), rng.uniform(ymin, ymax, n_germs)]
    )
    vor = Voronoi(germs)
    regions, vertices = _finite_voronoi_regions(vor)
    rectangle = box(xmin, ymin, xmax, ymax)

    cells = []
    for region in regions:
        polygon = Polygon(vertices[region]).intersection(rectangle)
        if polygon.is_empty:
            raise RuntimeError("A generated Voronoi cell is unexpectedly empty.")
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda geometry: geometry.area)
        cells.append(polygon)
    return cells, germs
