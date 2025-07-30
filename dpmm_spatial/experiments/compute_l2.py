import numpy as np
import traceback


def compute_l2_distance(Z_f, Z_g, grid_x, grid_y):
    """
    Compute the L2 distance between two 2D density estimates over a grid.

    Parameters
    ----------
    Z_f : ndarray of shape (n_y, n_x)
        First density evaluated on the grid.
    
    Z_g : ndarray of shape (n_y, n_x)
        Second density evaluated on the grid.
    
    grid_x : ndarray of shape (n_x,)
        Grid values along the x-axis.
    
    grid_y : ndarray of shape (n_y,)
        Grid values along the y-axis.

    Returns
    -------
    float
        L2 distance between Z_f and Z_g over the grid.
    """

    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    return np.sqrt(np.sum((Z_f - Z_g)**2) * dx * dy)


def evaluate_l2_distance_vs_param(
    param_values,
    reference_density,
    dpmm_density_constructor,
    grid_x,
    grid_y,
    param_name="alpha",
    constructor_kwargs=None,
    ref_args=None,
    verbose=True,
):
    """
    Evaluate the L2 distance between a reference density and a set of DPMM-estimated
    densities across values of a single parameter.

    Parameters
    ----------
    param_values : array-like
        List of parameter values to evaluate.
    
    reference_density : callable
        Reference density function, evaluated on a grid (X, Y).
    
    dpmm_density_constructor : callable
        Function that returns a callable density estimator when passed parameter values.
    
    grid_x : ndarray of shape (n_x,)
        Grid values along the x-axis.
    
    grid_y : ndarray of shape (n_y,)
        Grid values along the y-axis.

    param_name : str, default="alpha"
        Name of the varying parameter.

    constructor_kwargs : dict or None, default=None
        Base keyword arguments to pass to the DPMM constructor.

    ref_args : dict or None, default=None
        Additional arguments passed to the reference density function.

    verbose : bool, default=True
        If True, print progress and L2 distances during evaluation.

    Returns
    -------
    param_values : list
        List of parameter values used.

    l2_distances : list of float
        Corresponding L2 distances to the reference density.
    """
    constructor_kwargs = constructor_kwargs or {}
    ref_args = ref_args or {}

    X, Y = np.meshgrid(grid_x, grid_y)
    Z_ref = reference_density(X, Y, **ref_args)
    l2_distances = []

    for val in param_values:
        kwargs = constructor_kwargs.copy()
        kwargs[param_name] = val
        dpmm_density = dpmm_density_constructor(**kwargs)
        Z_dpmm = dpmm_density(X, Y)

        l2 = compute_l2_distance(Z_ref, Z_dpmm, grid_x, grid_y)
        l2_distances.append(l2)

        if verbose:
            print(f"{param_name} = {val:.4f} → Distance L² = {l2:.4f}")

    return param_values, l2_distances


def evaluate_l2_distance_vs_two_params(
    param1_values,
    param2_values,
    param1_name,
    param2_name,
    reference_density,
    dpmm_density_constructor,
    grid_x,
    grid_y,
    constructor_kwargs_base=None,
    ref_args=None,
    verbose=True
):
    """
    Evaluate the L2 distance between a reference density and DPMM-estimated densities
    across a grid of two varying parameters.

    Parameters
    ----------
    param1_values : array-like
        Values for the first varying parameter.

    param2_values : array-like
        Values for the second varying parameter.

    param1_name : str
        Name of the first parameter.

    param2_name : str
        Name of the second parameter.

    reference_density : callable
        Reference density function, evaluated on (X, Y).

    dpmm_density_constructor : callable
        Function that builds a density estimator from parameters.

    grid_x : ndarray
        Grid values along the x-axis.

    grid_y : ndarray
        Grid values along the y-axis.

    constructor_kwargs_base : dict or None, default=None
        Base parameters passed to the density constructor.

    ref_args : dict or None, default=None
        Additional arguments passed to the reference density.

    verbose : bool, default=True
        If True, print progress and L2 values.

    Returns
    -------
    param1_values : array
        Values of the first parameter.

    param2_values : array
        Values of the second parameter.

    distances : ndarray of shape (len(param1_values), len(param2_values))
        L2 distances between estimated and reference densities.
    """
    constructor_kwargs_base = constructor_kwargs_base or {}
    ref_args = ref_args or {}

    X, Y = np.meshgrid(grid_x, grid_y)
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]

    Z_ref = reference_density(X, Y, **ref_args)
    distances = np.zeros((len(param1_values), len(param2_values)))

    for i, p1 in enumerate(param1_values):
        for j, p2 in enumerate(param2_values):
            kwargs = constructor_kwargs_base.copy()
            kwargs[param1_name] = p1
            kwargs[param2_name] = p2

            try:
                dpmm_density_fn = dpmm_density_constructor(**kwargs)
                Z_dpmm = dpmm_density_fn(X, Y)
                diff_squared = (Z_ref - Z_dpmm) ** 2
                dist = np.sqrt(np.sum(diff_squared) * dx * dy)
                distances[i, j] = dist

                if verbose:
                    print(f"{param1_name}={p1:.3f}, {param2_name}={p2:.3f} → L2 = {dist:.4f}")

            except Exception as e:
                distances[i, j] = np.nan
                print(f"Erreur pour {param1_name}={p1:.3f}, {param2_name}={p2:.3f} : {e}")

    return param1_values, param2_values, distances


def eval_l2_dist_vs_two_params_avg_dpmm_inf(
    param1_values,
    param2_values,
    param1_name,
    param2_name,
    reference_density_array,
    grid_x,
    grid_y,
    N,
    dpmm_factory_fn,
    verbose=True
):
    """
    Evaluate the average L2 distance between a reference density and 
    an ensemble of DPMM density estimates across a 2D parameter grid.

    For each (param1, param2) pair, N realizations of the DPMM are averaged 
    before computing the L2 distance to the reference.

    Parameters
    ----------
    param1_values : array-like
        Grid values for the first parameter.

    param2_values : array-like
        Grid values for the second parameter.

    param1_name : str
        Name of the first parameter.

    param2_name : str
        Name of the second parameter.

    reference_density_array : ndarray
        Ground-truth density values evaluated over the grid.

    grid_x : ndarray
        Grid values along the x-axis.

    grid_y : ndarray
        Grid values along the y-axis.

    N : int
        Number of DPMM samples to average per grid point.

    dpmm_factory_fn : callable
        Function of (param1, param2) returning a DPMM estimator instance with `.evaluate_density(X, Y)`.

    verbose : bool, default=True
        Whether to print progress and L2 scores.

    Returns
    -------
    distances : ndarray of shape (len(param1_values), len(param2_values))
        Averaged L2 distances for each parameter pair.
    """
    X, Y = np.meshgrid(grid_x, grid_y)
    distances = np.zeros((len(param1_values), len(param2_values)))

    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            try:
                Z_sum = np.zeros_like(X)
                for _ in range(N):
                    dpmm = dpmm_factory_fn(val1, val2)()
                    Z = dpmm.evaluate_density(X, Y)
                    Z_sum += Z
                Z_avg = Z_sum / N

                dist = compute_l2_distance(Z_avg, reference_density_array, grid_x, grid_y)
                distances[i, j] = dist

                if verbose:
                    print(f"{param1_name}={val1:.2f}, {param2_name}={val2:.2f} → L² = {dist:.4f}")

            except Exception as e:
                distances[i, j] = np.nan
                print(f"[ERREUR] {param1_name}={val1}, {param2_name}={val2} → {e}")
                traceback.print_exc()

    return distances







