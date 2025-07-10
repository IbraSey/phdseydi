import numpy as np


def compute_l2_distance(Z_f, Z_g, grid_x, grid_y):
    """
    
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
    dpmm_density_fn,
    constructor_kwargs_base,
    verbose=True
):
    """
    Évalue la distance L² entre une densité de référence (déjà évaluée sur grille)
    et une moyenne empirique de densités DPMM en faisant varier deux hyperparamètres.

    Paramètres :
        - param1_values, param2_values : listes des valeurs à tester pour chaque hyperparamètre
        - param1_name, param2_name : noms des hyperparamètres à faire varier
        - reference_density_array : grille 2D de la densité de référence (Z)
        - grid_x, grid_y : vecteurs 1D de la grille
        - N : nombre de densités DPMM à moyenner
        - dpmm_density_fn : fonction `informative_dpmm_density(...)`
        - constructor_kwargs_base : dictionnaire des autres paramètres
        - verbose : bool

    Retourne :
        - distances : matrice (len(param1_values), len(param2_values)) des distances L²
    """
    distances = np.zeros((len(param1_values), len(param2_values)))
    X, Y = np.meshgrid(grid_x, grid_y)

    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            kwargs = constructor_kwargs_base.copy()
            kwargs[param1_name] = val1
            kwargs[param2_name] = val2

            try:
                Z_sum = np.zeros_like(X)
                for _ in range(N):
                    f_density = dpmm_density_fn(**kwargs)
                    Z_sum += f_density(X, Y)
                Z_mean = Z_sum / N

                l2 = compute_l2_distance(Z_mean, reference_density_array, grid_x, grid_y)
                distances[i, j] = l2

                if verbose:
                    print(f"{param1_name}={val1:.2f}, {param2_name}={val2:.2f} → L² = {l2:.4f}")
            except Exception as e:
                distances[i, j] = np.nan
                print(f"[ERREUR] {param1_name}={val1}, {param2_name}={val2} → {e}")

    return distances






