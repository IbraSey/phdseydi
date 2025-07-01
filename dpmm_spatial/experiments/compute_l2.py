import numpy as np

def compute_l2_distance(f1, f2, dx, dy):
    """
    En chantier
    """
    return np.sqrt(np.sum((f1 - f2)**2) * dx * dy)


