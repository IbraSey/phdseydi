import numpy as np

def compute_l2_distance(Z1, Z2, dx, dy):
    return np.sqrt(np.sum((Z1 - Z2)**2) * dx * dy)
