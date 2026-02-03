from src.hough import (
    THETA_MIN, THETA_MAX, NUM_THETAS,
    STRONG_EDGE_VALUE, WEAK_EDGE_VALUE,
    compute_rho, create_hough_accumulator,
    weighted_vote, hough_line_transform, find_peaks
)
from src.edge_ops import generate_gradient_magnitude
import numpy as np

def deg2rad(degrees: float):
    return np.deg2rad(degrees)

def rad2deg(radians: float):
    return np.rad2deg(radians)