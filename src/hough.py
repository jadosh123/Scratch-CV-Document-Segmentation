import numpy as np
from typing import Tuple, List, Optional

# Theta ranges from -90 to 89 degrees (180 values)
THETA_MIN = -90
THETA_MAX = 89
NUM_THETAS = 180

# Standard values for strong/weak edges (used in weighted voting)
STRONG_EDGE_VALUE = 255
WEAK_EDGE_VALUE = 128

def compute_rho(x: int, y: int, theta: float) -> float:
    """
    Compute the rho value for a point (x, y) at angle theta.
    
    The Hough transform represents a line in polar coordinates:
        rho = x * cos(theta) + y * sin(theta)
    
    Where:
        - rho is the perpendicular distance from origin to the line
        - theta is the angle of the perpendicular with respect to x-axis
        - (x, y) is a point that lies on the line
    
    Args:
        x: x-coordinate (column) of the point
        y: y-coordinate (row) of the point
        theta: angle in RADIANS
    
    Returns:
        rho value (can be negative)
    
    Example:
        >>> compute_rho(2, 2, np.deg2rad(45))  # Should return approximately 2.83
    """
    rho = x * np.cos(theta) + y * np.sin(theta)
    
    # If we draw a perpendicular line from the origin to line that passes through (x, y)
    # we get the distance, now to compute the rho we can use the sine and cosine of the theta angle
    # created by our perpendicular vector with the x-axis, the sum of x and y components multiplied
    # by cosine and sine respectively yields the rho value for the line that passes through that pixel
    
    return rho

def create_hough_accumulator(image_shape: Tuple[int, int], 
                              num_thetas: int = NUM_THETAS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create and initialize the Hough accumulator array for line detection.
    
    The accumulator is a 2D voting array where:
    - Rows correspond to different rho values
    - Columns correspond to different theta values
    
    Args:
        image_shape: (height, width) of the input image
        num_thetas: Number of theta bins (default 180 for 1-degree resolution)
    
    Returns:
        Tuple of (accumulator, rhos, thetas) where:
        - accumulator: 2D array of zeros with shape (num_rhos, num_thetas)
        - rhos: 1D array of rho values (from -diagonal to +diagonal)
        - thetas: 1D array of theta values in radians
    
    HINT: The maximum possible rho value equals the diagonal length of the image.
          Use: diagonal = sqrt(height^2 + width^2)
          Rho ranges from -diagonal to +diagonal.
    """
    height, width = image_shape
    
    # Calculate the diagonal length (maximum possible rho)
    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
    
    # Create array of theta values (in radians)
    thetas = np.deg2rad(np.arange(THETA_MIN, THETA_MAX + 1))

    rhos = np.arange(-diagonal, diagonal + 1)
    num_rhos = len(rhos)
    accumulator = np.zeros((num_rhos, num_thetas))
    
    # For the rhos array I initialized it from -diag to +diag+1 to include the final point,
    # num of rhos is just the length of our rhos array.
    # And for the accumulator I initialized it with the shape so that we have the equal
    # dimension of rho bins and equal dimension of theta bins
    
    return accumulator, rhos, thetas

def weighted_vote(gradient_value: float, 
                  max_gradient: float = 255.0,
                  min_weight: float = 0.1) -> float:
    """
    Calculate weighted vote based on gradient magnitude.
    
    EXTENSION BEYOND CLASS MATERIAL:
    Instead of each edge pixel contributing equally (vote=1), pixels with
    stronger gradients should contribute more to the accumulator. This makes
    the Hough Transform more robust to noise.
    
    Args:
        gradient_value: Gradient magnitude at the edge pixel (0-255)
        max_gradient: Maximum possible gradient value (default 255)
        min_weight: Minimum vote weight to prevent zero votes (default 0.1)
    
    Returns:
        Vote weight between min_weight and 1.0
    
    Formula:
        weight = max(gradient_value / max_gradient, min_weight)
    
    Example:
        >>> weighted_vote(255)  # Returns 1.0 (strongest edge)
        >>> weighted_vote(127.5)  # Returns 0.5 (medium edge)
        >>> weighted_vote(0)  # Returns 0.1 (minimum weight)
    """
    weight = np.max(gradient_value / max_gradient, min_weight)
    
    # We compute the vote weight of the edge pixel by picking the maximum between
    # the pixel's gradient value and the min weight. The bigger the gradient magnitude
    # of the current pixel the bigger the vote weight that is returned.
    
    return weight

def hough_line_transform(edge_image: np.ndarray, 
                          use_weighted: bool = False,
                          gradient_magnitude: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Hough Transform for line detection.
    
    For each edge pixel (x, y), vote for all possible lines that could pass
    through that point by iterating through all theta values and computing
    the corresponding rho.
    
    Args:
        edge_image: Binary image where edge pixels have value > 0
        use_weighted: If True, use gradient magnitude for weighted voting
        gradient_magnitude: Gradient magnitude image (required if use_weighted=True)
    
    Returns:
        Tuple of (accumulator, rhos, thetas)
    
    Algorithm:
        1. Create accumulator array
        2. Find all edge pixel coordinates
        3. For each edge pixel:
           a. For each theta value:
              - Compute rho
              - Convert rho to accumulator index
              - Increment accumulator (by 1 or by weight)
    """
    # Create accumulator
    accumulator, rhos, thetas = create_hough_accumulator(edge_image.shape)
    
    if accumulator is None:
        raise ValueError("create_hough_accumulator returned None - complete TODO 1 first!")
    
    # Get edge pixel coordinates
    # np.nonzero returns (row_indices, col_indices) = (y_coords, x_coords)
    edge_y, edge_x = np.nonzero(edge_image)

    cos_t = np.cos(thetas)[np.newaxis, :]
    sin_t = np.sin(thetas)[np.newaxis, :]

    x_coords = edge_x[:, np.newaxis]
    y_coords = edge_y[:, np.newaxis]

    # Instead of my old loop now I utilize numpy's operator overloading
    # to compute all rhos with vectorized operations to cutdown compute time
    all_rhos = (x_coords * cos_t) + (y_coords * sin_t)
    rho_indices = np.round(all_rhos + len(rhos)//2).astype(np.int32)

    # Store all weights
    if use_weighted and gradient_magnitude is not None:
        weights = gradient_magnitude[edge_y, edge_x]
        weights = weights[:, np.newaxis]
    else:
        weights = 1
    
    # Loop over rho columns and generate a boolean mask
    # Then use the boolean mask to grab valid rhos and valid weights
    # Then store them in the accumulator 
    for t_idx in range(len(thetas)):
        col_rhos = rho_indices[:, t_idx]
        valid_mask = (col_rhos >= 0) & (col_rhos < len(rhos))

        valid_r = col_rhos[valid_mask]

        if isinstance(weights, np.ndarray):
            valid_w = weights[valid_mask, 0]
            np.add.at(accumulator[:, t_idx], valid_r, valid_w)
        else:
            np.add.at(accumulator[:, t_idx], valid_r, 1)

    # For every edge pixel in our edge pixels image we iterate over a range of theta values
    # we compute the rho for the line passing through the edge pixel (x, y) then we compute the index
    # and check for its validity, afterwards we check if there is a weighted vote and a gradient magnitude
    # and increment the accumulator at that rho and theta values by the value of the vote
    
    return accumulator, rhos, thetas

def find_peaks(accumulator: np.ndarray, 
               rhos: np.ndarray, 
               thetas: np.ndarray,
               threshold: Optional[float] = None,
               num_peaks: int = 10) -> List[Tuple[float, float, int]]:
    """
    Find peaks (local maxima) in the Hough accumulator.
    
    Peaks represent the most likely lines in the image.
    
    Args:
        accumulator: 2D Hough accumulator array
        rhos: Array of rho values
        thetas: Array of theta values (in radians)
        threshold: Minimum vote count to consider (default: 50% of max)
        num_peaks: Maximum number of peaks to return
    
    Returns:
        List of tuples (rho, theta_degrees, votes) sorted by votes descending
    
    Algorithm:
        1. If threshold is None, set to 50% of max accumulator value
        2. Find all cells above threshold
        3. Sort by vote count (descending)
        4. Return top num_peaks results
    """
    if threshold is None:
        threshold = 0.5 * np.max(accumulator)

    x_idx, y_idx = np.where(accumulator > threshold)
    peaks = [(float(rhos[x_idx[i]]),
              float(np.rad2deg(thetas[y_idx[i]])),
              int(accumulator[x_idx[i], y_idx[i]])
            ) for i in range(len(x_idx))]
    
    peaks.sort(key=lambda item: item[2], reverse=True)
    
    # We first check the threshold and set it to default if its none,
    # afterwards we fetch all indices in accumulator above the threshold,
    # then we sort in descending order to get the top number of peaks at the first idx.
    # The peaks with highest vote counts tell us the highest confidence line candidates.
    # I used a lambda function in the sort function to sort by the 3rd element which is the vote count.
    
    return peaks[:num_peaks]

def get_lines(peaks):
    """
    Turns the peaks returned from hough accumulator into lines
    """
    lines = []
    for rho, theta_deg, _ in peaks:
        theta_rad = np.deg2rad(theta_deg)
        a = np.cos(theta_rad)
        b = np.sin(theta_rad)
        x0 = a * rho
        y0 = b * rho
        scale_factor = 4000
        x1 = int(x0 + scale_factor * (-b))
        y1 = int(y0 + scale_factor * (a))
        x2 = int(x0 - scale_factor * (-b))
        y2 = int(y0 - scale_factor * (a))
        lines.append((x1, y1, x2, y2))
    return lines
