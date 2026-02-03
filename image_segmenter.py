import numpy as np
import pandas as pd
import os
import cv2 
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
import csv
from typing import Tuple, List, Optional


DF_HEADERS = ('filename', 'x1', 'y1', 'x2', 'y2')
CSV_HEADERS = ('filename', 'x1', 'y1', 'x2', 'y2')

# Theta ranges from -90 to 89 degrees (180 values)
THETA_MIN = -90
THETA_MAX = 89
NUM_THETAS = 180

# Standard values for strong/weak edges (used in weighted voting)
STRONG_EDGE_VALUE = 255
WEAK_EDGE_VALUE = 128

def generate_gradient_magnitude(edge_image: np.ndarray) -> np.ndarray:
    """
    Generate synthetic gradient magnitudes for edge pixels.
    For real applications, this would come from Sobel/Canny.
    
    Returns:
        Gradient magnitude image (same shape as input)
    """
    # Simple gradient: higher values toward center of image
    height, width = edge_image.shape
    gradient = np.zeros_like(edge_image, dtype=np.float64)
    
    center_y, center_x = height // 2, width // 2
    max_dist = np.sqrt(center_y**2 + center_x**2)
    
    for y in range(height):
        for x in range(width):
            if edge_image[y, x] > 0:
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                gradient[y, x] = 255 * (1 - dist / max_dist) + 50
    
    return gradient

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
        >>> compute_rho(2, 2, deg2rad(45))  # Should return approximately 2.83
    """
    rho = x * np.cos(theta) + y * np.sin(theta)
    
    # If we draw a perpendicular line from the origin to line that passes through (x, y)
    # we get the distance, now to compute the rho we can use the sine and cosine of the theta angle
    # created by our perpendicular vector with the x-axis, the sum of x and y components multiplied
    # by cosine and sine respectively yields the rho value for the line that passes through that pixel
    
    return rho

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

def convolve_sliding_window(image, kernel):
    """
    Applies convolution using the standard Sliding Window approach.
    Optimized with einsum to prevent RAM crashes.
    """
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge').astype(np.float32)
    
    # This avoids copying and bloating the RAM because
    # it gives us a virtual view of the pixels instead of new objects
    windows = sliding_window_view(image_padded, (k_h, k_w))
    
    # Take windows (ijkl) and kernel (kl)
    # multiply the matching last two dimensions (kl)
    # sum them up
    # result is just the image dimensions (ij)
    output = np.einsum('ijkl,kl->ij', windows, kernel)
    
    return output

def morphology_1d(image, kernel_size, axis, operation='dilation'):
    """
    Applies 1D Min/Max filter along a specific axis (0=Vertical, 1=Horizontal).
    This is used to build a separable 2D filter.
    """
    h, w = image.shape
    pad = kernel_size // 2
    
    # Pad depending on axis
    if axis == 0:
        padded = np.pad(image, ((pad, pad), (0, 0)), mode='edge')
        windows = sliding_window_view(padded, (kernel_size, 1))
    else:
        padded = np.pad(image, ((0, 0), (pad, pad)), mode='edge')
        windows = sliding_window_view(padded, (1, kernel_size))
    
    if operation == 'dilation':
        return np.max(windows, axis=(-1, -2))
    elif operation == 'erosion':
        return np.min(windows, axis=(-1, -2))

def manual_closing_grayscale(image, kernel_size=15):
    """
    Separable implementation of Grayscale Closing.
    1. Dilation (Max) - Swallows dark text into bright paper.
    2. Erosion (Min) - Restores paper boundaries.
    """
    dil_x = morphology_1d(image, kernel_size, axis=1, operation='dilation')
    dilated = morphology_1d(dil_x, kernel_size, axis=0, operation='dilation')
    
    ero_x = morphology_1d(dilated, kernel_size, axis=1, operation='erosion')
    closed = morphology_1d(ero_x, kernel_size, axis=0, operation='erosion')
    
    return closed

def my_gaussian_blur(image):
    """
    Applies gaussian blur with a standard 5x5 kernel
    """
    kernel = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]], dtype=np.float32) / 273.0
    
    output = convolve_sliding_window(image, kernel)
    return np.clip(output, 0, 255).astype(np.uint8)

def my_sobel_edge(blurred_image):
    """
    Computes the magnitude and direction of the gradients
    """   
    Gx_kernel = np.array([[-1, 0, 1], 
                          [-2, 0, 2], 
                          [-1, 0, 1]], dtype=np.float32)
            
    Gy_kernel = np.array([[-1, -2, -1], 
                          [ 0,  0,  0], 
                          [ 1,  2,  1]], dtype=np.float32)

    grad_x = convolve_sliding_window(blurred_image, Gx_kernel)
    grad_y = convolve_sliding_window(blurred_image, Gy_kernel)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize
    magnitude = np.clip(magnitude, 0, 255)

    return magnitude, grad_x, grad_y

def non_maximum_suppression(magnitude, grad_x, grad_y):
    """
    Returns a suprressed thin line of the greatest magnitudes
    """

    rows, cols = magnitude.shape
    target = np.zeros((rows, cols), dtype=np.float32)

    angles = np.rad2deg(np.arctan2(grad_y, grad_x))
    angles[angles < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            angle = angles[i, j]

            # 0 degrees (Vertical Edge, Horizontal Gradient)
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # 45 degrees
            elif (22.5 <= angle < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # 90 degrees (Horizontal Edge, Vertical Gradient)
            elif (67.5 <= angle < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # 135 degrees
            elif (112.5 <= angle < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            # If current pixel is greater than neighbors, keep it. Otherwise kill it.
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                target[i, j] = magnitude[i, j]
            else:
                target[i, j] = 0
                
    return target

def fetch_image_paths(data_path: str):
    tmp_li = []
    if not os.path.exists(data_path): return []
    for item in os.listdir(data_path):
        tmp_li.append(os.path.join(data_path, item))
    return tmp_li

def clip_lines(df, img):
    """
    Clips lines so that they are within image boundaries
    """

    h, w = img.shape[:2]

    clipped_lines = []
    for _, line in df.iterrows():
        x1, y1, x2, y2 = line

        # Non-sloped lines
        if x1 == x2:
            new_y1 = max(0, min(y1, y2))
            new_y2 = min(h, max(y1, y2))
            
            if new_y1 < new_y2:
                clipped_lines.append((x1, int(new_y1), x1, int(new_y2)))
            continue
            
        # Sloped lines
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        candidates = []
        
        # Intersection with Left Wall (x=0) -> y = b
        y_left = b
        if 0 <= y_left <= h:
            candidates.append((0, int(y_left)))
        
        # Intersection with Right Wall (x=w) -> y = mw + b
        y_right = m * w + b
        if 0 <= y_right <= h:
            candidates.append((w, int(y_right)))
        
        # Intersection with Top Wall (y=0) -> x = -b/m
        if m != 0:
            x_top = -b / m
            if 0 <= x_top <= w:
                candidates.append((int(x_top), 0))
            
            # Intersection with Bottom Wall (y=h) -> x = (h-b)/m
            x_bot = (h - b) / m
            if 0 <= x_bot <= w:
                candidates.append((int(x_bot), h))
        
        # We need unique points (corners might be added twice)
        candidates = sorted(list(set(candidates)))
        
        # We need exactly 2 points to make a segment
        if len(candidates) >= 2:
            # Taking the first and last ensures we get the max span across the image
            p1, p2 = candidates[0], candidates[-1]
            clipped_lines.append((p1[0], p1[1], p2[0], p2[1]))

    # Return a new clean DataFrame
    return pd.DataFrame(clipped_lines, columns=['x1', 'y1', 'x2', 'y2'])

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

def save_img(img):
    pass

def find_document_corners(horizontal_lines, vertical_lines):
    """
    Finds all intersection points between every horizontal and vertical line.
    Returns a list of (x, y) points.
    """
    intersections = []
    
    # Convert DataFrame to list of tuples if needed
    if isinstance(horizontal_lines, pd.DataFrame):
        h_list = horizontal_lines[['x1', 'y1', 'x2', 'y2']].values.tolist()
    else:
        h_list = horizontal_lines
        
    if isinstance(vertical_lines, pd.DataFrame):
        v_list = vertical_lines[['x1', 'y1', 'x2', 'y2']].values.tolist()
    else:
        v_list = vertical_lines

    for h_line in h_list:
        for v_line in v_list:
            point = compute_intersection(h_line, v_line)
            if point:
                intersections.append(point)
                
    return intersections

def compute_intersection(line_h, line_v):
    """
    Calculates the (x, y) intersection of two lines.
    Lines are tuples: (x1, y1, x2, y2)
    """
    xh1, yh1, xh2, yh2 = line_h
    xv1, yv1, xv2, yv2 = line_v
    
    # Line H represented as a1*x + b1*y = c1
    a1 = yh2 - yh1
    b1 = xh1 - xh2
    c1 = a1 * xh1 + b1 * yh1
    
    # Line V represented as a2*x + b2*y = c2
    a2 = yv2 - yv1
    b2 = xv1 - xv2
    c2 = a2 * xv1 + b2 * yv1
    
    det = a1 * b2 - a2 * b1
    
    if det == 0:
        return None

    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det

    return int(x), int(y)

def clean_intersections(binary_mask, intersections, size=5):
    """
    Validates and drops intersections that are fully inside documents
    """
    h, w = binary_mask.shape
    r = size // 2

    clean_pairs = []
    
    for cx, cy in intersections:
        # Grab the Virtual Slice
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        virtual_window = binary_mask[y1:y2, x1:x2]
        
        if np.mean(virtual_window) > 150 or np.max(virtual_window) > 200:
            clean_pairs.append((cx, cy))
            
    return clean_pairs

def get_binary_mask(image, threshold=230):
    mask = np.zeros_like(image)
    mask[image > threshold] = 255
    return mask

def filter_and_cluster_lines(lines, binary_mask, orientation='horizontal', min_gap=50):
    """
    1. Scores lines based on whiteness (Background support).
    2. Clusters them to remove duplicates.
    """
    scored_lines = []
    h, w = binary_mask.shape
    
    # --- 1. SCORE ---
    for line in lines:
        x1, y1, x2, y2 = line
        
        # Measure length
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            continue
        
        # Sample points along the line
        num_samples = length
        # Using linspace to get coordinates
        xs = np.linspace(x1, x2, num_samples).astype(int)
        ys = np.linspace(y1, y2, num_samples).astype(int)

        # Mask to get pixels inside image bounds
        valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        valid_xs = xs[valid_mask]
        valid_ys = ys[valid_mask]
        pixels = binary_mask[valid_ys, valid_xs]
        score = np.mean(pixels) / 255.0
        
        # Keep lines that are atleast X% on the background
        if score > 0.2:
            scored_lines.append((score, line))
            
    # --- 2. CLUSTER ---
    # Sort: Highest score (whitest) first
    scored_lines.sort(key=lambda x: x[0], reverse=True)
    
    final_lines = []
    
    for score, line in scored_lines:
        x1, y1, x2, y2 = line
        
        if orientation == 'horizontal':
            candidate_pos = (y1 + y2) / 2
        else:
            candidate_pos = (x1 + x2) / 2
            
        is_duplicate = False
        for final_line in final_lines:
            fx1, fy1, fx2, fy2 = final_line
            
            if orientation == 'horizontal':
                final_pos = (fy1 + fy2) / 2
            else:
                final_pos = (fx1 + fx2) / 2
                
            if abs(candidate_pos - final_pos) < min_gap:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_lines.append(line)
            
    return final_lines

def is_path_clear(binary_image, p1, p2):
    """
    Checks if the straight line between p1 and p2 consists of background pixels (255).
    """
    x1, y1 = p1
    x2, y2 = p2

    # Using this for a wiggle stratgy to check corridors for valid lines
    offsets = [0, 5, -5]
    
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length <= 5: return True 
    
    num_samples = max(length // 5, 2)

    xs_orig = np.linspace(x1, x2, num_samples).astype(int)
    ys_orig = np.linspace(y1, y2, num_samples).astype(int)
    
    h, w = binary_image.shape
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    is_horizontal = dx > dy

    for offset in offsets:
        if is_horizontal:
            xs = xs_orig
            ys = ys_orig + offset
        else:
            xs = xs_orig + offset
            ys = ys_orig

        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        pixels = binary_image[ys, xs]
    
        # Strict tolerance: Path must be mostly white
        if np.mean(pixels) > 150: 
            return True
    return False

def draw_line(img, p1, p2, color, thickness=4):
    """
    Draws a line on a numpy array by calculating vector points.
    Handles slopes and thickness purely with numpy masking.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate how many points we need to make the line look solid
    # We use the hypotenuse distance to ensure no gaps
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0: return

    # Generate all coordinates along the line (float for precision, then int)
    x_coords = np.linspace(x1, x2, length)
    y_coords = np.linspace(y1, y2, length)
    
    # To handle thickness, we simply draw the same line at small offsets
    # This simulates a "brush" moving along the path
    radius = thickness // 2
    h, w = img.shape[:2]
    
    # We iterate over a small kernel (-2 to +2 for thickness=4)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            # Shift the coordinates
            xs = (x_coords + dx).astype(int)
            ys = (y_coords + dy).astype(int)
            
            # Clip to image bounds to prevent crashes
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)
            
            # Color the pixels (Batch assignment is very fast)
            img[ys, xs] = color

def draw_circle(img, center, radius, color):
    """
    Draws a filled circle on a numpy array using distance masking.
    """
    cx, cy = center
    h, w = img.shape[:2]
    
    # 1. Define a bounding box (ROI) to avoid checking every single pixel in the image
    y_min, y_max = max(0, cy - radius), min(h, cy + radius + 1)
    x_min, x_max = max(0, cx - radius), min(w, cx + radius + 1)
    
    if y_min >= y_max or x_min >= x_max:
        return

    # 2. Create a grid of coordinates for just this small box
    # np.ogrid creates coordinate matrices
    Y, X = np.ogrid[y_min:y_max, x_min:x_max]
    
    # 3. Calculate squared distance from center (Pythagoras)
    # (x-cx)^2 + (y-cy)^2 <= r^2
    dist_sq = (X - cx)**2 + (Y - cy)**2
    
    # 4. Create a boolean mask where pixels are inside the circle
    mask = dist_sq <= radius**2
    
    # 5. Apply color using the mask on the region of interest
    # We select the box [y_min:y_max, x_min:x_max], apply the mask, and set color
    img[y_min:y_max, x_min:x_max][mask] = color

def draw_validated_grid_pil(image, binary_mask, horizontal_lines, vertical_lines, intersections):
    """
    Draws valid segments using PIL (ImageDraw) instead of cv2.
    """
    img_out = image.copy()
    h, w = binary_mask.shape
    
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)

    # Process Horizontal
    for _, row in horizontal_lines.iterrows():
        y_center = int((row['y1'] + row['y2']) / 2)
        
        points = [(0, y_center), (w, y_center)]
        for (cx, cy) in intersections:
            if abs(cy - y_center) < 10:
                points.append((cx, y_center))
        
        # Sorting by horizontal order
        points.sort(key=lambda p: p[0])
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            if is_path_clear(binary_mask, p1, p2):
                draw_line(img_out, p1, p2, COLOR_RED, thickness=4)

    # Process Vertical
    for _, row in vertical_lines.iterrows():
        x_center = int((row['x1'] + row['x2']) / 2)
        
        points = [(x_center, 0), (x_center, h)]
        for (cx, cy) in intersections:
            if abs(cx - x_center) < 10:
                points.append((x_center, cy))
        
        # Sorting by vertical order
        points.sort(key=lambda p: p[1])
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            if is_path_clear(binary_mask, p1, p2):
                draw_line(img_out, p1, p2, COLOR_RED, thickness=4)

    # Draw Intersections (Green circles)
    r = 8
    for cx, cy in intersections:
        draw_circle(img_out, (cx, cy), radius=8, color=COLOR_GREEN)

    # Convert back to NumPy for compatibility if needed (or just save pil_image directly)
    return img_out

def main():
    data_path = './data'
    paths = fetch_image_paths(data_path)

    with open('lines_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

    if not paths:
        print("No images found.")
        return

    for i in range(len(paths)):
        img_real = cv2.imread(paths[i])
        img_gray = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        
        path = Path(paths[i])
        file_name = path.name

        img_blurred = my_gaussian_blur(img_gray)

        img_closed = manual_closing_grayscale(img_blurred, kernel_size=15)

        # Hold this for post-processing
        binary_mask = get_binary_mask(img_closed, threshold=230)

        # Edge detection logic
        magnitude, grad_x, grad_y = my_sobel_edge(img_closed)
        nms_res = non_maximum_suppression(magnitude, grad_x, grad_y)
        img_edges = np.zeros_like(nms_res, dtype=np.uint8)
        img_edges[nms_res >= 30] = 255
        
        # Hough line logic
        accumulator, rhos, thetas = hough_line_transform(img_edges)
        peaks = find_peaks(accumulator, rhos, thetas, 50, 50) 
        lines = get_lines(peaks)
        
        if not lines:
            print("No lines found.")
            return

        # Post Processing (Pandas)
        df = pd.DataFrame(data=lines, columns=DF_HEADERS[1:])

        df['x_spread'] = (df['x1'] - df['x2']).abs()
        df['y_spread'] = (df['y1'] - df['y2']).abs()
        horizontal_lines = df[df['x_spread'] > df['y_spread']]
        vertical_lines = df[df['y_spread'] > df['x_spread']]

        # Filter lines by white pixel covered ratio and drop clusters
        final_horizontal_lines = filter_and_cluster_lines(
            horizontal_lines[['x1', 'y1', 'x2', 'y2']].values.tolist(),
            binary_mask,
            'horizontal',
            min_gap=25
        )
        final_vertical_lines = filter_and_cluster_lines(
            vertical_lines[['x1', 'y1', 'x2', 'y2']].values.tolist(),
            binary_mask,
            'vertical',
            min_gap=25
        )

        final_horizontal_lines = pd.DataFrame(final_horizontal_lines, columns=['x1', 'y1', 'x2', 'y2'])
        final_vertical_lines = pd.DataFrame(final_vertical_lines, columns=['x1', 'y1', 'x2', 'y2'])

        intersections = find_document_corners(final_horizontal_lines, final_vertical_lines)
        intersections = clean_intersections(binary_mask, intersections)

        # Saving the final outlined image with intersections
        out_dir = 'annotated_images'
        os.makedirs(out_dir, exist_ok=True)
        img_outlined = draw_validated_grid_pil(
            img_real,
            binary_mask,
            final_horizontal_lines[['x1', 'y1', 'x2', 'y2']],
            final_vertical_lines[['x1', 'y1', 'x2', 'y2']],
            intersections
        )  # Could change return from nparray to the actual image to save it
        img_name = os.path.splitext(file_name)[0]
        new_img_name = f"{img_name}_annotated.jpg"
        save_path = os.path.join(out_dir, new_img_name)
        cv2.imwrite(save_path, img_outlined)

        # Saving the final outputs to csv
        combined_lines = pd.concat([final_horizontal_lines, final_vertical_lines])
        final_df = combined_lines[['x1', 'y1', 'x2', 'y2']].copy()
        final_df = clip_lines(final_df, img_real)
        final_df.insert(0, 'filename', file_name)
        final_df.to_csv(
            'lines_data.csv',
            mode='a',
            header=False,
            index=False
        )

if __name__ == "__main__":
    main()