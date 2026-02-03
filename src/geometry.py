import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

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
