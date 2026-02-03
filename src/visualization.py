import numpy as np
import pandas as pd
from typing import List, Tuple
from src.geometry import is_path_clear

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
        
    return img_out
