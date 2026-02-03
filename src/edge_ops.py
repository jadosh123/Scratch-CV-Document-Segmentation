import numpy as np

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
