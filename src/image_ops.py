import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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

def get_binary_mask(image, threshold=230):
    mask = np.zeros_like(image)
    mask[image > threshold] = 255
    return mask
