import numpy as np
from skimage.util import view_as_windows


def extract_windows_vectorized(volume, window_size, stride):
    """
    Extract overlapping windows from a 3D volume using vectorized operations.
    Parameters:
        volume (np.ndarray): 3D volume to extract windows from.
        window_size (tuple): Size of the window (width, height, depth).
        stride (tuple): Stride for the sliding window.
    Returns:
        tuple: 
            - np.ndarray: Array of windows with shape (num_windows, width, height, depth).
            - np.ndarray: Array of coordinates with shape (num_windows, 3).
    """
    # Generate sliding windows
    windows = view_as_windows(volume, window_size, step=stride)  # 6D array

    # Reshape windows to (num_windows, width, height, depth)
    num_windows = np.prod(windows.shape[:3])
    reshaped_windows = windows.reshape((num_windows, *window_size))  # Shape: (num_windows, width, height, depth)

    # Generate coordinates
    coords = np.array(np.meshgrid(
        np.arange(0, volume.shape[0] - window_size[0] + 1, stride[0]),
        np.arange(0, volume.shape[1] - window_size[1] + 1, stride[1]),
        np.arange(0, volume.shape[2] - window_size[2] + 1, stride[2]),
        indexing="ij"
    )).reshape(3, -1).T  # Shape: (num_windows, 3)

    return reshaped_windows, coords

def extract_windows(volume, window_size, stride):
    """
    Extract overlapping windows from a 3D volume.
    Parameters:
        volume (np.ndarray): 3D volume to extract windows from.
        window_size (tuple): Size of the window (width, height, depth).
        stride (tuple): Stride for the sliding window.
    Returns:
        list: List of windows and their corresponding coordinates.
    """
    windows = []
    coords = []
    w, h, d = volume.shape
    ww, wh, wd = window_size
    sw, sh, sd = stride

    for x in range(0, w - ww + 1, sw):
        for y in range(0, h - wh + 1, sh):
            for z in range(0, d - wd + 1, sd):
                windows.append(volume[x:x + ww, y:y + wh, z:z + wd])
                coords.append((x, y, z))
    return windows, coords


def merge_windows(volume_shape, windows, coords):
    """
    Merge overlapping windows into the full volume.
    Parameters:
        volume_shape (tuple): Shape of the original volume.
        windows (list of np.ndarray): List of windows to merge.
        coords (list of tuple): Corresponding coordinates of each window.
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    merged_volume = np.zeros(volume_shape, dtype=np.float64)
    weight_map = np.zeros(volume_shape, dtype=np.float64)

    for window, (x, y, z) in zip(windows, coords):
        ww, wh, wd = window.shape
        merged_volume[x:x + ww, y:y + wh, z:z + wd] += window
        weight_map[x:x + ww, y:y + wh, z:z + wd] += 1

    # Normalize by the weight map to account for overlapping windows
    merged_volume /= np.maximum(weight_map, 1)
    return merged_volume