import numpy as np
import cv2

def find_first_nonzero_slices(image):
    """
    Finds the first slice index with non-zero data along each axis.
    
    Parameters:
        image (numpy.ndarray): 3D image array.
    
    Returns:
        dict: Dictionary with the first non-zero slice indices for each axis.
    """
    first_nonzero_slices = {}
    
    # Iterate over each axis (0, 1, 2)
    for axis in range(image.ndim):
        # Sum along the other two axes to collapse the image along the current axis
        projection = np.any(image > 0, axis=tuple(i for i in range(image.ndim) if i != axis))
        
        # Find the first non-zero slice along the current axis
        first_nonzero_slices[f"Axis {axis}"] = np.argmax(projection)
    
    return first_nonzero_slices

def find_last_nonzero_slices(image):
    """
    Finds the last slice index with non-zero data along each axis.

    Parameters:
        image (numpy.ndarray): 3D image array.

    Returns:
        dict: Dictionary with the last non-zero slice indices for each axis.
    """
    last_nonzero_slices = {}
    for axis in range(image.ndim):
        # Sum along the other axes to collapse the image along the current axis
        projection = np.any(image > 0, axis=tuple(i for i in range(image.ndim) if i != axis))
        
        # Find the last non-zero slice along the current axis
        last_nonzero_slices[f"Axis {axis}"] = len(projection) - 1 - np.argmax(projection[::-1])
    
    return last_nonzero_slices

def remove_slices(image, axis, n1, n2):
    """
    Removes n1 slices from the start and n2 slices from the end along the specified axis.
    
    Parameters:
        image (numpy.ndarray): 3D image array.
        axis (int): Axis along which to remove slices (0, 1, or 2).
        n1 (int): Number of slices to remove from the start.
        n2 (int): Number of slices to remove from the end.
    
    Returns:
        numpy.ndarray: The cropped 3D image.
    """
    if axis < 0 or axis >= image.ndim:
        raise ValueError(f"Axis {axis} is invalid for an image with {image.ndim} dimensions.")
    if n1 < 0 or n2 < 0:
        raise ValueError("n1 and n2 must be non-negative integers.")
    if n1 + n2 >= image.shape[axis]:
        raise ValueError("n1 + n2 exceeds the size of the image along the specified axis.")
    
    # Slicing to remove n1 slices from the start and n2 slices from the end
    slices = [slice(None)] * image.ndim  # Create a full slice for all dimensions
    slices[axis] = slice(n1, image.shape[axis] - n2)  # Modify the slice for the specified axis
    
    return image[tuple(slices)]

def add_slices(image, axis, n1, n2, fill_value=0):
    """
    Adds n1 slices to the start and n2 slices to the end along the specified axis.
    
    Parameters:
        image (numpy.ndarray): 3D image array.
        axis (int): Axis along which to add slices (0, 1, or 2).
        n1 (int): Number of slices to add at the start.
        n2 (int): Number of slices to add at the end.
        fill_value (int, float): Value to fill the added slices. Default is 0.
    
    Returns:
        numpy.ndarray: The expanded 3D image.
    """
    if axis < 0 or axis >= image.ndim:
        raise ValueError(f"Axis {axis} is invalid for an image with {image.ndim} dimensions.")
    if n1 < 0 or n2 < 0:
        raise ValueError("n1 and n2 must be non-negative integers.")
    
    # Create the shape of the padding for each dimension
    pad_shape = [(0, 0)] * image.ndim  # Default no padding for all axes
    pad_shape[axis] = (n1, n2)  # Add n1 slices at the start and n2 slices at the end
    
    # Add padding to the image
    return np.pad(image, pad_shape, mode="constant", constant_values=fill_value)

def scale_to_8bit(image, percentile=99.99):
    """
    Scales a 3D image to 8-bit (0-255) after clipping values above the 99.99 percentile.

    Steps:
    1. Calculate the minimum value of the image.
    2. Calculate the 99.99 percentile value.
    3. Clip values above the 99.99 percentile to the percentile value.
    4. Perform min-max scaling to 0-255 (8-bit).

    Parameters:
        image (numpy.ndarray): 3D image array.

    Returns:
        numpy.ndarray: Scaled 8-bit image.
    """
    # Step 1: Calculate the minimum and 99.99 percentile values
    image_min = np.min(image)
    percentile_9999 = np.percentile(image, percentile)

    # Step 2: Clip values above the 99.99 percentile
    image_clipped = np.clip(image, image_min, percentile_9999)

    # Step 3: Perform min-max scaling to [0, 255]
    image_scaled = (image_clipped - image_min) / (percentile_9999 - image_min) * 255
    # image_scaled = np.round(image_scaled).astype(np.uint8)  # Convert to 8-bit

    return image_scaled

def apply_clahe_to_roi(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to 2D or 3D images,
    focusing only on the non-zero region (ROI).

    Parameters:
        image (numpy.ndarray): Input image (2D or 3D array).
        clip_limit (float): Threshold for contrast limiting (default: 2.0).
        tile_grid_size (tuple): Size of the grid for histogram equalization (default: (8, 8)).

    Returns:
        numpy.ndarray: Image with CLAHE applied to the ROI.
    """
    # Check if the image is 2D or 3D
    if len(image.shape) == 2:  # 2D image
        # Create a mask for the non-zero region
        mask = image > 0

        # Normalize the image to uint8 for CLAHE
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Apply CLAHE to the entire slice
        clahe_image = clahe.apply(normalized_image)

        # Restore the CLAHE-enhanced ROI
        output_image = np.zeros_like(image, dtype=np.uint8)
        output_image[mask] = clahe_image[mask]

    elif len(image.shape) == 3:  # 3D image
        # Create a mask for the non-zero region
        mask = image > 0

        # Initialize output image
        output_image = np.zeros_like(image, dtype=np.uint8)

        # Apply CLAHE slice-by-slice along the third axis
        for i in range(image.shape[2]):
            slice_ = image[:, :, i]

            if np.any(mask[:, :, i]):  # Process only if the slice has non-zero values
                # Normalize the slice to uint8 for CLAHE
                normalized_slice = cv2.normalize(slice_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Create CLAHE object
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

                # Apply CLAHE to the normalized slice
                clahe_slice = clahe.apply(normalized_slice)

                # Restore the CLAHE-enhanced slice to the non-zero region
                output_image[:, :, i][mask[:, :, i]] = clahe_slice[mask[:, :, i]]

    else:
        raise ValueError("Input image must be a 2D or 3D array.")

    return output_image



