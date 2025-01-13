### **Helper Scripts**

#### `src/utils/helpers.py`
import nibabel as nib
import numpy as np


def normalize_image(image):
    """Normalize image intensities to zero mean and unit variance."""
    return (image - np.mean(image)) / np.std(image)

def load_nifti(file_path):
    """
    Load a NIfTI file and return the image data and affine transformation matrix.
    Parameters:
        file_path (str): Path to the NIfTI file.
    Returns:
        np.ndarray: 3D image data.
        np.ndarray: Affine transformation matrix.
    """
    img = nib.load(file_path)
    return img.get_fdata(), img.affine


def save_nifti(filepath, data, reference):
    """Save a NIFTI file with reference metadata."""
    nib.save(nib.Nifti1Image(data, affine=reference.affine), filepath)

def compute_correlation_weights(fixed_image, registered_images):
    """
    Compute correlation-based weights for registered images.
    Parameters:
        fixed_image (np.ndarray): The validation image (fixed image).
        registered_images (list of np.ndarray): List of elastically registered images.
    Returns:
        np.ndarray: Normalized correlation-based weights (1D array).
    """
    # Ensure fixed_image and registered_images are 3D arrays
    fixed_image = np.squeeze(fixed_image)
    registered_images = [np.squeeze(img) for img in registered_images]

    # Compute Pearson correlation coefficients
    correlations = []
    for reg_img in registered_images:
        # Compute Pearson correlation coefficient
        corr = np.corrcoef(fixed_image.ravel(), reg_img.ravel())[0, 1]
        correlations.append(max(0, corr))  # Use only positive correlations

    # Normalize weights to sum to 1
    weights = np.array(correlations) / (np.sum(correlations) + 1e-10)

    return weights

def compute_mutual_information(fixed_window, registered_window, bins=256):
    """
    Compute the mutual information between two image windows.
    Parameters:
        fixed_window (np.ndarray): The fixed (reference) image window.
        registered_window (np.ndarray): The registered atlas image window.
        bins (int): Number of bins for the joint histogram.
    Returns:
        float: The mutual information value.
    """
    # Joint histogram
    joint_hist, _, _ = np.histogram2d(fixed_window.ravel(), registered_window.ravel(), bins=bins, density=True)

    # Marginal histograms
    p_fixed = np.sum(joint_hist, axis=1)
    p_registered = np.sum(joint_hist, axis=0)

    # Compute mutual information
    p_joint = joint_hist + 1e-10  # Avoid division by zero
    p_fixed = p_fixed + 1e-10
    p_registered = p_registered + 1e-10
    mutual_info = np.sum(p_joint * np.log(p_joint / (p_fixed[:, None] * p_registered[None, :])))
    return mutual_info