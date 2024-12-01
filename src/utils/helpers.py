### **Helper Scripts**

#### `src/utils/helpers.py`

import numpy as np

def compute_dice_coefficient(seg1, seg2):
    """Compute Dice Similarity Coefficient between two binary masks."""
    intersection = np.sum((seg1 > 0) & (seg2 > 0))
    return 2 * intersection / (np.sum(seg1 > 0) + np.sum(seg2 > 0))

def normalize_image(image):
    """Normalize image intensities to zero mean and unit variance."""
    return (image - np.mean(image)) / np.std(image)

def load_nifti(filepath):
    """Load a NIFTI file using nibabel."""
    import nibabel as nib
    return nib.load(filepath).get_fdata()

def save_nifti(filepath, data, reference):
    """Save a NIFTI file with reference metadata."""
    import nibabel as nib
    nib.save(nib.Nifti1Image(data, affine=reference.affine), filepath)