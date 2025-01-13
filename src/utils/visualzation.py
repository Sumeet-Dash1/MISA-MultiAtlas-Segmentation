import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

def plot_2d_slice(image, seg=None, slice_index=None, title="2D Slice", cmap="gray"):
    """
    Plot a 2D slice of the image with optional segmentation overlay.
    Parameters:
        image (np.ndarray): 3D image array.
        seg (np.ndarray, optional): 3D segmentation array (same dimensions as `image`).
        slice_index (int, optional): Index of the slice to plot. Defaults to the middle slice.
        title (str): Title for the plot.
        cmap (str): Colormap for the image.
    """
    if slice_index is None:
        slice_index = image.shape[2] // 2  # Default to the middle slice

    plt.figure(figsize=(8, 8))
    plt.imshow(image[:, :, slice_index], cmap=cmap, interpolation="none")
    if seg is not None:
        plt.contour(seg[:, :, slice_index], levels=[0.5], colors="red", linewidths=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_3d_segmentation(seg, title="3D Segmentation", threshold=0.5):
    """
    Visualize a 3D segmentation using isosurfaces.
    Parameters:
        seg (np.ndarray): 3D segmentation array.
        title (str): Title for the plot.
        threshold (float): Threshold to create the isosurface.
    """
    verts, faces, _, _ = measure.marching_cubes(seg, level=threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    mesh = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        cmap="Spectral", lw=1, alpha=0.6
    )
    ax.set_title(title)
    plt.show()

def visualize_overlay(image, seg, alpha=0.5, slice_index=None, cmap="gray", title="Overlay"):
    """
    Overlay segmentation on the image.
    Parameters:
        image (np.ndarray): 3D image array.
        seg (np.ndarray): 3D segmentation array.
        alpha (float): Transparency level of the overlay.
        slice_index (int, optional): Index of the slice to plot. Defaults to the middle slice.
        cmap (str): Colormap for the image.
        title (str): Title for the plot.
    """
    if slice_index is None:
        slice_index = image.shape[2] // 2  # Default to the middle slice

    plt.figure(figsize=(8, 8))
    plt.imshow(image[:, :, slice_index], cmap=cmap, interpolation="none")
    plt.imshow(seg[:, :, slice_index], cmap="Reds", alpha=alpha, interpolation="none")
    plt.title(title)
    plt.axis("off")
    plt.show()

def load_nifti_and_plot(filepath, slice_index=None):
    """
    Load a NIFTI file and plot a 2D slice.
    Parameters:
        filepath (str): Path to the NIFTI file.
        slice_index (int, optional): Index of the slice to plot. Defaults to the middle slice.
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    plot_2d_slice(data, slice_index=slice_index, title=f"Slice from {filepath}")

def analyze_image(fixed_image, ground_truth=None, set_name='Train_Set'):
    """
    Analyzes a given fixed image and its ground truth, computes statistics, and plots histograms and slices.
    
    Parameters:
        fixed_image (numpy.ndarray): 3D array of the fixed image.
        ground_truth (numpy.ndarray): 3D array of the ground truth (optional, None for test sets).
        set_name (str): Name of the dataset ('Train_Set', 'Validation_Set', or 'Test_Set').
    
    Returns:
        dict: Statistics of the fixed image and ground truth.
    """
    # Intensity Statistics
    stats = {
        "Min": np.min(fixed_image),
        "Min (Excluding Zeros)": np.min(fixed_image[fixed_image > 0]),
        "Max": np.max(fixed_image),
        "Image Type": str(fixed_image.dtype),
        "Intensity Range": (fixed_image.min(), fixed_image.max()),
        "Voxel Count (Total)": fixed_image.size,
        "Voxel Count (Non-Zero)": np.count_nonzero(fixed_image),
        "99.99th Percentile": np.percentile(fixed_image, 99.99)
    }

    # Count of voxels per label in ground truth
    if ground_truth is not None:
        unique_labels, voxel_counts = np.unique(ground_truth, return_counts=True)
        label_distribution = dict(zip(unique_labels, voxel_counts))
    else:
        label_distribution = None

    # Plotting histograms and slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Fixed image histogram (excluding zero values)
    non_zero_fixed_image = fixed_image[fixed_image > 0]
    axes[0, 0].hist(non_zero_fixed_image.ravel(), bins=50, color='blue', alpha=0.7)
    axes[0, 0].set_title("Histogram of Fixed Image (Excluding Zeros)")
    axes[0, 0].set_xlabel("Pixel Intensity")
    axes[0, 0].set_ylabel("Frequency")

    # Adding a vertical line at the 99.99th Percentile
    axes[0, 0].axvline(
        x=stats["99.99th Percentile"], 
        color='red', linestyle='--', linewidth=2, 
        label=f'99.99th Percentile = {stats["99.99th Percentile"]:.2f}'
    )
    axes[0, 0].legend()

    # Ground truth histogram
    if ground_truth is not None and set_name != 'Test_Set':
        axes[0, 1].bar(unique_labels, voxel_counts, color='orange', alpha=0.7)
        axes[0, 1].set_title("Ground Truth Label Distribution")
        axes[0, 1].set_xlabel("Labels")
        axes[0, 1].set_ylabel("Voxel Count")

    # Slices of fixed image
    slice_index = fixed_image.shape[2] // 2  # Taking middle slice
    axes[1, 0].imshow(fixed_image[:, :, slice_index], cmap="gray")
    axes[1, 0].set_title("Fixed Image (Middle Slice)")
    axes[1, 0].axis("off")

    # Slices of ground truth
    if ground_truth is not None and set_name != 'Test_Set':
        axes[1, 1].imshow(ground_truth[:, :, slice_index], cmap="tab10")
        axes[1, 1].set_title("Ground Truth (Middle Slice)")
        axes[1, 1].axis("off")

    # Overlayed View
    if ground_truth is not None and set_name != 'Test_Set':
        overlay = fixed_image[:, :, slice_index].copy()
        axes[1, 2].imshow(overlay, cmap="gray")
        axes[1, 2].imshow(ground_truth[:, :, slice_index], cmap="tab10", alpha=0.4)
        axes[1, 2].set_title("Overlay of Fixed Image and Ground Truth")
        axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

    # Printing statistics
    print("Fixed Image Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    if label_distribution is not None:
        print("\nGround Truth Label Distribution:")
        for label, count in label_distribution.items():
            print(f"Label {label}: {count}")

    return stats