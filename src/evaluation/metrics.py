import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance
from scipy.spatial.distance import directed_hausdorff

def calculate_dice_score(img1_data, img2_data, ignore_background=True):
    """
    Calculate the Dice score between two NIfTI images.
    
    Parameters:
    nii_image1(np.ndarray:): first NIfTI image (e.g., ground truth).
    nii_image2 (np.ndarray:): second NIfTI image (e.g., predicted labels).
    ignore_background (bool): Whether to ignore the background label (default: True).
    
    Returns:
    dice_scores (dict): Dictionary of Dice scores for each label present in the images.
    """    
    # Ensure the two images have the same shape
    if img1_data.shape != img2_data.shape:
        raise ValueError("The two NIfTI images must have the same shape.")
    
    # Calculate the Dice score for each label
    dice_scores = {}
    labels = np.unique(img1_data)  # Find all unique labels in the first image
    
    for label in labels:
        if label == 0 and ignore_background:
            continue  # Skip background (assuming label 0 is background)
        
        # Create binary masks for the current label in both images
        img1_mask = (img1_data == label)
        img2_mask = (img2_data == label)
        
        # Calculate the intersection and union
        intersection = np.sum(img1_mask & img2_mask)
        union = np.sum(img1_mask) + np.sum(img2_mask)
        
        # Calculate Dice score (handle division by zero)
        if union == 0:
            dice_score = 1.0  # Perfect match if both masks are empty
        else:
            dice_score = (2. * intersection) / union
        
        # Store the Dice score for the current label
        dice_scores[label] = dice_score
    
    return dice_scores

def calculate_hausdorff_distance(prediction, ground_truth, num_classes, include_background=False):
    """
    Compute the Hausdorff Distance for each class.

    Args:
        prediction (np.ndarray): Predicted segmentation (shape: [Z, Y, X]).
        ground_truth (np.ndarray): Ground truth segmentation (shape: [Z, Y, X]).
        num_classes (int): Number of classes.
        include_background (bool): Whether to include background class in the computation.

    Returns:
        dict: Hausdorff distance for each class.
    """
    hausdorff_distances = {}

    for class_id in range(num_classes):
        if not include_background and class_id == 0:
            continue

        # Extract binary masks for the current class
        pred_mask = (prediction == class_id)
        gt_mask = (ground_truth == class_id)

        # Skip if either mask is empty
        if not np.any(pred_mask) or not np.any(gt_mask):
            hausdorff_distances[class_id] = np.nan  # Assign NaN for missing classes
            continue

        # Convert binary masks to point clouds
        pred_points = np.argwhere(pred_mask)
        gt_points = np.argwhere(gt_mask)

        # Calculate directed Hausdorff distances
        forward_hd = directed_hausdorff(pred_points, gt_points)[0]
        backward_hd = directed_hausdorff(gt_points, pred_points)[0]

        # Use the maximum of the forward and backward distances
        hausdorff_distances[class_id] = max(forward_hd, backward_hd)

    return hausdorff_distances

def calculate_average_volumetric_difference(prediction, ground_truth, num_classes, include_background=False):
    """
    Compute the Average Volumetric Difference (AVD) for each class.

    Args:
        prediction (np.ndarray): Predicted segmentation (shape: [Z, Y, X]).
        ground_truth (np.ndarray): Ground truth segmentation (shape: [Z, Y, X]).
        num_classes (int): Number of classes.
        include_background (bool): Whether to include background class in the computation.

    Returns:
        dict: AVD for each class.
    """
    avd_scores = {}

    for class_id in range(num_classes):
        if not include_background and class_id == 0:
            continue

        # Calculate volume for the current class
        pred_volume = np.sum(prediction == class_id)
        gt_volume = np.sum(ground_truth == class_id)

        # Skip if ground truth volume is zero
        if gt_volume == 0:
            avd_scores[class_id] = np.nan  # Assign NaN for missing classes
            continue

        # Compute AVD
        avd_scores[class_id] = abs(pred_volume - gt_volume) / (gt_volume + 1e-6)

    return avd_scores