import numpy as np


def majority_vote(labels):
    """
    Perform majority voting on a list of label arrays.
    Parameters:
        labels (list of np.ndarray): List of label arrays.
    Returns:
        np.ndarray: Majority-voted label map.
    """
    # Stack label maps into a single 4D array
    stacked_labels = np.stack(labels, axis=-1)

    # Compute the majority vote along the last axis
    majority_labels = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int)).argmax(), axis=-1, arr=stacked_labels
    )

    return majority_labels


def weighted_vote(labels, weights):
    """
    Perform weighted voting on a list of label arrays.
    Parameters:
        labels (list of np.ndarray): List of label arrays (3D).
        weights (list of float): List of weights corresponding to the labels.
    Returns:
        np.ndarray: Weighted-voted label map (3D).
    """
    if len(labels) != len(weights):
        raise ValueError("Number of labels and weights must be the same.")

    # Stack labels into a 4D array: (X, Y, Z, N), where N is the number of label maps
    stacked_labels = np.stack(labels, axis=-1)  # Shape: (X, Y, Z, N)

    # Get unique classes
    unique_classes = np.unique(stacked_labels)  # Shape: (num_classes,)
    num_classes = len(unique_classes)

    # Initialize weighted sums for each class
    weighted_sums = np.zeros(stacked_labels.shape[:-1] + (num_classes,), dtype=np.float64)  # Shape: (X, Y, Z, num_classes)

    # Compute weighted votes for each class
    for i, cls in enumerate(unique_classes):
        class_votes = (stacked_labels == cls).astype(np.float64)  # Binary map for the current class
        for j, weight in enumerate(weights):
            weighted_sums[..., i] += weight * class_votes[..., j]  # Add weighted votes for the current class

    # Compute the final label based on maximum weighted sum
    final_labels = unique_classes[np.argmax(weighted_sums, axis=-1)]  # Shape: (X, Y, Z)

    return final_labels

def majority_vote_with_probabilities(labels):
    """
    Perform majority voting on a list of label arrays and calculate the probabilities for each class.
    
    Parameters:
        labels (list of np.ndarray): List of label arrays.
        
    Returns:
        tuple:
            - np.ndarray: Majority-voted label map.
            - np.ndarray: Probability map for each class (same shape as input with one extra dimension for classes).
    """
    # Stack label maps into a single 4D array (last dimension for ensemble)
    stacked_labels = np.stack(labels, axis=-1)

    # Get the number of classes from the unique labels
    unique_classes = np.unique(stacked_labels)
    num_classes = len(unique_classes)

    # Compute probabilities for each class along the last axis
    probabilities = np.zeros(stacked_labels.shape[:-1] + (num_classes,), dtype=np.float32)
    for i, cls in enumerate(unique_classes):
        probabilities[..., i] = np.mean(stacked_labels == cls, axis=-1)

    # Majority voting: Choose the class with the highest probability
    majority_labels = np.argmax(probabilities, axis=-1)

    return majority_labels, probabilities


def staple_fusion(labels, max_iterations=10, tolerance=1e-5):
    """
    Perform STAPLE fusion on a list of label arrays.
    Parameters:
        labels (list of np.ndarray): List of label arrays.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence threshold.
    Returns:
        np.ndarray: STAPLE-fused label map.
    """
    labels = np.stack(labels, axis=-1)  # Stack labels into a 4D array
    n_voxels = labels.shape[:-1]
    n_labels = labels.shape[-1]

    # Initialize probabilities and sensitivities
    p = np.ones(n_voxels) * 0.5  # Prior probabilities
    sensitivities = np.ones(n_labels) * 0.8  # Sensitivity of each labeler
    specificities = np.ones(n_labels) * 0.8  # Specificity of each labeler

    for iteration in range(max_iterations):
        # E-step: Compute weights
        weights = np.zeros_like(labels, dtype=np.float64)
        for i in range(n_labels):
            weights[..., i] = sensitivities[i] * labels[..., i] + \
                              specificities[i] * (1 - labels[..., i])

        weights /= np.sum(weights, axis=-1, keepdims=True)

        # M-step: Update probabilities, sensitivities, and specificities
        new_p = np.mean(weights, axis=-1)
        new_sensitivities = np.mean(weights * labels, axis=(0, 1, 2)) / np.mean(weights, axis=(0, 1, 2))
        new_specificities = np.mean(weights * (1 - labels), axis=(0, 1, 2)) / np.mean(weights, axis=(0, 1, 2))

        # Check for convergence
        if np.max(np.abs(new_p - p)) < tolerance:
            break

        p = new_p
        sensitivities = new_sensitivities
        specificities = new_specificities

    # Final fused label map
    return (p > 0.5).astype(np.int32)


def probability_fusion(labels):
    """
    Perform probability-based label fusion by selecting the label with the maximum probability.
    Parameters:
        labels (list of np.ndarray): List of 3D label arrays (e.g., (256, 128, 256)).
    Returns:
        np.ndarray: Fused label map (3D array).
    """
    # Stack labels into a single 4D array: (x, y, z, n_labels)
    stacked_labels = np.stack(labels, axis=-1)  # Shape: (x, y, z, n_labels)

    # Get unique labels from all arrays
    unique_labels = np.unique(stacked_labels)  # Assume integer labels

    # Initialize a probability array: (x, y, z, n_unique_labels)
    probabilities = np.zeros(stacked_labels.shape[:-1] + (len(unique_labels),), dtype=np.float64)

    # Compute probabilities for each unique label
    for i, label in enumerate(unique_labels):
        probabilities[..., i] = np.mean(stacked_labels == label, axis=-1)

    # Select the label with the maximum probability for each voxel
    fused_label = unique_labels[np.argmax(probabilities, axis=-1)]  # Shape: (x, y, z)

    return fused_label