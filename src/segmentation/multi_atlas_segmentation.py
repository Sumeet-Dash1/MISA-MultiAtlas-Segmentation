import numpy as np
from utils.helpers import compute_dice_coefficient

def label_fusion(votes, weights):
    """Perform weighted label fusion."""
    return np.argmax(np.sum(votes * weights[:, None, None, None], axis=0), axis=0)

def segment_image(target_image, atlases, atlas_labels, weights):
    """Segment the target image using multi-atlas approach."""
    votes = np.array([atlas_labels[i] for i in range(len(atlases))])
    return label_fusion(votes, weights)