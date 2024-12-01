# MISA-MultiAtlas-Segmentation

# Multi-Atlas Segmentation with Joint Label Fusion

This repository contains the implementation of Multi-Atlas Segmentation using the Joint Label Fusion method. The goal is to segment biomedical images by leveraging multiple labeled atlases, deformable image registration, and a novel label fusion technique.

## **Project Description**

Multi-Atlas Segmentation is an advanced technique for labeling regions in biomedical images by utilizing multiple expert-labeled datasets (atlases). This method compensates for registration errors by integrating segmentation results through weighted voting. Our implementation builds on the Joint Label Fusion method, which models dependencies between atlases to improve segmentation accuracy.

### **Features**
- **Deformable Registration**: Align multiple atlases to the target image.
- **Weighted Voting**: Combine atlas labels based on intensity similarity.
- **Joint Label Fusion**: Account for pairwise dependencies to optimize label fusion.
- **Evaluation Metrics**: Dice Similarity Coefficient (DSC), Hausdorff Distance (HD), etc.

## **File Structure**

- `data/`: Dataset storage (raw and processed).
- `notebooks/`: Jupyter Notebooks for data exploration and experiments.
- `src/`: Core implementation of segmentation and label fusion.
  - `atlas/`: Atlas creation and registration scripts.
  - `segmentation/`: Segmentation and label fusion logic.
  - `evaluation/`: Evaluation scripts for segmentation performance.
  - `utils/`: Helper functions.
- `tests/`: Unit tests for core functions.
- `docs/`: Documentation and project reports.
- `scripts/`: Shell scripts for automation.

## **Requirements**
- Python 3.8 or higher
- Libraries: `numpy`, `scipy`, `nibabel`, `SimpleITK`, `matplotlib`, `pandas`
- Registration tools: `ANTsPy` or `Elastix`

## **Usage**

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/MultiAtlasSegmentation.git
   cd MultiAtlasSegmentation