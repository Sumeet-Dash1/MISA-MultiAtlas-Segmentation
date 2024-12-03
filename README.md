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

```
MISA-MultiAtlas-Segmentation/
├── data/
│   ├── raw/                     # Original raw data
│   ├── processed/               # Registered images and TransformParameters
│   ├── transformed_labels/      # Transformed labels for each fixed image
│   └── window_based/            # Intermediate and final results from window-based segmentation
│       └── IBSR_11_fused.nii.gz # Example fused output for IBSR_11
├── notebooks/                   # Jupyter notebooks for visualization and analysis
├── src/                         # Source code for functionalities
│   ├── atlas/                   # Atlas-related code
│   ├── segmentation/
│   │   ├── fusion_methods.py         # Fusion strategies (majority voting, weighted voting, etc.)
│   │   ├── label_fusion_pipeline.py  # Label fusion pipeline logic
│   │   └── window_segmentation.py    # Window-based segmentation pipeline
│   ├── evaluation/
│   │   └── metrics.py               # Evaluation metrics (Dice, Hausdorff, etc.)
│   └── utils/
│       ├── helpers.py               # General utilities (loading, saving, etc.)
│       └── sliding_window.py        # Logic for window extraction and merging
├── tests/                       # Unit tests for core functions
│   ├── test_fusion_methods.py       # Tests for fusion methods
│   ├── test_sliding_window.py       # Tests for sliding window logic
│   └── test_window_segmentation.py  # Tests for window segmentation pipeline
├── scripts/                     # Automation scripts
│   ├── run_elastix.sh                  # Registration script
│   ├── transform_labels.sh             # Transform labels using transformix
│   └── window_segmentation_runner.py   # End-to-end script for window-based segmentation
├── README.md                    # Project description and instructions
├── requirements.txt             # Project dependencies
└── .gitignore                   # Ignored files/folders
```

## **Requirements**
- Python 3.8 or higher
- Libraries: `numpy`, `scipy`, `nibabel`, `SimpleITK`, `matplotlib`, `pandas`
- Registration tools: `ANTsPy` or `Elastix`

## **Usage**

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/MultiAtlasSegmentation.git
   cd MultiAtlasSegmentation