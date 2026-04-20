# Breast MRI Texture Analysis - Instructions

This project provides a pipeline to extract GLCM (texture) features from breast MRI scans, train an AI model to predict Estrogen Receptor (ER) status, and reuse the trained model for future predictions.

## 1. Prerequisites
Ensure you have the required Python libraries installed:
```bash
pip install -r requirements.txt
pip install joblib
```

## 2. Training the Model
To train the model and save it for future use, follow these steps:
1.  Open the `texture_analysis_code.ipynb` Jupyter Notebook.
2.  Run all cells.
3.  The notebook will:
    - Extract GLCM features from the `Processed NIFTI Dataset/`.
    - Load clinical data from `Clinical_and_Other_Features.xlsx`.
    - Train a `RandomForestClassifier`.
    - Save the model to `models/rf_model.joblib`.
    - Save the feature column order to `models/feature_columns.joblib`.

**Note:** You must have the Excel file `Clinical_and_Other_Features.xlsx` in the root directory for training to work.

## 3. Making Predictions
Once the model is trained, you can use the `predict.py` script to predict the ER status of a new patient's MRI.

### Usage:
```bash
python predict.py --image "path/to/post_1.img.gz" --mask "path/to/segmentation.img.gz" --visualize
```

### Example:
```bash
python predict.py --image "Processed NIFTI Dataset/Breast_MRI_001/post_1.img.gz" --mask "Processed NIFTI Dataset/Breast_MRI_001/segmentation.img.gz" --visualize
```

### Optional Arguments:
- `--visualize`: Show the Tumor Overlay and Texture Heatmap plots.
- `--columns`: Path to a custom feature columns file (default: `models/feature_columns.joblib`)

## 4. File Structure
- `texture_analysis_code.ipynb`: Training and exploration notebook.
- `predict.py`: Standalone prediction script.
- `models/`: Folder containing saved AI models.
- `my_glcm_textures.csv`: Extracted features for the dataset.
- `instructions.md`: This file.
