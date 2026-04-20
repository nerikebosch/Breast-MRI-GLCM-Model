# Breast Cancer MRI Texture Analysis & AI Prediction

This project uses **Radiomics** (texture analysis) and **Machine Learning** to predict the **Estrogen Receptor (ER) status** of breast cancer tumors from MRI scans. By analyzing the "chaos" or heterogeneity in tumor textures, the AI provides decision support for molecular classification.

## ⚠️ Important Note on Dataset
The `Processed NIFTI Dataset/` directory is **not included** in this repository due to size. To train the model or run predictions:
1.  You must provide your own dataset of NIfTI/Analyze images.
2.  Each patient should have a post-contrast MRI (`post_1.img.gz`) and a tumor segmentation mask (`segmentation.img.gz`).
3.  Place them in a structure like: `Processed NIFTI Dataset/Patient_ID/`.

---

## 🚀 How It Works

### 1. Feature Extraction (GLCM)
The system automatically identifies the 3D slice with the largest tumor area. It then extracts **Gray-Level Co-occurrence Matrix (GLCM)** features using `scikit-image`:
- **Contrast**: Measures local variations.
- **Correlation**: Measures joint probability occurrence.
- **Energy**: Measures image homogeneity.
- **Homogeneity**: Measures the closeness of the distribution of elements in the GLCM.

### 2. AI Training
A **Random Forest Classifier** is trained using the extracted texture features as inputs and the molecular ER status (from the clinical Excel file) as the target label.

### 3. Reusable Prediction
The trained model is exported using `joblib`. This allows researchers to use a standalone script to predict ER status for new patients without retraining the entire system.

---

## 🛠️ Installation & Setup

1. **Clone the project** and navigate to the directory.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install joblib
   ```
3. **Prepare Clinical Data**: Ensure `Clinical_and_Other_Features.xlsx` is in the root directory if you plan to train.

---

## 📖 Usage Guide

### Step 1: Train the AI
Open and run all cells in `texture_analysis_code.ipynb`. 
- This will generate `models/rf_model.joblib` and `models/feature_columns.joblib`.
- It also generates a `my_glcm_textures.csv` containing all extracted features.

### Step 2: Predict New Data
Use the `predict.py` script to analyze a specific patient. You can also generate visualizations to see where the AI is looking.

**Run a simple prediction:**
```bash
python predict.py --image "path/to/post_1.img.gz" --mask "path/to/segmentation.img.gz"
```

**Run prediction with Visualizations (Heatmaps):**
```bash
python predict.py --image "path/to/post_1.img.gz" --mask "path/to/segmentation.img.gz" --visualize
```

---

## 📂 Project Structure
- `texture_analysis_code.ipynb`: The main research notebook for feature extraction and training.
- `predict.py`: Standalone CLI tool for reusing the model.
- `models/`: Contains the serialized AI model files.
- `instructions.md`: Detailed technical usage for the scripts.
- `requirements.txt`: List of necessary Python libraries.

---

## 🔬 Visualizations Included
When running with the `--visualize` flag, the project generates:
1. **Tumor Segmentation Overlay**: Highlights the tumor in red over the raw MRI.
2. **Texture Heatmap (Entropy)**: A color-coded map showing areas of high heterogeneity ("chaos") within the tumor, which often correlates with malignancy.
