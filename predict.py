import os
import argparse
import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk

def extract_glcm_features(image_path, mask_path):
    """
    Extracts GLCM features from the 3D MRI slice with the largest tumor area.
    """
    try:
        img_data = nib.load(image_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata()

        slice_areas = np.sum(mask_data, axis=(0, 1))
        best_slice_idx = np.argmax(slice_areas)

        img_slice = img_data[:, :, best_slice_idx]
        mask_slice = mask_data[:, :, best_slice_idx]

        tumor_only = img_slice * mask_slice
        tumor_normalized = np.interp(tumor_only, (tumor_only.min(), tumor_only.max()), (0, 255)).astype('uint8')

        glcm = graycomatrix(tumor_normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

        features = {
            'GLCM_Contrast': graycoprops(glcm, 'contrast')[0, 0],
            'GLCM_Correlation': graycoprops(glcm, 'correlation')[0, 0],
            'GLCM_Energy': graycoprops(glcm, 'energy')[0, 0],
            'GLCM_Homogeneity': graycoprops(glcm, 'homogeneity')[0, 0]
        }
        return features, best_slice_idx

    except Exception as e:
        print(f"  [!] Error processing features: {e}")
        return None, None

def visualize_tumor(image_path, mask_path, best_slice_idx):
    print(f"\nGenerating Tumor Overlay Visual...")
    img_data = nib.load(image_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()

    img_slice = img_data[:, :, best_slice_idx]
    mask_slice = mask_data[:, :, best_slice_idx]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_slice, cmap='gray')
    plt.title(f'Raw MRI (Slice {best_slice_idx})')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_slice, cmap='gray')
    masked_tumor = np.ma.masked_where(mask_slice == 0, mask_slice)
    plt.imshow(masked_tumor, cmap='autumn', alpha=0.6)
    plt.title('AI Tumor Segmentation Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_texture_heatmap(image_path, mask_path, best_slice_idx):
    print(f"\nGenerating Texture Heatmap (Entropy)...")
    img_data = nib.load(image_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()

    img_slice = img_data[:, :, best_slice_idx]
    mask_slice = mask_data[:, :, best_slice_idx]

    img_normalized = np.interp(img_slice, (img_slice.min(), img_slice.max()), (0, 255)).astype('uint8')
    texture_map = entropy(img_normalized, disk(3))
    masked_texture = np.ma.masked_where(mask_slice == 0, texture_map)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_slice, cmap='gray')
    coords = np.argwhere(mask_slice)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        plt.ylim(y1 + 30, y0 - 30)
        plt.xlim(x0 - 30, x1 + 30)
    plt.title('Zoomed-in MRI')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_slice, cmap='gray')
    heat = plt.imshow(masked_texture, cmap='jet', alpha=0.7)
    plt.colorbar(heat, fraction=0.046, pad=0.04, label="Texture Heterogeneity (Entropy)")
    if len(coords) > 0:
        plt.ylim(y1 + 30, y0 - 30)
        plt.xlim(x0 - 30, x1 + 30)
    plt.title('Tumor Texture Heatmap')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def predict_er_status(image_path, mask_path, model_path, columns_path, visualize=False):
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        print(f"[!] Error: Model files not found.")
        return

    print(f"[*] Extracting features...")
    features, best_slice_idx = extract_glcm_features(image_path, mask_path)
    if features is None: return

    print("[*] Loading model and predicting...")
    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
    X = pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(X)[0]
    
    print("\n" + "="*30)
    print(f"PREDICTION RESULT")
    print("="*30)
    print(f"Patient ID: {os.path.basename(os.path.dirname(image_path))}")
    print(f"Predicted ER Status: {prediction}")
    print("="*30)

    if visualize:
        visualize_tumor(image_path, mask_path, best_slice_idx)
        visualize_texture_heatmap(image_path, mask_path, best_slice_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ER Status from Breast MRI with Visualizations.")
    parser.add_argument("--image", required=True, help="Path to MRI (.img.gz)")
    parser.add_argument("--mask", required=True, help="Path to mask (.img.gz)")
    parser.add_argument("--visualize", action="store_true", help="Show visualizations")
    parser.add_argument("--model", default="models/rf_model.joblib")
    parser.add_argument("--columns", default="models/feature_columns.joblib")

    args = parser.parse_args()
    predict_er_status(args.image, args.mask, args.model, args.columns, args.visualize)
