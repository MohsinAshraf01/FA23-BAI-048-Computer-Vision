# NEU Steel Surface Defect Classification and HOG Inspection

A computer vision and machine learning project for **steel surface defect inspection** using the **NEU steel surface defect dataset**. The notebook studies Histogram of Oriented Gradients (HOG) feature extraction, compares several classical machine-learning classifiers, evaluates robustness to image perturbations, and builds an end-to-end screening system for detecting defective regions and classifying the defect type.

## Overview

The project has two main machine-learning tracks:

- **Track A — Six-class defect classification:** classify a steel-surface image into one of six defect categories.
- **Track B — Normal vs Defective inspection:** classify local image windows as either normal or defective, then scan a complete image to decide whether the product should be accepted or rejected.

The implementation is designed as an experimental notebook and includes dataset inspection, preprocessing, HOG parameter studies, classifier comparison, hyperparameter tuning, robustness testing, visualization, checkpointing, and final end-to-end evaluation.

## Defect Classes

The notebook uses six classes:

1. `crazing`
2. `inclusion`
3. `patches`
4. `pitted_surface`
5. `rolled-in_scale`
6. `scratches`

## Dataset

The notebook downloads the dataset using `kagglehub`:

```text
sovitrath/neu-steel-surface-defect-detect-trainvalid-split
```

The downloaded dataset contains:

```text
train_images/
train_annotations/
valid_images/
valid_annotations/
```

The notebook works with image annotations containing defect bounding boxes and uses those annotations when constructing the defective/normal window dataset.

### Dataset split

The final run contains:

| Split | Images |
|---|---:|
| Train | 1,260 |
| Validation | 270 |
| Test | 270 |
| **Total** | **1,800** |

The six-class image classification task uses the image-level labels, while the binary inspection task additionally uses annotated defect boxes to construct local windows.

## Methodology

### 1. Image preprocessing

For the main image-classification track:

- Images are converted to grayscale.
- Images are resized to **128 × 128** pixels.
- A fixed random seed of **42** is used for reproducibility.

The notebook also keeps native grayscale images for the inspection pipeline.

### 2. HOG feature extraction

The project uses **Histogram of Oriented Gradients (HOG)** to represent local shape and texture information.

The HOG study evaluates:

- Cell sizes: `4 × 4`, `8 × 8`, `16 × 16`
- Orientation bins: `6`, `9`, `12`
- Block size: `2 × 2` cells
- Block normalization: `L2-Hys`

The notebook measures both classification performance and feature-vector dimensionality.

### 3. Feature parsimony

The notebook does not simply select the configuration with the highest score. It also considers feature-vector size.

A **parsimony tolerance of 0.005** is used to identify smaller feature representations that remain within 0.5 percentage points of the best cross-validation macro-F1 score.

This provides a practical trade-off between:

- predictive performance
- computational cost
- feature dimensionality

## Machine Learning Models

The notebook compares several classical classifiers:

- **SVM with RBF kernel**
- **Linear SVM**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Random Forest**

For the main SVM experiments, hyperparameters such as `C` and the RBF gamma multiplier are tuned using cross-validation.

The configured search spaces include:

```text
SVM C:             [1, 10, 100, 1000]
SVM gamma factor:  [0.25, 0.5, 1.0, 2.0, 4.0]

Binary SVM C:             [1, 10, 100]
Binary SVM gamma factor:  [0.5, 1.0, 2.0]

Linear SVM C:      [0.1, 1, 10]
KNN k:             [1, 3, 5, 7, 9]
Logistic C:        [1, 10, 100]
Random Forest trees: 300
```

## Track A — Six-Class Defect Classification

Track A predicts the defect category directly from the complete steel-surface image.

The best HOG configuration recorded in the notebook was:

| Parameter | Value |
|---|---:|
| Cell size | `16 × 16` |
| Orientation bins | `12` |
| HOG dimensions | 2,352 |
| Mean CV macro-F1 | 0.9223 |

Using the tuned **SVM (RBF)** model on the held-out test set produced:

| Metric | Score |
|---|---:|
| Accuracy | **92.96%** |
| Precision | **92.99%** |
| Recall | **92.96%** |
| F1-score | **92.89%** |

The notebook also generates confusion matrices and visualizations of classification errors.

## Track B — Normal vs Defective Windows

Track B converts the image-level inspection problem into a binary local-window classification problem.

Each `64 × 64` image window is assigned:

- `0` → Normal
- `1` → Defective

Defective windows are selected around annotated defect regions.

Normal windows are selected from areas that do not intersect the annotated defect boxes. The notebook uses:

- Window size: `64 × 64`
- Search stride for normal windows: `8`
- Bounding-box margin: `8` pixels
- Maximum defective windows per image: `2`
- Maximum normal windows per image: `3`
- Minimum defect-window gap: `16`
- Minimum normal-window gap: `48`

The resulting dataset contains:

| Window type | Count |
|---|---:|
| Normal | 2,624 |
| Defective | 3,117 |
| **Total** | **5,741** |

The final split contains:

| Split | Normal | Defective |
|---|---:|---:|
| Train | 1,818 | 2,189 |
| Validation | 423 | 452 |
| Test | 383 | 476 |

## Binary HOG Study

The best recorded Track B HOG result was:

- Cell size: `16 × 16`
- Orientation bins: `12`
- HOG dimensionality: `216`
- Best mean CV macro-F1: **0.7784**

The summary also records a nearby configuration using 6 orientation bins with a mean F1 of approximately `0.7741`.

## End-to-End Inspection

The notebook combines the binary window classifier with the six-class defect classifier to simulate a complete inspection process.

For each `200 × 200` test image:

1. The image is divided into overlapping local windows.
2. Each window receives a probability of being defective.
3. The most suspicious window is used to make the accept/reject decision.
4. If the image is rejected, the defect classifier predicts the defect category.
5. A threshold controls the final screening decision.

The configured decision threshold is:

```text
P(defect) >= 0.5  →  REJECT
P(defect) <  0.5  →  ACCEPT
```

### End-to-end test results

On the held-out test images:

| Measure | Result |
|---|---:|
| Defective test images | 270 |
| Defective images detected | **270 / 270** |
| Detection rate | **100.00%** |
| Missed defective images | **0** |
| Correct defect type among rejected images | **92.96%** |
| Normal test windows accepted | **75.46%** |
| Decision time | **~50.9 ms per 200 × 200 image** |

The recorded detection rate was 100% for each of the six defect types in this test run.

> **Important:** The 100% detection result is specific to the held-out test experiment implemented in this notebook. It should not be interpreted as a guarantee of real-world production performance.

## Threshold Analysis

The notebook evaluates how the screening threshold changes detection and false-rejection behavior.

| Threshold | Defective image detection | Normal-window false reject |
|---:|---:|---:|
| 0.1 | 100.00% | 75.72% |
| 0.3 | 100.00% | 47.52% |
| 0.5 | 100.00% | 24.54% |
| 0.7 | 97.78% | 10.70% |
| 0.9 | 78.15% | 1.31% |

This demonstrates the expected **precision/recall-style trade-off**: increasing the rejection threshold reduces false rejection of normal regions but eventually causes defective images to be missed.

## Robustness Testing

The notebook evaluates model sensitivity to image perturbations.

### Brightness

Brightness gains tested:

```text
0.5, 0.75, 1.25, 1.5
```

### Noise

Gaussian noise levels:

```text
σ = 5, 10, 20, 40
```

### Rotation

Rotation angles:

```text
5°, 15°, 45°, 90°
```

### Blur

Gaussian blur levels:

```text
σ = 1, 2, 3, 5
```

The recorded worst-case changes for the six-class SVM-RBF model were:

| Condition | Worst case | Δ F1 | Δ Accuracy |
|---|---|---:|---:|
| Brightness | Gain ×1.5 | -25.18 | -24.81 |
| Noise | σ = 40 | -86.22 | -75.19 |
| Rotation | 90° | -55.66 | -53.70 |
| Blur | σ = 5 | -88.13 | -76.30 |

For the binary SVM-RBF model:

| Condition | Worst case | Δ F1 | Δ Accuracy |
|---|---|---:|---:|
| Brightness | Gain ×0.5 | -6.03 | -4.54 |
| Noise | σ = 40 | -59.48 | -32.01 |
| Rotation | 90° | -7.26 | -8.73 |
| Blur | σ = 2 | -8.54 | -22.70 |

These experiments show that the six-class classifier can be highly sensitive to severe noise and blur, while the binary screening task is comparatively more tolerant under several tested perturbations.

## Visualizations

The notebook generates visual outputs for:

- Class distribution and annotation coverage
- Sample images with defect bounding boxes
- Resized/grayscale preprocessing
- HOG representations for each defect class
- HOG cell-size/orientation comparisons
- HOG parameter heatmaps
- Classifier comparison
- Confusion matrices
- Classification errors
- Binary ROC curves
- Normal/defective window examples
- Window placement
- Robustness experiments
- Screening-score distributions
- Window-scanning heatmaps

## Checkpointing and Reproducibility

The notebook contains a checkpoint system so long-running experiments can be resumed.

A run ID is generated from the configuration, and results are stored in a run-specific directory.

The default Google Drive location is:

```text
/content/drive/MyDrive/Lab05_NEU_HOG/run_<run_id>/
```

If Google Drive is unavailable, the notebook falls back to local storage.

The checkpoint system stores:

- JSON experiment metadata
- CSV result tables
- trained models
- generated figures
- summary results

Atomic file writes are used for important checkpoint files to reduce the risk of incomplete files after runtime interruptions.

## Saved Model Artifacts

The final notebook run saves models including:

```text
models/
├── 6class__logistic_regression.joblib
├── 6class__random_forest.joblib
├── 6class__svm_linear.joblib
├── 6class__svm_rbf.joblib
├── binary__logistic_regression.joblib
├── binary__random_forest.joblib
├── binary__svm_linear.joblib
├── binary__svm_rbf.joblib
└── inspection_config.json
```

It also saves result files such as:

```text
comparison_results.csv
robustness_results.csv
sweep_results.csv
tuning_results.csv
results_summary.json
extras.json
```

## Requirements

The notebook uses Python and the following major packages:

- NumPy
- Pandas
- Matplotlib
- Pillow
- OpenCV
- scikit-image
- scikit-learn
- Joblib
- KaggleHub

The notebook is configured for **Google Colab** and can use a **T4 GPU**, although the main classical ML/HOG pipeline is not a deep neural-network training pipeline.

## How to Run

### Option 1 — Google Colab

1. Open the notebook in Google Colab.
2. Run the setup cells.
3. Allow Google Drive access when prompted.
4. Run the cells from top to bottom.
5. The dataset is downloaded through `kagglehub`.
6. Results and checkpoints are saved under the configured Drive folder.

### Option 2 — Local Jupyter Environment

Install the required Python packages and run the notebook in Jupyter.

Some notebook cells are specifically written for Google Colab, especially:

```python
from google.colab import drive
drive.mount("/content/drive")
```

For a local environment, the Drive-specific persistence section may need to be adapted.

## Project Structure

Conceptually, the notebook follows this pipeline:

```text
NEU Dataset
    │
    ├── Images
    └── XML Annotations
            │
            ▼
     Dataset Inspection
            │
            ▼
   Grayscale + Resize
            │
            ▼
       HOG Features
            │
       ┌────┴────┐
       │         │
       ▼         ▼
   Track A    Track B
   6-Class    Binary
 Classification Inspection
       │         │
       │         ▼
       │    Window Scanning
       │         │
       └────┬────┘
            ▼
     Model Evaluation
            │
            ▼
   Robustness Testing
            │
            ▼
    End-to-End Inspection
```

## Main Configuration

The notebook's primary configuration includes:

```python
SEED = 42
IMG_SIZE = 128

CELL_SIZES = [4, 8, 16]
ORIENTATIONS = [6, 9, 12]

HOG_BLOCK = 2
HOG_BLOCK_NORM = "L2-Hys"

CV_FOLDS = 5

WIN_SIZE = 64
SCAN_STRIDE = 34

REJECT_THRESHOLD = 0.5
```

The exact values are controlled through the `CONFIG` dictionary in the notebook.

## Key Findings

The experiment demonstrates several important observations:

1. **HOG can represent steel-surface defect patterns effectively** for classical machine-learning models.
2. The best recorded six-class HOG configuration used **16 × 16 cells and 12 orientation bins**.
3. The tuned **RBF SVM achieved 92.96% test accuracy** on the six-class task.
4. The binary window classifier achieved **78.11% test accuracy** with an AUC of **87.13%**.
5. The end-to-end screening experiment detected all 270 defective test images at the selected threshold of 0.5.
6. The defect-type classifier correctly identified the defect type for approximately **92.96%** of rejected defective images.
7. Robustness testing shows that severe blur and noise can substantially reduce classification performance.
8. The screening threshold strongly affects the trade-off between detecting defects and falsely rejecting normal regions.

## Limitations

The results should be interpreted within the experimental setup of the notebook.

- The evaluation uses a specific dataset and split.
- The end-to-end test set contains defective images for the detection experiment, while normal performance is evaluated using normal test windows.
- Robustness transformations are synthetic perturbations and may not represent every real manufacturing condition.
- Classical HOG features may be sensitive to substantial changes in image appearance, blur, noise, and orientation.
- The reported latency depends on the runtime hardware and software environment.
- High test performance on this dataset does not automatically imply equivalent performance on unseen industrial production data.

## Technologies

**Python · OpenCV · scikit-image · scikit-learn · NumPy · Pandas · Matplotlib · Joblib · KaggleHub · Google Colab**

## Project Purpose

The overall goal is to investigate whether a lightweight, interpretable computer-vision pipeline based on **HOG features and classical machine learning** can support automated steel-surface inspection.

The project combines:

- image preprocessing
- handcrafted feature extraction
- supervised classification
- local defect detection
- defect-type recognition
- robustness analysis
- quantitative evaluation
- end-to-end inspection

