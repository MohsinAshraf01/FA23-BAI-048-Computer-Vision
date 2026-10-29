# HOG vs SIFT on Blood-Cell Images

Computer Vision Lab Mid Exam | Registration number: 048

A small comparison of two classical image-classification pipelines, **HOG + SVM** and **SIFT + SVM**, on four types of blood cell. The goal is to find out which pipeline is more reliable when images are rotated or noisy.

## Files

| File | Description |
|---|---|
| `CV_Lab_Mid_Exam_Reg048_Final.ipynb` | Final notebook with code, outputs and written answers for Q1 to Q5 |
| `README.md` | This file |

## Personal settings (from registration number 048)

| Setting | Value | Rule |
|---|---|---|
| Random seed | 48 | Last two digits |
| HOG cell size | 8 x 8 | Last digit 8 is even |
| SIFT grid | 6 x 6 (36 keypoints) | Last digit 8 is in 5 to 9 |

## Dataset and fixed settings

- **Dataset:** BloodMNIST (`medmnist`), image size 64
- **Classes:** eosinophil (1), monocyte (5), neutrophil (6), platelet (7)
- **Images per class:** 100 train, 30 validation, 30 test, sampled with seed 48
- **Preparation:** grayscale, resized to 128 x 128
- **HOG:** 9 orientations, 2 x 2 cells per block
- **Classifier:** `LinearSVC`, C chosen from {0.1, 1, 10} on the validation set
- **Robustness tests:** rotation by 20 degrees, Gaussian noise with sigma = 0.05 (image scaled to 0 to 1)

## Requirements

- Python 3
- `medmnist` (version 3.0 or later, needed for `size=64`)
- `opencv-python`, `scikit-image`, `scikit-learn`, `matplotlib`, `pandas`, `numpy`

The first code cell installs everything:

```
pip install medmnist opencv-python scikit-image scikit-learn matplotlib pandas
```

## How to run

1. Open `CV_Lab_Mid_Exam_Reg048_Final.ipynb` in Jupyter or Google Colab.
2. Run all cells from top to bottom. The dataset downloads automatically on the first run.
3. Nothing needs to be typed in. The registration number is set in the first code cell (`REG_NO = "048"`) and the other settings follow from it.

The test set is used only in Q4. The SVM models are trained once and frozen before the test evaluation.

## What the notebook contains

| Section | Content |
|---|---|
| Q1 | Data loading, split shapes and class counts, example image grid |
| Q2 | HOG features (length 8100, calculated by hand), HOG pictures, SVM with C chosen on validation |
| Q3 | Normal SIFT keypoint statistics, fixed 6 x 6 grid SIFT (length 4608 = 36 x 128), SVM with C chosen on validation |
| Q4 | Prediction written before testing, then clean / rotated / noisy test accuracy for both models |
| Q5 | Short answers on block normalisation and choosing C, most confused classes, AI log |

## Results

**Validation (used to choose C)**

| Model | Best C | Accuracy | Macro-F1 |
|---|---|---|---|
| HOG + SVM | 0.1 | 0.7167 | 0.7158 |
| SIFT (grid) + SVM | 10 | 0.8583 | 0.8564 |

**Test accuracy under changes**

| Model | Clean | Rotated 20 degrees | Noisy (sigma = 0.05) |
|---|---|---|---|
| HOG + SVM | 0.6917 | 0.5833 | 0.5167 |
| SIFT + SVM | 0.8333 | 0.3333 | 0.8250 |

## Main findings

- **Clean images:** SIFT with a fixed grid is clearly more accurate than HOG (0.8333 vs 0.6917 on the test set).
- **Noise:** SIFT is far more robust. It lost 0.0083 accuracy, while HOG lost 0.1750.
- **Rotation:** SIFT is much less robust here. It lost 0.5000 accuracy, while HOG lost 0.1083. The likely reason is that the fixed-grid SIFT does not estimate orientation, so a turned cell no longer matches the grid points the model learned.
- **Confusions:** eosinophil and neutrophil were confused most (22 errors across both models on the validation set). Platelet was almost never confused. Grayscale removes the red-orange granule colour that helps identify eosinophils.

## Notes

- Normal SIFT detection (11 to 175 keypoints per image) cannot be fed to an SVM directly because the vector length changes from image to image. The fixed grid fixes this by giving every image exactly 36 keypoints.
- SIFT grid descriptors are divided by 255 so the values are between 0 and 1.
- Ties between values of C are broken by keeping the smaller C.
- All numbers in the written answers come from the outputs inside the notebook.
