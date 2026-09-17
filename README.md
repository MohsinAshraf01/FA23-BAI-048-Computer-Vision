# Effect of Image Filtering on Skin-Lesion Classification

## Lab Task 02

This project investigates how different spatial-domain image filters affect pretrained deep-learning models for skin-lesion classification.

## Dataset

The same skin-cancer dataset used in Task 01 is used for this experiment.

Dataset source:
`nodoubttome/skin-cancer9-classesisic`

The dataset contains 9 skin-lesion classes and is automatically downloaded using KaggleHub.

## Models

The three best-performing pretrained models from Task 01 are evaluated:

- EfficientNet-B0
- DenseNet121
- ResNet101

## Image Filtering

Each model is evaluated using six image conditions:

1. No Filter (Baseline)
2. Average / Mean Filter
3. Gaussian Filter
4. Median Filter
5. Sharpening Filter
6. Sobel Edge Filter

Therefore:

**3 Models × 6 Filters = 18 Experiments**

## Evaluation Metrics

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- AUC

## Experimental Settings

| Setting | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 5 |
| Learning Rate | 0.0001 |
| Train/Validation Split | 85% / 15% |
| Optimizer | Adam |
| Loss Function | Cross Entropy Loss |
| Random Seed | 42 |

The same train/validation split and evaluation procedure are used for all experiments.

## Code Organization

The code is organized into the following sections:

1. Dataset Preparation
2. Model Loading
3. Baseline Experiment
4. Image Filtering
5. Training
6. Evaluation
7. Visualization
8. Comparative Analysis

## Results

After running the experiments, the results are saved automatically as:

```text
task02_results/
├── Task_02_Filtering_Comparison.xlsx
├── Task_02_Filtering_Results.csv
└── models/
    ├── efficientnet_b0_no_filter.pth
    ├── efficientnet_b0_average.pth
    ├── ...
    └── resnet101_sobel.pth
```

The Excel file contains:

- Comparison Table
- Summary of the highest Accuracy, Macro-F1, and AUC

## How to Run

Install the required packages:

```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python-headless kagglehub openpyxl matplotlib
```

Run the Python file:

```bash
python Task_02_Skin_Lesion_Filtering.py
```

A GPU is recommended because the project performs 18 deep-learning experiments.

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- KaggleHub

## Author

**Mohsin Ashraf**

Artificial Intelligence | Machine Learning
