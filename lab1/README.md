# FA23-BAI-048-Computer-Vision
# Skin Cancer Classification with Model Comparison

A deep learning project for skin cancer image classification using
transfer learning, deep feature extraction, and multiple machine
learning classifiers.

## Results

### Table 1 --- Transfer Learning Models

---

Model Accuracy (%) Precision (%) Recall (%) F1-Score (%) AUC (%)

---

AlexNet 48.305085 53.041045 48.305085 44.223493 87.141402

VGG16 51.694915 63.776938 51.694915 47.366096 87.621411

VGG19 50.000000 59.720837 50.000000 45.322933 83.401679

ResNet18 55.932203 56.268707 55.932203 51.894714 87.969714

ResNet50 55.084746 56.410360 55.084746 52.556535 87.085121

ResNet101 56.779661 60.270689 56.779661 54.407321 87.103761

DenseNet121 57.627119 62.280431 57.627119 55.557868 88.693196

**EfficientNet-B0** **60.169492** **63.872117** **60.169492** **56.988500** **90.424162**

---

**Best overall model:** EfficientNet-B0 with **60.17% accuracy** and
**90.42% AUC**.

### Table 2 --- Classifier Comparison

---

Feature Classifier Accuracy Precision Recall (%) F1-Score AUC (%)
Extractor (%) (%) (%)

---

Deep Logistic 55.084746 63.066631 55.084746 53.031102 89.453891
Features Regression

Deep Decision 50.000000 53.534192 50.000000 48.822035 71.133845
Features Tree

Deep Random 56.779661 61.513453 56.779661 52.797898 79.759490
Features Forest

Deep KNN 55.084746 53.332642 55.084746 52.325414 84.991944
Features

Deep Linear SVM 55.932203 66.720205 55.932203 53.601097 91.176471
Features

Deep RBF-SVM 59.322034 61.921800 59.322034 56.181498 90.706648
Features

Deep XGBoost 56.779661 61.951404 56.779661 53.010762 89.281359
Features

---

**Best classifier accuracy:** RBF-SVM --- **59.32%**\
**Highest classifier AUC:** Linear SVM --- **91.18%**

### Table 3 --- Computational Efficiency

---

Model Parameters (M) Model Size (MB) FLOPs (G) Inference Time Accuracy (%)
(ms)

---

AlexNet 57.040713 217.593052 0.731048 0.432223 48.305085

VGG16 134.297417 512.303989 15.466255 5.746416 51.694915

VGG19 139.607113 532.558872 19.628054 6.470977 50.000000

ResNet18 11.181129 42.683936 1.823256 1.070550 55.932203

ResNet50 23.526473 89.949413 4.131713 3.665665 55.084746

DenseNet121 6.963081 26.882061 2.895992 3.951768 57.627119

**EfficientNet-B0** **4.019077** **15.492214** **0.413877** **1.515961** **60.169492**

---

**Best balance of accuracy and efficiency:** EfficientNet-B0.

## Project Objectives

- Classify skin-cancer images using deep learning.
- Compare multiple pre-trained CNN models.
- Evaluate Accuracy, Precision, Recall, F1-Score, and AUC.
- Extract deep features and compare traditional classifiers.
- Compare computational efficiency using parameters, size, FLOPs, and
  inference time.
- Automatically download the dataset using KaggleHub.

## Models and Classifiers

**CNN Models:** AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101,
DenseNet121, EfficientNet-B0.

**Classifiers:** Logistic Regression, Decision Tree, Random Forest, KNN,
Linear SVM, RBF-SVM, and XGBoost.

## Dataset

The project uses the Skin Cancer ISIC dataset:

`nodoubttome/skin-cancer9-classesisic`

The dataset is downloaded automatically using KaggleHub.

## Technologies

Python • PyTorch • Torchvision • Scikit-learn • XGBoost • KaggleHub •
THOP • Pandas • OpenPyXL

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

Results are saved in:

```text
results/All_Tables.xlsx
results/classifier_results.csv
results/models/
```

## Project Structure

```text
skin-cancer-classification-with-model-comparison/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── results/
    ├── All_Tables.xlsx
    ├── classifier_results.csv
    └── models/
```

> The dataset is not included in the repository. It is downloaded
> automatically when the program runs.

## Author

**Mohsin Ashraf**

AI / Machine Learning Student

## License

For educational and research purposes.

