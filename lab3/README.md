# Edge Detection and Skin Lesion Classification

This project explores **edge detection techniques, image noise effects, image preprocessing, and skin lesion classification** using the **HAM10000 dataset**.

The project compares traditional computer vision methods with classical machine learning and deep learning models to understand how different image processing conditions affect classification performance.

## 📌 Project Overview

The main objective of this project is to study how edge detection and image preprocessing techniques influence the analysis and classification of skin lesion images.

The notebook covers:

- Comparative edge detection
- Noise analysis
- Canny edge detector parameter analysis
- Image preprocessing
- CNN-based classification
- Classical machine learning classification
- Performance comparison
- Confusion matrices and evaluation metrics

## 📂 Dataset

The project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset.

The dataset contains dermatoscopic images of skin lesions belonging to multiple diagnostic categories.

The classes used in the experiments include the HAM10000 diagnostic categories, with representative visual comparisons performed on multiple classes.

## 🔍 Tasks Performed

### 1. Comparative Edge Detection

Several edge detection techniques are implemented and compared:

- **Sobel**
- **Prewitt**
- **Laplacian**
- **Laplacian of Gaussian (LoG)**
- **Canny**

The notebook generates visual comparisons between the original image and the output of each edge detector.

### 2. Effect of Noise on Edge Detection

The project investigates how different types of noise affect edge detection.

The following noise conditions are considered:

- Original images
- Gaussian noise
- Salt-and-pepper noise

Filtering techniques are also evaluated:

- Gaussian filtering
- Median filtering

The purpose is to observe how preprocessing can improve edge detection when images contain noise.

### 3. Canny Parameter Analysis

Different Canny edge detector configurations are tested by changing:

- Lower threshold
- Upper threshold
- Gaussian smoothing kernel size

The experiments compare the resulting edge maps using quantitative edge-quality measurements.

### 4. Dataset Preparation

Three image conditions are prepared for classification:

| Condition | Description |
|---|---|
| Raw | Original image |
| Filtered | Gaussian-filtered image |
| Edge | Canny edge representation |

This allows the classification models to be evaluated under different image-processing conditions.

## 🤖 Models Used

### Deep Learning Models

The project uses pretrained ImageNet models with modified classification heads:

- **ResNet50**
- **ResNet101**
- **DenseNet121**

These models are fine-tuned for the skin-lesion classification task.

### Classical Machine Learning Models

The following traditional machine learning algorithms are also evaluated:

- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

Feature extraction is performed from the neural network models so that the extracted representations can also be evaluated using classical machine learning algorithms.

## 📊 Evaluation Metrics

Model performance is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

The experiments compare performance across the:

- Raw image condition
- Filtered image condition
- Edge image condition

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- PIL
- KaggleHub
- Jupyter Notebook / Google Colab

## 📁 Project Structure

```text
Edge-Detection/
│
├── Edge-Detection.ipynb
├── README.md
└── outputs/
    ├── figures/
    ├── confusion_matrices/
    └── tables/
```

> The exact output folders may vary depending on where the notebook is executed.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Edge-Detection.git
cd Edge-Detection
```

### 2. Install the required libraries

```bash
pip install numpy pandas opencv-python matplotlib pillow scikit-learn torch torchvision kagglehub
```

### 3. Open the notebook

Run:

```bash
jupyter notebook Edge-Detection.ipynb
```

Alternatively, the notebook can be opened and executed in **Google Colab**.

### 4. Dataset

The notebook is configured to locate the HAM10000 dataset from Kaggle or download it using `kagglehub`.

If Kaggle authentication is required, configure the appropriate Kaggle API credentials before running the dataset preparation section.

## 📈 Experimental Comparison

The project performs a cross-condition comparison between different models and preprocessing techniques.

The final experiments compare:

```text
                Raw Images
                    │
                    ├── Classical ML
                    │
                    └── CNN Models
                         │
                         ▼
                Classification Results

              Filtered Images
                    │
                    ├── Classical ML
                    │
                    └── CNN Models

                Edge Images
                    │
                    ├── Classical ML
                    │
                    └── CNN Models
```

The resulting accuracy, precision, recall, and F1-score values are used to analyze the effect of image preprocessing and model architecture.

## 🎯 Learning Objectives

Through this project, the following concepts are explored:

1. Understanding how edge detection works.
2. Comparing different edge detection operators.
3. Understanding the effect of image noise.
4. Applying image filtering techniques.
5. Understanding Canny edge detector parameters.
6. Preparing images for machine learning.
7. Applying transfer learning using pretrained CNNs.
8. Comparing CNN models with classical ML algorithms.
9. Evaluating classification models using standard metrics.
10. Analyzing confusion matrices and classification performance.

## ⚠️ Notes

This project is intended for **academic and educational purposes**. The classification results should not be interpreted as a medical diagnostic system.

The notebook may require significant computational resources, particularly when training deep learning models such as ResNet and DenseNet.

## 👨‍💻 Author

**Mohsin Ashraf**

BS Artificial Intelligence  
COMSATS University Islamabad

---

⭐ If you find this project useful, consider giving the repository a star.
