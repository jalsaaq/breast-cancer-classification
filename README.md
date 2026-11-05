# Breast Cancer Classification

A machine learning project that classifies breast tumours as **malignant (M)** or **benign (B)** from measurements of cell nuclei. It compares three models, **Logistic Regression**, **Perceptron** and **Support Vector Machine (SVM)**. The SVM reached **97.4% test accuracy**.

This project was written for a university Machine Learning assignment (CPCS433, Assignment 1). The code is kept simple and readable, using basic pandas, matplotlib, seaborn and scikit-learn.

## Dataset

- File: `breast-cancer.csv` (569 samples, 30 numeric features plus an `id` column and the `diagnosis` target)
- Target: `diagnosis` (`M` = malignant, `B` = benign)
- Features: quantitative metrics of cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry and more), given as mean, standard error and worst value

## What the Notebook Does

1. Loads the dataset with pandas
2. **Data preprocessing**
   - Checks for missing values
   - Checks for outliers with the IQR method and boxplots
   - Encodes `diagnosis` as numbers (M = 1, B = 0)
   - Scales the features with `StandardScaler` (fitted on the training data only)
3. Splits the data 80% train / 20% test (`random_state=42`, stratified)
4. Trains **Logistic Regression** and **Perceptron**
5. Trains an **SVM** and uses `GridSearchCV` to tune `C` and `kernel`
6. Evaluates all models on the test set with `classification_report`
7. Plots a confusion matrix (seaborn heatmap) for each model
8. Plots the ROC curves of all three models on one graph
9. **PCA visualizations**: reduces the features to 2D and plots the decision boundary of each model, with the two classes in different colours

## Results

Test set of 114 samples:

| Model | Test accuracy | AUC |
|---|---|---|
| Logistic Regression | 96.5% | 0.996 |
| Perceptron | 97.4% | 0.991 |
| **SVM** (`C=1`, `kernel='rbf'`) | **97.4%** | 0.995 |

- The best SVM settings from `GridSearchCV` were `C=1` and the `rbf` kernel (5-fold cross-validation accuracy 97.6%).
- All three models do well. The Perceptron matches the SVM's accuracy, and Logistic Regression has the highest AUC.
- The 2D PCA plots are for visualization only. The two components cover about 63% of the variance, so the boundaries are a simplified view.

## Project Structure

```
breast-cancer-classification/
├── breast-cancer-classification.ipynb   # Main notebook with code and outputs
├── breast-cancer.csv                    # Dataset
├── breast-cancer-report.pdf             # Written report (PDF)
├── breast-cancer-report.docx            # Written report (Word)
└── README.md
```

## Getting Started

### Requirements

- Python 3.8 or newer
- pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter

### Install

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run

```bash
jupyter notebook breast-cancer-classification.ipynb
```

Keep `breast-cancer.csv` in the same folder as the notebook. To use Google Colab instead, upload both files and run the notebook there.

## Tools Used

- Python
- pandas
- matplotlib and seaborn
- scikit-learn (Logistic Regression, Perceptron, SVC, GridSearchCV, PCA, StandardScaler)
- Jupyter Notebook

## Author

Jana Alsaaq
