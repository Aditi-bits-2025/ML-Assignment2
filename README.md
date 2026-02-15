# README.md --- ML Assignment 2

## Problem Statement

The objective of this project is to build, evaluate, and deploy multiple
machine learning classification models on a real-world dataset. The
models are compared using standard evaluation metrics and demonstrated
through an interactive Streamlit web application.

## Dataset Description

The dataset used is the Breast Cancer Wisconsin Dataset containing 569
samples with 30 numerical features for binary tumor classification.

## Models Implemented

-   Logistic Regression
-   Decision Tree Classifier
-   K-Nearest Neighbors (KNN)
-   Naive Bayes (Gaussian)
-   Random Forest (Ensemble)
-   XGBoost (Ensemble)

## Model Performance Comparison

| ML Model              | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
|----------------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.979    | 0.997 | 0.988     | 0.977  | 0.983    | 0.955 |
| Decision Tree       | 0.944    | 0.944 | 0.965     | 0.943  | 0.954    | 0.882 |
| KNN                 | 0.958    | 0.985 | 0.966     | 0.966  | 0.966    | 0.910 |
| Naive Bayes         | 0.951    | 0.994 | 0.965     | 0.955  | 0.960    | 0.896 |
| Random Forest       | 0.972    | 0.995 | 0.967     | 0.988  | 0.977    | 0.940 |
| XGBoost             | 0.965    | 0.992 | 0.966     | 0.977  | 0.972    | 0.925 |

## Observations

  ML Model              Observation
  --------------------- -----------------------------
  | Model               | Performance Summary                         |
|--------------------|---------------------------------------------|
| Logistic Regression | Strong baseline performance                 |
| Decision Tree       | Slight overfitting                          |
| KNN                 | High accuracy with scaling                 |
| Naive Bayes         | Fast but slightly lower performance        |
| Random Forest       | Stable and accurate                        |
| XGBoost             | Best overall                               |

## Streamlit Features

-   CSV upload
-   Model selection
-   Metrics display
-   Confusion matrix

## Project Structure

project-folder/ ├── app.py
├── model\model_training.py
├── requirements.txt
├── README.md
└── model\
