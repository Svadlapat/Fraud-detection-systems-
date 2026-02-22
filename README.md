# Fraud-detection-systems

## Project Overview

This project implements a fraud detection system using a deep learning–based AutoEncoder model from the PyOD library. The model identifies fraudulent transactions by detecting anomalies based on reconstruction error. It is trained only on normal transactions and flags transactions with high reconstruction error as potential fraud.

## Objective

To build and evaluate an anomaly detection model capable of identifying fraudulent credit card transactions using unsupervised machine learning techniques.

## Dataset

Dataset used: Credit Card Fraud Detection Dataset (Kaggle)
Total Transactions: 284,807
Fraud Cases: 492
Highly imbalanced real-world dataset

## Methodology

1.Load dataset
2.Preprocess and normalize features
3.Train AutoEncoder on normal transactions only
4.Calculate reconstruction error for all samples
5.Classify anomalies as fraud
6.Evaluate performance using metrics and visualization

## Technologies Used

Python
PyOD
NumPy
Pandas
Scikit-learn
Matplotlib

## Installation & Setup

Install dependencies:
```
pip install -r requirements.txt
```
Run model:
```
python src/fraud_autoencoder.py
```
## Output Metrics

The model produces:
Classification Report
Confusion Matrix
ROC-AUC Score
Reconstruction Error Distribution Plot

Example performance:

ROC-AUC Score ≈ 0.94+
Detects anomalies based on deviation from learned normal patterns

## Result Interpretation

The AutoEncoder successfully learns patterns of legitimate transactions. Fraudulent transactions produce higher reconstruction errors, allowing the model to distinguish anomalies effectively despite dataset imbalance.

## Project Structure
project/
│
├── data/                   # Dataset folder
      └──data
            └──creditcard.csv                
├── src/
│     └── fraud_autoencoder.py
├── requirements.txt
├── manifest.txt
└── README.md


