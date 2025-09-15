# Pragati L - CE22B089

# DA5401 A4: GMM-Based Synthetic Sampling for Imbalanced Data

## Overview
This assignment addresses class imbalance in fraud detection using **Gaussian Mixture Model (GMM)-based synthetic sampling**. The goal is to improve detection of minority (fraudulent) transactions without overfitting.  

The notebook includes:
- Baseline model training on imbalanced data.
- GMM-based synthetic oversampling and clustering-based undersampling (CBU).
- Performance evaluation and comparison with the baseline.

## Dataset
- **Credit Card Fraud Detection Dataset** from Kaggle: [Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Highly imbalanced, with few fraudulent transactions.

## Tasks

**Part A: Baseline Model**
- Load and explore data; show class imbalance.
- Train Logistic Regression on imbalanced training set.
- Evaluate using Precision, Recall, and F1-score.

**Part B: GMM-Based Sampling**
- Theoretical explanation: GMM vs SMOTE.
- Fit GMM to minority class and select components using AIC/BIC.
- Generate synthetic samples and rebalance dataset with CBU.

**Part C: Evaluation**
- Train Logistic Regression on GMM-balanced data.
- Compare metrics (Precision, Recall, F1) with baseline.
- Provide final recommendation on GMM effectiveness.

## Usage
1. Place `creditcard.csv` in `data/` folder.
2. Run `Assignment 4.ipynb` in Jupyter Notebook.
3. Required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
