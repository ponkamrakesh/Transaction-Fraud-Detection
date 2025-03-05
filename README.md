# Fraud Detection using Machine Learning

## Overview
This project focuses on detecting fraudulent transactions using machine learning models. The dataset contains various transaction details, including transaction amount, account balance, device type, location, and more. The goal is to build and evaluate models that can accurately classify fraudulent and non-fraudulent transactions.

## Dataset
The dataset consists of 50,000 records with the following features:
- **Transaction Details:** Transaction Amount, Transaction Type, Timestamp
- **User Information:** Account Balance, Device Type, Location, Merchant Category
- **Security Features:** Authentication Method, IP Address Flag, Previous Fraudulent Activity
- **Transaction History:** Daily Transaction Count, Failed Transactions in the Last 7 Days
- **Risk Factors:** Card Age, Risk Score, Transaction Distance
- **Target Variable:** Fraud_Label (0 - Not Fraudulent, 1 - Fraudulent)

## Data Preprocessing
1. **Dropped Unnecessary Columns:**
   - Removed `Transaction_ID`, `User_ID`, `Timestamp`, `Card_Type`, and `Card_Age` as they do not contribute to fraud detection.
2. **Categorical Feature Encoding:**
   - One-Hot Encoding was applied to `Transaction_Type`, `Device_Type`, `Location`, `Merchant_Category`, and `Authentication_Method`.
3. **Numerical Feature Scaling:**
   - Standardized numerical features using `StandardScaler` to normalize the data.

## Machine Learning Models
Two models were implemented and evaluated:

### 1. Logistic Regression
- A simple linear classifier was trained using `LogisticRegression`.
- Accuracy: **80.19%**
- Classification report highlights:
  - Precision: 84% for non-fraud, 72% for fraud
  - Recall: 88% for non-fraud, 64% for fraud

### 2. Random Forest Classifier
- An ensemble-based model using `RandomForestClassifier`.
- Number of estimators: **100** (increased from 1 for better accuracy)
- Cross-validation accuracy: **96.45%**
- Accuracy on test data: **96.83%**

## Overfitting Check
- Training Accuracy: **80.87%**
- Testing Accuracy: **80.19%**
- The model is generalizing well with no significant overfitting.

## Installation & Usage
### Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas scikit-learn
```

### Running the Script
Execute the Python script:
```bash
python Transaction.py
```

## Results
- Logistic Regression performed well but had lower recall for fraud detection.
- Random Forest significantly improved accuracy and cross-validation scores.
- Final model can be used to detect fraudulent transactions with high precision and recall.


