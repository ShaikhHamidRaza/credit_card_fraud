# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques.

Fraud detection is a critical challenge in the financial industry because fraudulent transactions can lead to major financial losses and security risks. Since fraudulent transactions are extremely rare compared to normal transactions, this becomes a highly imbalanced classification problem.

The goal of this project is to build machine learning models that can accurately identify fraudulent transactions while balancing recall and precision.

---

# Objectives

- Detect fraudulent credit card transactions
- Handle highly imbalanced data effectively
- Compare multiple machine learning models
- Improve fraud detection using threshold tuning
- Evaluate models using appropriate classification metrics

---

# Dataset Information

- **Dataset Source:** Kaggle - Credit Card Fraud Detection Dataset
- **Total Transactions:** ~284,807
- **Fraud Cases:** ~0.17% of total transactions

## Dataset Features

| Feature | Description |
|------|------|
| V1–V28 | PCA-transformed anonymized features |
| Amount | Transaction amount |
| Time | Time elapsed between transactions |
| Class | Target variable (0 = Normal, 1 = Fraud) |

---

# Exploratory Data Analysis (EDA)

Several exploratory analyses were performed to better understand the dataset and fraud patterns.

## EDA Performed

- Fraud vs Normal transaction distribution
- Correlation heatmap
- Transaction amount distribution
- Fraud vs Amount analysis
- Feature distribution analysis
- Important feature visualization

## Key Findings

- Dataset is highly imbalanced
- Fraudulent transactions show different patterns compared to normal transactions
- Certain features like `V14`, `V17`, and `V12` strongly contribute to fraud detection

---

# Data Preprocessing

The following preprocessing steps were applied:

- Removed duplicate records
- Feature scaling using `StandardScaler`
- Stratified train-test split
- Handled class imbalance using `SMOTE`

---

# Machine Learning Models Used

The following machine learning models were trained and compared:

| Model | Purpose |
|------|------|
| Logistic Regression | Baseline classification model |
| Random Forest | Handles non-linear relationships |
| Gradient Boosting | Improves performance using boosting |
| XGBoost | Optimized boosting algorithm |

---

# Why These Models?

## Logistic Regression
Used as a simple and interpretable baseline model.

## Random Forest
Performs well on structured datasets and captures non-linear patterns effectively.

## Gradient Boosting
Improves prediction performance by combining multiple weak learners.

## XGBoost
Advanced boosting algorithm known for strong performance on tabular datasets.

---

# Machine Learning Workflow

The project follows this workflow:

1. Data loading
2. Data preprocessing
3. Exploratory Data Analysis
4. Feature scaling
5. Train-test split
6. SMOTE for imbalance handling
7. Model training
8. Cross-validation
9. Hyperparameter tuning
10. Model evaluation
11. Threshold optimization
12. Final prediction

---

# Cross Validation

Stratified K-Fold Cross Validation was used to ensure stable evaluation while maintaining class distribution across folds.

This helps reduce overfitting and provides more reliable model performance estimates.

---

# Hyperparameter Tuning

`GridSearchCV` was used to find the best hyperparameters for each model based on recall score.

Examples of tuned parameters:
- Number of estimators
- Maximum tree depth
- Learning rate
- Regularization parameters

---

# Evaluation Metrics

Since the dataset is highly imbalanced, accuracy alone is not reliable.

The following evaluation metrics were used:

| Metric | Description |
|------|------|
| Precision | Correctness of fraud predictions |
| Recall | Ability to detect actual fraud |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Overall classification performance |

---

# Threshold Optimization

Instead of using the default threshold (`0.5`), threshold tuning was applied to improve fraud detection performance.

### Why Threshold Tuning?

- Lower thresholds increase recall
- Higher thresholds reduce false positives
- Helps balance recall and precision based on business requirements

---

# Results

| Model | Recall | Precision | F1 Score |
|------|------|------|------|
| Logistic Regression | 0.84 | 0.09 | 0.17 |
| Random Forest | 0.94 | 0.92 | 0.93 |
| XGBoost | 0.95 | 0.93 | 0.94 |

## Best Performing Model

Random Forest and XGBoost achieved the best overall performance based on recall and F1-score.

---

# ROC Curve

ROC Curve was used to evaluate model performance across multiple classification thresholds.

ROC-AUC Score measures how well the model separates fraudulent and non-fraudulent transactions.

---

# Precision-Recall Curve

Precision-Recall Curve is more informative for highly imbalanced datasets like fraud detection.

It helps visualize the trade-off between:
- detecting fraud (recall)
- reducing false alerts (precision)

---

# Visualizations

## Fraud Distribution
![Fraud Distribution](images/fraud_distribution.png)

---

## Correlation Heatmap
![Heatmap](images/heatmap.png)

---

## Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

---

## ROC Curve
![ROC Curve](images/roc_curve.png)

---

## Precision-Recall Curve
![PR Curve](images/pr_curve.png)

---

# Feature Importance

Feature importance analysis showed that features such as:
- `V14`
- `V17`
- `V12`

had strong influence in detecting fraudulent transactions.

---

# Business Insights

In real-world banking systems:

- High recall is important because missing fraudulent transactions can lead to financial loss
- Extremely low precision can generate excessive false alerts
- A balance between recall and precision is necessary for practical fraud detection systems

---

# Future Improvements

Possible future improvements include:

- Real-time fraud detection system
- Explainable AI using SHAP
- Streamlit dashboard
- FastAPI deployment
- Deep learning approaches (Autoencoders)
- Advanced anomaly detection systems

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- XGBoost

---

# Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── images/
│   ├── fraud_distribution.png
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pr_curve.png
│
├── README.md
├── requirements.txt
```

---

#  How to Run

## 1. Clone Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
```

---

## 2. Navigate to Project Directory

```bash
cd credit-card-fraud-detection
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Jupyter Notebook

```bash
jupyter notebook
```

Open:
```text
fraud_detection.ipynb
```

---

#  Requirements

Create a `requirements.txt` file containing:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

---

#  Key Learnings

- Handling imbalanced datasets is essential in fraud detection
- Recall is more important than accuracy in fraud classification
- Threshold tuning significantly improves fraud detection performance
- Cross-validation and hyperparameter tuning improve model reliability
- Business understanding is important alongside machine learning performance

---

# Author

**Hamid Raza**

Aspiring AI/ML Engineer focused on building real-world machine learning systems and high-impact AI projects.
