# Credit Card Fraud Detection

A machine learning classification project for detecting fraudulent credit card transactions using multiple classification models and techniques for handling highly imbalanced data.

## Project Highlights

- Worked with a highly imbalanced credit card transaction dataset
- Performed data cleaning, exploratory data analysis, and feature analysis
- Handled class imbalance using **SMOTE**
- Compared multiple classification models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Used **Stratified K-Fold Cross-Validation**
- Performed **GridSearchCV** for hyperparameter tuning
- Evaluated models using **Recall, Precision, and F1 Score**
- Performed threshold optimization
- Analyzed feature importance
- Focused on fraud detection rather than accuracy alone

## Results

The project compared the classification models using Recall, Precision, and F1 Score.

| Model | Recall | Precision | F1 Score |
|---|---:|---:|---:|
| Logistic Regression | 0.84 | 0.09 | 0.17 |
| Random Forest | 0.94 | 0.92 | 0.93 |
| XGBoost | 0.95 | 0.93 | 0.94 |

Based on the reported results, **XGBoost achieved the highest Recall and F1 Score** among the models shown in the results table.

The notebook also includes threshold analysis, confusion matrix, ROC-AUC, Precision-Recall analysis, and feature importance.

## Project Workflow

```text
Credit Card Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Preprocessing
        ↓
Train-Test Split
        ↓
SMOTE
        ↓
Model Training
        ↓
Cross-Validation
        ↓
Hyperparameter Tuning
        ↓
Model Comparison
        ↓
Threshold Optimization
        ↓
Final Evaluation
        ↓
Feature Importance
```

## Dataset

The project uses the **Credit Card Fraud Detection dataset**.

The dataset contains approximately **284,000 transactions**, with fraudulent transactions representing only a very small percentage of the total records.

### Features

| Feature | Description |
|---|---|
| `Time` | Time elapsed between transactions |
| `V1` – `V28` | PCA-transformed transaction features |
| `Amount` | Transaction amount |
| `Class` | Target variable |

### Target

**Class**

- `0` — Normal transaction
- `1` — Fraudulent transaction

## Machine Learning Models

The notebook evaluates several classification models:

| Model | Evaluation |
|---|---|
| Logistic Regression | Recall, Precision, F1 |
| Random Forest | Recall, Precision, F1 |
| Gradient Boosting | Recall, Precision, F1 |
| XGBoost | Recall, Precision, F1 |

The models are evaluated using stratified cross-validation and hyperparameter tuning.

## Handling Class Imbalance

Fraudulent transactions are highly underrepresented in the dataset.

To address this imbalance, the project uses **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data.

The project focuses mainly on **Recall, Precision, and F1 Score** because accuracy alone can be misleading for highly imbalanced classification problems.

## Key Insights

- The dataset contains a severe class imbalance between normal and fraudulent transactions.
- Ensemble models perform substantially better than the Logistic Regression baseline in the reported results.
- **XGBoost achieved the highest reported Recall (0.95) and F1 Score (0.94)**.
- Features such as **V14, V17, and V12** were highlighted as important features in the analysis.
- Threshold selection can change the balance between detecting fraud and generating false alerts.

## Project Structure

```text
credit-card-fraud-detection/
│
├── .gitignore
├── README.md
├── credit_card_fraud_detection.ipynb
├── creditcard.csv
└── requirements.txt
```

### File Description

**`credit_card_fraud_detection.ipynb`**  
Contains the complete analysis, preprocessing, model training, cross-validation, hyperparameter tuning, evaluation, threshold analysis, and visualizations.

**`creditcard.csv`**  
Input dataset used for fraud detection.

**`requirements.txt`**  
Lists the Python dependencies required to run the project.

**`.gitignore`**  
Prevents unnecessary files and local environment files from being tracked by Git.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd credit-card-fraud-detection
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
credit_card_fraud_detection.ipynb
```

Run the cells sequentially to reproduce the analysis and model evaluation.

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Imbalanced-learn**
- **XGBoost**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook**
- **Git & GitHub**

## What This Project Demonstrates

- Exploratory Data Analysis
- Data cleaning and preprocessing
- Handling highly imbalanced data
- SMOTE
- Classification modeling
- Stratified Cross-Validation
- Hyperparameter tuning
- Precision, Recall, and F1 evaluation
- Threshold optimization
- Feature importance analysis
- Model comparison

## Future Improvements

The project can be extended with:

- Improved preprocessing pipelines
- Time-based validation
- Separate validation data for threshold selection
- SHAP-based model explainability
- Additional feature engineering
- FastAPI inference API
- Docker containerization
- Cloud deployment
- Real-time fraud prediction

These are planned extensions and are not represented as completed features in the current version.

## Author

**Shaikh Hamid Raza**

- GitHub: [ShaikhHamidRaza](https://github.com/ShaikhHamidRaza)
- LinkedIn: [shaikhhamidraza](https://linkedin.com/in/shaikhhamidraza)
