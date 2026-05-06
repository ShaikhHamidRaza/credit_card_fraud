#  Credit Card Fraud Detection using Machine Learning

##  Problem Statement
Credit card fraud is a major issue in the financial industry. The goal of this project is to detect fraudulent transactions using machine learning techniques.

This is a highly imbalanced classification problem where fraudulent transactions represent a very small fraction of the dataset.

---

##  Objective
- Build a model to detect fraud transactions
- Maximize **Recall** (important for catching fraud)
- Maintain a balance between precision and recall

---

##  Dataset
- Source: Kaggle Credit Card Fraud Dataset
- Total transactions: 284,807
- Fraud cases: ~0.17%
- Features:
  - V1–V28 (PCA transformed features)
  - Amount
  - Time
  - Target: Class (0 = Normal, 1 = Fraud)

---

##  Exploratory Data Analysis
- Class imbalance visualization
- Correlation heatmap
- Fraud vs Amount analysis
- Feature distribution (V14, V17 etc.)

---

##  Data Preprocessing
- Removed duplicates
- Feature scaling (Amount)
- Train-Test Split (Stratified)
- SMOTE for handling class imbalance

---

##  Models Used
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

---

##  Model Training Approach
- Pipeline used (SMOTE + Model)
- Cross Validation (Stratified K-Fold)
- Hyperparameter Tuning (GridSearchCV)

---

##  Evaluation Metrics
- Recall (Primary metric)
- Precision
- F1 Score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

---

##  Threshold Optimization
Instead of default 0.5 threshold, custom threshold tuning was used to balance recall and precision.

---

##  Results
| Model | Recall | Precision | F1 Score |
|------|------|------|------|
| Logistic Regression | 0.89 | 0.91 | 0.90 |
| Random Forest | 0.94 | 0.92 | 0.93 |
| XGBoost | 0.95 | 0.93 | 0.94 |

Best Model: **Random Forest / XGBoost**

---

##  Key Insights
- Fraud detection requires high recall
- Imbalanced datasets need special handling (SMOTE)
- Threshold tuning significantly improves performance

---

##  Future Improvements
- Real-time fraud detection system
- Deep learning (Autoencoders)
- Explainable AI (SHAP)
- Deployment using Streamlit / FastAPI

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib, Seaborn
- XGBoost

---

## 📂 Project Structure

credit-card-fraud-detection/
│
├── data/
├── notebooks/
│ └── fraud_detection.ipynb
├── src/
├── README.md
├── requirements.txt

---

##  Author
Hamid Raza
