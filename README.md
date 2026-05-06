#  Credit Card Fraud Detection using Machine Learning

##  Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning models.

Fraud detection is a critical problem in the financial industry, as undetected fraud can lead to significant financial losses. The dataset used in this project is highly imbalanced, making the problem more challenging and realistic.

The goal is to build a model that can accurately identify fraudulent transactions while maintaining a balance between precision and recall.

---

##  Dataset Information
- Source: Kaggle - Credit Card Fraud Detection Dataset  
- Total Transactions: ~284,000  
- Fraud Cases: ~0.17% (highly imbalanced dataset)

### Data Description:
- `V1–V28`: PCA-transformed anonymized features  
- `Amount`: Transaction amount  
- `Time`: Time elapsed between transactions  
- `Class`: Target variable  
  - 0 → Normal transaction  
  - 1 → Fraud transaction  

---

##  Features
The model uses:
- PCA features (`V1–V28`)
- Transaction amount
- Time feature

Key observations:
- Features like **V14, V17, and V12** contribute significantly to fraud detection.

---

##  Machine Learning Approach

### Models Used:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost (optional)

### Why These Models?
- Logistic Regression → simple and interpretable baseline  
- Random Forest → handles non-linear relationships well  
- Gradient Boosting → improves performance on complex patterns  
- XGBoost → optimized boosting for higher accuracy  

---

##  Evaluation Metrics
Due to class imbalance, accuracy is not a reliable metric.

Metrics used:
- **Precision** → correctness of fraud predictions  
- **Recall** → ability to detect fraud cases  
- **F1 Score** → balance between precision and recall  
- **ROC-AUC** → model discrimination capability  

### Key Insight:
The model prioritizes **high recall**, as missing fraudulent transactions is more costly than false positives.

- In real-world systems, too many false positives can affect user experience, so a balance between recall and precision is required.

---

## Project Workflow

### 1. Data Preprocessing
- Removed duplicate records  
- Scaled the `Amount` feature  
- Handled class imbalance using SMOTE  

### 2. Train-Test Split
- Used stratified split to preserve class distribution  

### 3. Model Training
- Built pipelines for each model  
- Applied cross-validation (Stratified K-Fold)  
- Performed hyperparameter tuning (GridSearchCV)  

### 4. Model Evaluation
- Evaluated using recall, precision, and F1-score  
- Visualized performance using confusion matrix and ROC curve  
- Applied threshold tuning for better fraud detection

### 5.Visual Section Missing (Huge Impact)

##  Visualizations

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
![Confusion Matrix](images/confusion_matrix.png)

### 6. Prediction
- Generated fraud predictions based on probability thresholds  


---

## Threshold Optimization
Instead of using the default threshold (0.5), different thresholds were tested to balance recall and precision.

Lower threshold increases recall (detects more fraud), while higher threshold reduces false positives. The optimal threshold was selected based on F1 score and business requirements.

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

##  Project Structure

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
