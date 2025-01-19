# Lily - Customer Retention AI

Lily is a machine learning project designed to predict customer retention rates, providing actionable insights for business strategies. Currently in development, this repository showcases modeling and evaluation processes, with deployment features coming soon.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Steps Taken](#steps-taken)
5. [Challenges Faced](#challenges-faced)
6. [Evaluation of v1 vs v2 Models](#evaluation-of-v1-vs-v2-models)
7. [Key Features](#key-features)
8. [Insights](#insights)
9. [Installation and Usage](#installation-and-usage)
10. [Next Steps](#next-steps)
11. [Contact Info](#contact-info)

---

## Project Overview

The Lily project aims to:

- Predict customer churn using machine learning models.
- Provide insights into the factors influencing customer churn.
- Generate business insights to improve decision-making and reduce churn rates.

**Key Metrics**: 
- Logistic Regression AUC-ROC: 0.8557
- Random Forest AUC-ROC: 0.8001

---

## Dataset

- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  - The dataset can be downloaded from [Kaggle](https://www.kaggle.com/...).
- **Key Columns**:
  - Categorical features: Gender, InternetService, Contract, etc.
  - Numerical features: MonthlyCharges, tenure, TotalCharges.
  - Target: `Churn` (Binary: Yes/No)

Preprocessed data includes:
- **Feature engineering**: One-hot encoding, tenure grouping, cost-per-service calculations.
- **Handling imbalances**: SMOTE oversampling for balanced classes.

---

## Technologies Used

- **Python**: Programming language
- **Libraries**:
  - pandas, NumPy: Data manipulation
  - Scikit-learn: Machine learning models
  - SHAP: Model explainability
  - Matplotlib, Seaborn: Visualization
  - imbalanced-learn: Oversampling techniques
- **Environment**:
  - Linux
  - PyCharm IDE

---

## Steps Taken

1. **Data Preprocessing**:
   - Cleaned raw data by handling missing values and encoding categorical variables.
   - Engineered new features (e.g., CostPerService, TotalServices).
   - Addressed class imbalance using SMOTE.

2. **Model Training**:
   - Trained Logistic Regression and Random Forest models.
   - Evaluated models using AUC-ROC and classification reports.

3. **Model Explainability**:
   - Used SHAP to understand feature importance and explain model predictions.

4. **Business Insights**:
   - Derived actionable insights (e.g., cost-per-service analysis by churn).

5. **Documentation**:
   - Comprehensive documentation.

---

## Challenges Faced

1. **Class Imbalance**:
   - Churned customers represented a smaller proportion of the dataset.
   - Solution: Applied SMOTE to balance classes, improving model performance.

2. **Feature Importance**:
   - Identifying impactful features among 6,500+ engineered features was challenging.
   - Solution: Leveraged SHAP for feature importance analysis.

3. **Model Overfitting**:
   - Random Forest showed near-perfect training performance but lower testing accuracy.
   - Solution: Adjusted hyperparameters and evaluated with AUC-ROC and classification reports.

4. **Data Inconsistencies**:
   - Issues with non-numeric fields like `TotalCharges` and inconsistent data types.
   - Solution: Ensured proper cleaning and feature engineering.

5. **Explainability Issues**:
   - SHAP values initially mismatched the dataset structure.
   - Solution: Debugged by ensuring consistent data shapes and formats.

---

## Evaluation of v1 vs v2 Models

### **v1 Models**:
- **Logistic Regression**:
  - AUC-ROC: 0.8447
  - Simpler model, limited feature engineering.
- **Random Forest**:
  - AUC-ROC: 0.9998
  - Overfitting observed, with near-perfect training performance.

### **v2 Models**:
- **Logistic Regression**:
  - AUC-ROC: 0.8557
  - Improved balance between recall and precision, better generalization.
- **Random Forest**:
  - AUC-ROC: 0.8001
  - Reduced overfitting, though performance slightly dropped on testing data.

**Key Improvements in v2**:
- Balanced class representation using SMOTE.
- Advanced feature engineering (e.g., tenure grouping, TotalServices).
- Enhanced evaluation metrics for clearer performance insights.

---

## Key Features

- **Predictive Models**: Logistic Regression and Random Forest.
- **Model Evaluation**: ROC-AUC, classification reports.
- **Explainability**: SHAP summary plots for feature importance.
- **Business Insights**: Cost analysis for churn vs. non-churn customers.

---

## Insights

- **Model Performance**:
  - Logistic Regression achieved balanced performance with AUC-ROC of 0.8557.
  - Random Forest exhibited slightly lower generalization ability.

- **Business Insights**:
  - Churned customers incurred higher average costs per service than non-churned customers.
  - Long-term contracts and bundled services significantly reduced churn rates.

- **Top Features Influencing Churn**:
  - Tenure
  - Monthly charges
  - Contract type

---

## Installation and Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ShinsoWolfson/Customer_Retention_Prediction.git
   cd Customer_Retention_Prediction

2. **Install Dependencies**:
    ``` bash
   pip install -r requirements.txt

3. **Run Preprocessing**:
    ``` bash
   python scripts/data_preprocessing.py

4. **Train Models**:
    ``` bash
   python scripts/model_training_v2.py

5. **Evaluate Models**:
    ``` bash
   python scripts/model_evaluation.py

6. **Generate SHAP Summary Plot**:
    ``` bash
   python scripts/analysis_and_insights.py

---

## Next Steps

- **Deployment**:
  - Deployment-ready models.

- **Enhance Explainability**:
  - Explore additional tools like LIME.
  - Provide localized explanations for individual predictions.

- **Explore Advanced Models**:
  - Gradient Boosting (e.g., XGBoost, LightGBM)
  - Neural Networks for more complex insights.

- **Automate Reporting**:
  - Integrate dashboards for real-time insights.

---

## Contact Info

-**Proprietary Notice**
This repository contains components that are proprietary to the creator, including but not limited to:
- Machine learning models located in the `models/` directory.
- Specific business logic implemented in `flask_app/` and `scripts/`.

The contents of these directories are for educational and portfolio purposes only. Reproduction, distribution, or modification for commercial purposes is strictly prohibited without prior permission.
For inquiries about licensing or collaboration, please contact Shinso at the contact info below for more details about this project or other inquiries:
- **Email:** [wolfsonshinso@gmail.com]
- **GitHub:** [https://github.com/ShinsoWolfson]