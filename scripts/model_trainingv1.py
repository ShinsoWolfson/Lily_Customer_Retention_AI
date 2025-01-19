import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import joblib

# Load the processed dataset
file_path = '../data/telco_churn_processed.csv'
data = pd.read_csv(file_path)

# Debugging: Preview the dataset
print("Dataset Preview:")
print(data.head())

# Debugging: Check dataset info
print("\nDataset Info:")
print(data.info())

# Define target column and features
target_column = 'Churn'  # Update if necessary
X = data.drop(columns=[target_column])
y = data[target_column]

# Handle non-numerical columns
non_numerical_cols = X.select_dtypes(include=['object']).columns
if not non_numerical_cols.empty:
    print(f"Non-Numerical Columns: {non_numerical_cols.tolist()}")

# One-hot encode non-numerical columns
X = pd.get_dummies(X, drop_first=True)
print(f"Shape after encoding: {X.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict_proba(X_test)[:, 1]

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict_proba(X_test)[:, 1]

# Evaluate Logistic Regression
lr_auc = roc_auc_score(y_test, lr_preds)
print(f"\nLogistic Regression AUC-ROC: {lr_auc:.4f}")
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_model.predict(X_test)))

# Evaluate Random Forest
rf_auc = roc_auc_score(y_test, rf_preds)
print(f"\nRandom Forest AUC-ROC: {rf_auc:.4f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_model.predict(X_test)))

# Plot AUC-ROC Curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_preds)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_preds)

plt.figure(figsize=(10, 6))
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.4f})", color="blue")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.4f})", color="green")
plt.plot([0, 1], [0, 1], '--', label="Random Guessing", color="red")
plt.title("AUC-ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig('../dashboards/auc_roc_curve.png')
plt.show()

# Save Models
joblib.dump(lr_model, '../models/logistic_regression_modelv1.pkl')
joblib.dump(rf_model, '../models/random_forest_modelv1.pkl')
print("\nModels saved successfully.")
