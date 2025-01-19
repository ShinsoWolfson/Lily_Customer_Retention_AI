import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def train_and_save_models(train_file, output_dir):
    # Load preprocessed training data
    data = pd.read_csv(train_file)

    # Separate features and target
    X_train = data.drop(columns=['Churn'])
    y_train = data['Churn']

    # Train logistic regression
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train, y_train)

    # Evaluate logistic regression
    y_pred_lr = lr_model.predict(X_train)
    roc_auc_lr = roc_auc_score(y_train, lr_model.predict_proba(X_train)[:, 1])
    print("\nLogistic Regression Metrics:")
    print(f"ROC-AUC: {roc_auc_lr:.4f}")
    print("Classification Report:")
    print(classification_report(y_train, y_pred_lr))

    # Train random forest
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate random forest
    y_pred_rf = rf_model.predict(X_train)
    roc_auc_rf = roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])
    print("\nRandom Forest Metrics:")
    print(f"ROC-AUC: {roc_auc_rf:.4f}")
    print("Classification Report:")
    print(classification_report(y_train, y_pred_rf))

    # Save models
    joblib.dump(lr_model, f"{output_dir}/logistic_regression.pkl")
    joblib.dump(rf_model, f"{output_dir}/random_forest.pkl")
    print(f"\nModels saved to {output_dir}.")


if __name__ == "__main__":
    train_and_save_models('../data/preprocessed_train.csv', '../models')
