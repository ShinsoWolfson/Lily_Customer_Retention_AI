import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score


def evaluate_models(test_file, model_dir):
    # Load preprocessed test data
    test_data = pd.read_csv(test_file)

    # Separate features and target
    X_test = test_data.drop(columns=['Churn'])
    y_test = test_data['Churn']

    # Load models
    lr_model = joblib.load(f"{model_dir}/logistic_regressionv2.pkl")
    rf_model = joblib.load(f"{model_dir}/random_forestv2.pkl")

    # Evaluate Logistic Regression
    print("Evaluating Logistic Regression model...")
    lr_predictions = lr_model.predict(X_test)
    lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
    print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, lr_predictions))

    # Evaluate Random Forest
    print("Evaluating Random Forest model...")
    rf_predictions = rf_model.predict(X_test)
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, rf_predictions))


if __name__ == "__main__":
    evaluate_models('../data/preprocessed_test.csv', '../models')
