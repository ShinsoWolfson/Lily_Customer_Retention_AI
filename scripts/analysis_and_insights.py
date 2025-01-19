import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def model_explainability(train_data_path):
    # Load preprocessed training data
    data = pd.read_csv(train_data_path)

    # Separate features and target
    X_train = data.drop(columns=['Churn'])
    y_train = data['Churn']

    # Train a Random Forest model for explainability
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # SHAP explainability using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Debugging
    print("SHAP values shape:", shap_values)
    print("SHAP values for class 1 shape:", shap_values[1].shape if isinstance(shap_values, list) else shap_values.shape)
    print("X_train shape:", X_train.shape)

    # Adjust SHAP values for summary plot
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]  # Values for class 1
    else:
        shap_values_class1 = shap_values

    # Generate SHAP summary plot
    shap.summary_plot(shap_values_class1, X_train, show=False)
    plt.title("SHAP Summary Plot for Churn Prediction (Churned Customers)")
    plt.savefig('shap_summary_plot.png')
    plt.close()
    print("SHAP summary plot saved as 'shap_summary_plot.png'.")


def business_insights(train_data_path):
    # Load preprocessed training data
    data = pd.read_csv(train_data_path)

    # Debug: Check dataset columns
    print("Columns in the dataset:", data.columns.tolist())

    # Recompute CostPerService with safeguards
    print("Recomputing CostPerService column...")
    data['CostPerService'] = data.apply(
        lambda row: row['MonthlyCharges'] / row['TotalServices'] if row['TotalServices'] > 0 else 0,
        axis=1
    )

    # Analyze average cost per service for churned vs non-churned customers
    insights = data.groupby('Churn')['CostPerService'].mean().reset_index()
    print("Business insights:")
    print(insights)

    # Save insights
    insights.to_csv('business_insights.csv', index=False)
    print("Business insights saved as 'business_insights.csv'.")


if __name__ == "__main__":
    train_file = '../data/preprocessed_train.csv'
    model_explainability(train_file)
    business_insights(train_file)
