import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE


def preprocess_data(input_file, output_train_file, output_test_file):
    # Load dataset
    data = pd.read_csv(input_file)

    # Separate features and target
    X = data.drop(columns=['Churn'])
    y = data['Churn']

    # Identify non-numeric columns
    non_numeric_cols = X.select_dtypes(include=['object', 'bool']).columns

    # Encode non-numeric columns
    X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)

    # Drop constant features
    X = X.loc[:, (X != X.iloc[0]).any()]  # Retain only non-constant columns

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=20)  # Select top 20 features
    X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
    X_test_selected = selector.transform(X_test)

    # Save preprocessed datasets
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selector.get_feature_names_out())
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selector.get_feature_names_out())

    X_train_selected_df['Churn'] = y_train_resampled.values
    X_test_selected_df['Churn'] = y_test.values

    X_train_selected_df.to_csv(output_train_file, index=False)
    X_test_selected_df.to_csv(output_test_file, index=False)

    print(f"Preprocessed data saved to {output_train_file} and {output_test_file}")


if __name__ == "__main__":
    preprocess_data('../data/telco_churn_processed.csv', '../data/preprocessed_train.csv', '../data/preprocessed_test.csv')
