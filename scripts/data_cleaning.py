import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataaset
file_path = '../data/telco_churn.csv'
data = pd.read_csv(file_path)

# Preview the dataset
print("Dataset Preview:")
print(data.head())

# Dataset information
print("Dataset Info:")
print(data.info())

# Summary Statistics
print("Summary Statistics:")
print(data.describe())

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Drop unncessary columns
data.drop(columns=['customerID'], inplace=True, errors='ignore')

# Fill missing values for numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# Fill missing values for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

print("missing Values After Cleaning:")
print(data.isnull().sum())

# Label encode binary columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
label_enc = LabelEncoder()
for col in binary_cols:
    data[col] = label_enc.fit_transform(data[col])

# One-hot encode categorical columns
data = pd.get_dummies(data, drop_first=True)

print("Data after Encoding:")
print(data.head())

# Save the cleaned data
cleaned_file_path = '../data/telco_churn_cleaned.csv'
data.to_csv(cleaned_file_path, index=False)
print("Cleaned data saved to {cleaned_file_path}")
