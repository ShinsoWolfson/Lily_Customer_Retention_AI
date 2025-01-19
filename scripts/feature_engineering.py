import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
file_path = '../data/telco_churn_cleaned.csv'
data = pd.read_csv(file_path)

# Ensure 'tenure' exists
if 'tenure' not in data.columns:
    raise ValueError("The 'tenure' column is missing from the dataset. Please check the data cleaning step.")

# Adjust the maximum value for bins to avoid duplicates
max_tenure = data['tenure'].max() + 1  # Add 1 to avoid duplicate bin edges
bins = [0, 12, 24, 48, 72, max_tenure]
labels = ['0-1 Year', '1-2 Years', '2-4 Years', '4-6 Years', '6+ Years']
data['TenureGroup'] = pd.cut(data['tenure'], bins=bins, labels=labels, right=False)

# Feature 2: Total Services Subscribed
service_columns = [
    'PhoneService', 'MultipleLines', 'InternetService_Fiber optic',
    'InternetService_DSL', 'StreamingTV', 'StreamingMovies',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport'
]
existing_service_columns = [col for col in service_columns if col in data.columns]
data['TotalServices'] = data[existing_service_columns].sum(axis=1)

# Feature 3: Cost Per Service
data['CostPerService'] = data['MonthlyCharges'] / (data['TotalServices'] + 1)  # Avoid divide by zero

# Scale numerical features
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CostPerService']
numerical_cols = [col for col in numerical_cols if col in data.columns]  # Ensure all columns exist
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Save the processed dataset
processed_file_path = '../data/telco_churn_processed.csv'
data.to_csv(processed_file_path, index=False)
print(f"\nProcessed data saved to {processed_file_path}")
