import pandas as pd
import joblib

# Load the trained model
model = joblib.load('C:/Users/Varna/Desktop/mini_project/model/xgboost_model.pkl')

# Load the feature columns used during training
feature_columns = joblib.load('C:/Users/Varna/Desktop/mini_project/model/feature_columns.pkl')

# Load the input data
df = pd.read_csv('C:/Users/Varna/Desktop/mini_project/data/clustered_diabetes.csv')

# Drop the target label column if present
if 'cluster' in df.columns:
    X = df.drop(columns=['cluster'])
else:
    X = df.copy()

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X)

# Align with training columns
for col in feature_columns:
    if col not in X_encoded.columns:
        X_encoded[col] = 0  # Add missing columns

# Reorder columns to match training data
X_encoded = X_encoded[feature_columns]

# Predict using the model
predictions = model.predict(X_encoded)

# Save predictions to the dataframe
df['predicted_cluster'] = predictions

# Save the updated data to a new CSV file
df.to_csv('C:/Users/Varna/Desktop/mini_project/data/predicted_clusters.csv', index=False)

print("✅ Predictions done and saved to predicted_clusters.csv")
