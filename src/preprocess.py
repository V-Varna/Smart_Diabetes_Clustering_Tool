import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the raw dataset
df = pd.read_csv('C:\\Users\\Varna\\Desktop\\mini_project\\data\\diabetic_data.csv')

# Step 2: Replace '?' with np.nan (missing values)
df.replace('?', np.nan, inplace=True)

# Step 3: Drop useless columns
df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

# Step 4: Drop rows with invalid gender
df = df[df['gender'] != 'Unknown/Invalid']

# Step 5: Drop columns with too many missing values (optional)
df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

# Step 6: Fill remaining NaNs in useful columns
df['race'].fillna(df['race'].mode()[0], inplace=True)
df['max_glu_serum'].fillna('None', inplace=True)
df['A1Cresult'].fillna('None', inplace=True)

# Step 7: Encode binary columns (change, diabetesMed, readmitted)
binary_cols = ['change', 'diabetesMed', 'readmitted']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Step 8: One-hot encode categorical columns
df = pd.get_dummies(df, columns=['race', 'gender', 'age', 'max_glu_serum', 'A1Cresult'])

# Step 9: Show result
print("✅ Preprocessing complete!")
print("📊 Final shape:", df.shape)

# Step 10: Save cleaned dataset for next steps
df.to_csv('C:\\Users\\Varna\\Desktop\\mini_project\\data\\cleaned_diabetes.csv', index=False)
