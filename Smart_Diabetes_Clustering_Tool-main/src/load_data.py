import pandas as pd

# Load the dataset from the data/ folder
df = pd.read_csv('C:\\Users\\Varna\\Desktop\\mini_project\\data\\diabetic_data.csv')

# Show the first 5 rows
print("🔍 First 5 rows:")
print(df.head())

# Show total rows and columns
print(f"\n📊 Dataset shape: {df.shape}")

# Check for missing values
print("\n🧩 Missing values:")
print(df.isnull().sum())

# Check the data types
print("\n🔠 Data types:")
print(df.dtypes)

# See unique values for a few important columns
print("\n🎯 Unique values in 'race':", df['race'].unique())
print("🎯 Unique values in 'gender':", df['gender'].unique())
print("🎯 Unique values in 'age':", df['age'].unique())
print("🎯 Unique values in 'readmitted':", df['readmitted'].unique())
