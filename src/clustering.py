import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset using full path
df = pd.read_csv('C:\\Users\\Varna\\Desktop\\mini_project\\data\\cleaned_diabetes.csv')

# Drop non-numeric columns before clustering
features = df.drop('readmitted', axis=1)
features_numeric = features.select_dtypes(include=['int64', 'float64'])  # Only numeric columns

# Scale the numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features_numeric)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_data)

# Show cluster distribution
print("✅ Cluster distribution:")
print(df['cluster'].value_counts())

# Save new dataset with cluster feature
df.to_csv('C:\\Users\\Varna\\Desktop\\mini_project\\data\\clustered_diabetes.csv', index=False)
print("\n📦 Saved new file with clusters: clustered_diabetes.csv")
