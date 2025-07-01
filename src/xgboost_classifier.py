import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os

# Load the clustered dataset
df = pd.read_csv('C:/Users/Varna/Desktop/mini_project/data/clustered_diabetes.csv')

# Separate features and label
X = df.drop('cluster', axis=1)
y = df['cluster']

# One-hot encode categorical variables
X = pd.get_dummies(X)

# ✅ Rename columns to remove invalid characters for XGBoost
X.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_').replace('(', '').replace(')', '') for col in X.columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

joblib.dump(X.columns.tolist(), 'C:/Users/Varna/Desktop/mini_project/model/feature_columns.pkl')
# Save the model

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluation
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# ROC AUC Score for multiclass
print("🔥 ROC-AUC Score:", round(roc_auc_score(y_test, y_proba, multi_class='ovr'), 4))


# Create a 'model' directory if it doesn't exist
os.makedirs('C:/Users/Varna/Desktop/mini_project/model', exist_ok=True)

# Save the trained XGBoost model
joblib.dump(model, 'C:/Users/Varna/Desktop/mini_project/model/xgboost_model.pkl')

print("💾 Model saved successfully at: model/xgboost_model.pkl")
