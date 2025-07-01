import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# === Step 1: Load Dataset ===
df = pd.read_csv('C:/Users/Varna/Desktop/mini_project/data/clustered_diabetes.csv')

# === Step 2: Split features and labels ===
X = df.drop('cluster', axis=1)
y = df['cluster']

# === Step 3: One-hot encode categorical features ===
X = pd.get_dummies(X)

# === Step 4: Clean column names ===
X.columns = [col.replace('[', '')
                .replace(']', '')
                .replace('<', '')
                .replace('>', '')
                .replace(' ', '_')
                .replace('(', '')
                .replace(')', '') for col in X.columns]

# === Step 5: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# === Step 6: Train XGBoost Model ===
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# === Step 7: Save model and feature columns ===
model_dir = 'C:/Users/Varna/Desktop/mini_project/model'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, 'xgboost_model.pkl'))
joblib.dump(X.columns.tolist(), os.path.join(model_dir, 'feature_columns.pkl'))
print("💾 Model and feature columns saved successfully!")

# === Step 8: Make Predictions ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# === Step 9: Evaluation ===
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
roc_auc = round(roc_auc_score(y_test, y_proba, multi_class='ovr'), 4)
print(f"\n🎯 Accuracy: {accuracy} %")
print(f"🔥 ROC-AUC Score: {roc_auc}")

# === Step 10: Plot Top 30 Feature Importances (Vertical Bars) ===
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order

top_k = 30
top_indices = indices[:top_k]
top_features = [X.columns[i] for i in top_indices]
top_importances = importances[top_indices]

plt.figure(figsize=(14, 6))
plt.bar(range(top_k), top_importances)
plt.xticks(range(top_k), top_features, rotation=90, fontsize=8)
plt.xlabel("Feature Name")
plt.ylabel("Importance Score")
plt.title("Top 30 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()
