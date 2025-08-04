
# 🩺 Smart Diabetes Clustering Tool

A machine learning-powered Streamlit web app designed to analyze and cluster patient data from hospitals across the United States. This project uses unsupervised and supervised learning to group patients and predict their diabetes risk profiles using real-world hospital data.

> 🔬 Dataset used: [Diabetes 130-US hospitals for years 1999–2008](https://www.kaggle.com/datasets/whenamancodes/diabetes-patient-data-for-smarthealth-analysis)

---

## 📌 Features

- ✅ K-Means Clustering on cleaned and preprocessed hospital diabetes data
- ✅ XGBoost Classification to predict cluster labels
- ✅ Streamlit frontend with:
  - File upload and preprocessing
  - Cluster prediction and result download
  - Cluster distribution pie chart
  - Feature importance bar plot
- 💾 Saved models for reuse without retraining
- 📈 SMOTE used to handle class imbalance
- 🧪 Evaluation metrics: Accuracy, Confusion Matrix, ROC AUC
- 📂 Organized modular backend for easy maintenance

---

## 🚧 Upcoming Features (Planned)

- 📄 OCR + Regex: Extract lab values from PDF/image reports
- 🧠 Rule-based Chatbot: Provide self-care advice for diabetic patients
- 🎯 Awareness Quiz: Educate users through interactive diabetes quizzes
- 📄 PDF Report Generator: Generate personalized analysis summary

---

## 🏗️ Project Structure

```

Smart\_Diabetes\_Clustering\_Tool/
├── app.py                     # Streamlit main app
├── data/
│   └── diabetes.csv           # Original dataset
├── model/
│   ├── xgboost\_model.pkl      # Trained model
│   ├── scaler.pkl             # Saved MinMax scaler
│   ├── encoder.pkl            # Categorical encoder
│   └── feature\_names.pkl      # Selected feature list
├── backend/
│   ├── load\_data.py
│   ├── preprocess.py
│   ├── clustering.py
│   ├── xgboost\_classifier.py
│   └── predict\_from\_model.py
├── requirements.txt
└── README.md

````

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/V-Varna/Smart_Diabetes_Clustering_Tool.git
cd Smart_Diabetes_Clustering_Tool
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```


---

## 📊 Visuals

> Below are sample visuals rendered by the app:

* ✅ Cluster Distribution Pie Chart
* 📈 Feature Importance Bar Graph
* 🗃️ Downloadable prediction CSV file

<img width="789" height="675" alt="Screenshot 2025-07-04 103959" src="https://github.com/user-attachments/assets/b4e5b236-4d08-4e57-8470-6747280196eb" />

<img width="739" height="738" alt="Screenshot 2025-07-04 104012" src="https://github.com/user-attachments/assets/55d1922f-5fe3-4a8b-aceb-d9550e890970" />

<img width="763" height="681" alt="Screenshot 2025-07-04 104025" src="https://github.com/user-attachments/assets/ab74239b-0aaa-4ba6-b2e4-aea0b933b482" />

---

## 🤖 ML Techniques Used

* **K-Means Clustering**: To create cluster labels from preprocessed patient data
* **XGBoost Classifier**: Trained using the above clusters as labels
* **SMOTE**: For handling class imbalance in the training set

---

## 🧠 Future Scope

* 🧾 Personalized PDF reports
* 🧠 Interactive diabetes chatbot
* 🧪 Knowledge quiz for awareness and prevention

---

## 📚 References

* Dataset: [https://www.kaggle.com/datasets/whenamancodes/diabetes-patient-data-for-smarthealth-analysis](https://www.kaggle.com/datasets/whenamancodes/diabetes-patient-data-for-smarthealth-analysis)
* SMOTE: [https://imbalanced-learn.org/stable/over\_sampling.html](https://imbalanced-learn.org/stable/over_sampling.html)
* Streamlit Docs: [https://docs.streamlit.io/](https://docs.streamlit.io/)
* XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)


## 📝 License

This project is under the MIT License. See the [LICENSE](LICENSE) file for details.

```
