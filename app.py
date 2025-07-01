import streamlit as st
import pandas as pd
import joblib

# Set a light, modern theme and custom styles
st.set_page_config(page_title="Diabetes Cluster Predictor", layout="centered")

# Custom CSS for a professional, web-like look
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #f4f8fb;
        color: #222;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 4px 24px rgba(79,142,247,0.10);
        max-width: 700px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4F8EF7 0%, #6FC3FF 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.7em 2em;
        font-size: 1.1em;
        font-weight: 600;
        transition: background 0.2s;
        box-shadow: 0 2px 8px rgba(79,142,247,0.08);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #4F8EF7 100%);
    }
    .stDataFrame, .stTable {
        background: #f5f7fa;
        border-radius: 10px;
        font-size: 1.05em;
        box-shadow: 0 1px 4px rgba(79,142,247,0.04);
    }
    .stFileUploader {
        background: #eaf6ff;
        border-radius: 8px;
        padding: 1em;
        border: 1.5px dashed #4F8EF7;
        margin-bottom: 1.5em;
    }
    /* Style for st.info block */
    div[data-testid="stAlertInfo"] {
        background-color: #D3E6F8 !important;
        border-left: 6px solid #4F8EF7 !important;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(79,142,247,0.07);
        margin-top: 1.5em;
        margin-bottom: 1.5em;
    }
    div[data-testid="stAlertInfo"] > div {
        color: #000 !important;
        font-weight: 600;
        font-size: 1.1em;
    }
    .st-emotion-cache-1w7qfeb {
        color: #000 !important;
    }
    h1, .stTitle {
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #2563eb;
    }
    /* Style for st.download_button */
    .st-emotion-cache-z8vbw2 {
        color: #fff !important;
        background-color: rgb(19, 23, 32) !important;
        border: 1px solid rgba(250, 250, 250, 0.2) !important;
        transition: color 0.2s, background 0.2s;
    }
    .st-emotion-cache-z8vbw2:hover {
        color: #16a34a !important; /* green */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Diabetes Cluster Predictor")
st.markdown(
    """
    <div style='font-size:1.15em; color:#333; margin-bottom:1.5em; background: #fff; border-radius: 14px; box-shadow: 0 2px 12px rgba(79,142,247,0.10); padding: 1.5em 2em; border: 1.5px solid #e3eafc;'>
    Upload your diabetes data and get instant cluster predictions using our advanced hybrid ML model.<br>
    <b>Fast, accurate, and easy to use!</b>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("ℹ️ How to use this app", expanded=True):
    st.markdown("""
    <ul style='font-size:1.05em;'>
      <li><b>Step 1:</b> Click <b>Browse files</b> and select your CSV file (up to 200MB).</li>
      <li><b>Step 2:</b> Wait for the predictions to appear in the table below.</li>
      <li><b>Step 3:</b> Download the results if you want!</li>
    </ul>
    """, unsafe_allow_html=True)

# Load model and feature columns
def load_model():
    model = joblib.load("model/xgboost_model.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload a diabetes dataset CSV file (max 200MB)")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("<b>👀 Preview of uploaded data:</b>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    # One-hot encode and align columns
    X_encoded = pd.get_dummies(df)
    for col in feature_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[feature_columns]

    # Predict
    predictions = model.predict(X_encoded)
    df['predicted_cluster'] = predictions

    st.markdown("<b>🧩 Predictions:</b>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=csv,
        file_name="predicted_clusters.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file to get started.")
