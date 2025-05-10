import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import requests

# Load data from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/your-username/your-repo-name/main/Solarai.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()  # Clean column names
    return df

df = load_data()

st.title("🌞 Solar Generation Predictor")
st.markdown("Predict monthly solar generation (in MU) using future month and solar capacity.")

# Show data preview
with st.expander("Preview Dataset"):
    st.dataframe(df)

# Prepare model
X = df[['Month', 'Solar Capacity']]
y = df["Monthly Solar Generation(Mu's)"]

# Preprocessing
preprocessor = ColumnTransformer([
    ('month_enc', OneHotEncoder(handle_unknown='ignore'), ['Month'])
], remainder='passthrough')

model = make_pipeline(preprocessor, LinearRegression())
model.fit(X, y)

# Prediction UI
st.header("🔮 Make a Prediction")

months = list(df['Month'].unique())
input_month = st.selectbox("Select Month", sorted(months))
input_capacity = st.number_input("Enter Cumulative Solar Capacity (MW)", min_value=0.0, step=10.0)

if st.button("Predict Solar Generation"):
    input_data = pd.DataFrame([[input_month, input_capacity]], columns=['Month', 'Solar Capacity'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Monthly Solar Generation: **{prediction:.2f} MU**")

# Optional: model download
with st.expander("Download Trained Model"):
    joblib.dump(model, "solar_model.pkl")
    with open("solar_model.pkl", "rb") as f:
        st.download_button("Download Model", f, "solar_model.pkl")
