import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import datetime

@st.cache_data
def load_data():
    df = pd.read_csv("Solarai.csv")
    df.columns = df.columns.str.strip()
    df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')  # e.g. 'Apr-23'
    df['Month_Num'] = df['Month'].dt.month  # 1–12
    return df

df = load_data()

# Features: month (1–12) as categorical, capacity as numeric
X = df[['Month_Num', 'Solar Capacity']]
y = df["Monthly Solar Generation(Mu's)"]

# OneHot encode month
preprocessor = ColumnTransformer([
    ('month_enc', OneHotEncoder(handle_unknown='ignore'), ['Month_Num'])
], remainder='passthrough')

model = make_pipeline(preprocessor, LinearRegression())
model.fit(X, y)

# --- UI ---
st.title("🔆 Improved Solar Generation Predictor")

st.markdown("Enter future **month** and **solar capacity** to get a more accurate MU prediction (based on seasonality).")

future_month = st.date_input("Select a month", min_value=datetime.date(2025, 5, 1))
capacity_input = st.number_input("Enter Solar Capacity (MW)", min_value=0.0, step=1.0)

# Extract numeric month
month_num = future_month.month

if st.button("Predict"):
    input_df = pd.DataFrame([[month_num, capacity_input]], columns=['Month_Num', 'Solar Capacity'])
    predicted_mu = model.predict(input_df)[0]
    st.success(f"Predicted Solar Generation in {future_month.strftime('%B %Y')}: **{predicted_mu:.2f} MU**")
