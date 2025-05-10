import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime

@st.cache_data
def load_data():
    df = pd.read_csv("Solarai.csv")
    df.columns = df.columns.str.strip()
    df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')  # e.g., 'Apr-23'
    return df

df = load_data()

st.title("🔆 Solar Generation Predictor (Future-Ready)")

# Convert to numerical 'months since start'
start_date = df['Month'].min()
df['Months_Since_Start'] = df['Month'].apply(lambda x: (x.year - start_date.year) * 12 + (x.month - start_date.month))

# Prepare training data
X = df[['Months_Since_Start', 'Solar Capacity']]
y = df["Monthly Solar Generation(Mu's)"]

model = LinearRegression()
model.fit(X, y)

# ---- UI for future prediction ----
st.header("📅 Predict Future Solar Generation")
future_month = st.date_input("Select a future month", min_value=datetime.date(2025, 5, 1))
capacity_input = st.number_input("Enter Cumulative Solar Capacity (MW)", min_value=0.0, step=10.0)

# Convert future_month to numeric value
months_since_start = (future_month.year - start_date.year) * 12 + (future_month.month - start_date.month)

if st.button("Predict"):
    input_df = pd.DataFrame([[months_since_start, capacity_input]], columns=['Months_Since_Start', 'Solar Capacity'])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Solar Generation: **{prediction:.2f} MU** in {future_month.strftime('%B %Y')}")
