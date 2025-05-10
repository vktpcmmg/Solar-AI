import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load CSV from the same directory
@st.cache_data
def load_data():
    df = pd.read_csv("Solarai.csv")  # File must be in the same GitHub repo
    df.columns = df.columns.str.strip()  # Clean whitespace
    return df

df = load_data()

st.title("🔆 Solar Generation Predictor")
st.markdown("Enter a month and solar capacity to predict the monthly solar generation in MU's.")

# Show data preview
with st.expander("🔍 View Sample Data"):
    st.dataframe(df)

# Prepare model
X = df[['Month', 'Solar Capacity']]
y = df["Monthly Solar Generation(Mu's)"]

# Preprocessing: OneHot encode Month, pass Solar Capacity through
preprocessor = ColumnTransformer([
    ('month_enc', OneHotEncoder(handle_unknown='ignore'), ['Month'])
], remainder='passthrough')

model = make_pipeline(preprocessor, LinearRegression())
model.fit(X, y)

# Prediction section
st.header("📈 Predict Monthly Solar Generation")
month_input = st.selectbox("Select Month", sorted(df['Month'].unique()))
capacity_input = st.number_input("Enter Cumulative Solar Capacity (MW)", min_value=0.0, step=10.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[month_input, capacity_input]], columns=['Month', 'Solar Capacity'])
    predicted_mu = model.predict(input_df)[0]
    st.success(f"Predicted Solar Generation: **{predicted_mu:.2f} MU**")

# Optional: model download
with st.expander("⬇️ Download Trained Model"):
    joblib.dump(model, "solar_model.pkl")
    with open("solar_model.pkl", "rb") as f:
        st.download_button("Download Model", f, "solar_model.pkl")
