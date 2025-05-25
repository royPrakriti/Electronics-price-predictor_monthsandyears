import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import base64

# Background image CSS
def set_bg(img_file):
    with open(img_file, "rb") as file:
        b64_img = base64.b64encode(file.read()).decode()
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64_img}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2em;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# Load model and metadata
model = joblib.load("price_prediction_model_prakriti.pkl")
with open("trained_X_columns.json", "r") as f:
    trained_X_columns = json.load(f)
data = pd.read_csv("electronics_prices_2010_2030 (3).csv")
products = sorted(data["Product"].unique())
companies = sorted(data["Company"].unique())

# Setup
st.set_page_config("Electronics Price Predictor", layout="centered")
set_bg("pexels-cottonbro-8721318.jpg")

# Title and inputs
st.title("🔮 Electronics Price Predictor")
st.markdown("Predict **future electronics prices** and visualize trends from your selections.")

product = st.selectbox("📦 Select Product", products)
company = st.selectbox("🏷️ Select Company", companies)
year = st.number_input("📅 Year (e.g., 2026)", min_value=2010, max_value=2100, value=2026, step=1)
month = st.slider("🗓️ Month", 1, 12, 1)
compare_months = st.checkbox("📊 Compare All Months for Year")

# Prediction logic
def predict_prices(product, company, month, year, compare_months):
    try:
        if compare_months:
            months = list(range(1, 13))
            prices = []
            for m in months:
                time_index = (year - 2023) * 12 + (m - 1)
                row = dict.fromkeys(trained_X_columns, 0)
                row[f'Product_{product}'] = 1
                row[f'Company_{company}'] = 1
                row["TimeIndex"] = time_index
                input_df = pd.DataFrame([row])[trained_X_columns]
                prices.append(model.predict(input_df)[0])
            return months, prices
        else:
            years = [year - 2, year - 1, year]
            prices = []
            for y in years:
                time_index = (y - 2023) * 12 + (month - 1)
                row = dict.fromkeys(trained_X_columns, 0)
                row[f'Product_{product}'] = 1
                row[f'Company_{company}'] = 1
                row["TimeIndex"] = time_index
                input_df = pd.DataFrame([row])[trained_X_columns]
                prices.append(model.predict(input_df)[0])
            return years, prices
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return [], []

# Trigger
if st.button("🔍 Predict"):
    labels, results = predict_prices(product, company, month, year, compare_months)
    if labels:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(labels, results, marker='o', linewidth=2, color="#FF6B35")
        ax.set_title(f"Price Trend for {product} ({company})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Month" if compare_months else "Year", fontsize=12)
        ax.set_ylabel("Predicted Price (₹)", fontsize=12)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('₹{x:,.0f}'))
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.success(f"💰 Predicted Price for {product} ({company}) in {month}/{year}: ₹{round(results[-1], 2)}")
