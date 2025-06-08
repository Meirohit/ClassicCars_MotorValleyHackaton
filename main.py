# File: app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os
import re

# Load model and metrics
model = load("model/ridge_model.joblib")
with open("model/metrics.json") as f:
    metrics = json.load(f)
with open("model/model_insights.json") as f:
    model_insights = json.load(f)

# Load real monthly prices
monthly_df = pd.read_csv("model/monthly_prices.csv", parse_dates=["year"])

# Filter for the last 12 months only
last_year = monthly_df["year"].max() - pd.DateOffset(months=12)
monthly_df_last_year = monthly_df[monthly_df["year"] >= last_year]

# App config
st.set_page_config(page_title="TikTok Car Price Analyzer", layout="wide")
st.markdown("""
    <style>
        .car-card {
            border: 2px solid #444;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            background-color: #1a1a1a;
        }
        .car-title {
            color: #00ccff;
            font-size: 1.4rem;
            font-weight: bold;
        }
        .stImage > img {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 SentimentDrive: Investment Trends in Classic Cars")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Upload & Predict", "Model Metrics", "Per-Car Insights"])

# Feature list
features = [
    'Video views', 'Engagement rate', 'Like count', 'Comment count',
    'Share count', 'views_per_day', 'like_per_view', 'comment_per_view', 'share_per_view',
    'Sarcasm', 'Cons', 'Pros', 'Neutral', 'Not Related'
]

# Function to format car model name
format_car_name = lambda name: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name).title()

if page == "Upload & Predict":
    st.header("📤 Upload TikTok Video Data")
    uploaded_file = st.file_uploader("Upload a CSV file with feature columns", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if all(f in df.columns for f in features):
            preds = model.predict(df[features])
            df["Predicted 1-Month Price Change"] = preds/100
            st.success("✅ Prediction complete!")
            st.dataframe(df)
        else:
            st.error("❌ Missing required feature columns in uploaded file.")

elif page == "Model Metrics":
    st.header("📊 Model Performance")
    st.metric("R² Score", f"{metrics['r2_score']:.2f}")
    st.metric("Mean Absolute Error", f"${metrics['mae']:,.0f}")

    st.subheader("Feature Coefficients (Ridge Regression)")
    coef_dict = metrics.get("feature_coefficients", {})
    if coef_dict:
        coef_df = pd.DataFrame(coef_dict.items(), columns=["Feature", "Coefficient"]).sort_values(by="Coefficient")
        st.dataframe(coef_df)

        st.subheader("📉 Coefficient Chart")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            coef_df["Feature"],
            coef_df["Coefficient"],
            color=["#00cc66" if c > 0 else "#cc3300" for c in coef_df["Coefficient"]]  # Dark green and deep red
        )
        ax.axvline(0, color="#888", linestyle="--")
        ax.set_facecolor("#f0f0f0")
        ax.set_title("Effect of Each Feature on Price Change", fontsize=14)
        ax.tick_params(axis='x', colors='#444')
        ax.tick_params(axis='y', colors='#444')
        fig.patch.set_facecolor('#f0f0f0')
        st.pyplot(fig)
    else:
        st.warning("No coefficients found in metrics.json.")

    st.subheader("Top Positive Predictors")
    st.write(", ".join(metrics.get("top_positive", [])))
    st.subheader("Top Negative Predictors")
    st.write(", ".join(metrics.get("top_negative", [])))

elif page == "Per-Car Insights":
    st.header("🚘 Per-Car Model Analysis")
    for car, info in model_insights.items():
        car_display = format_car_name(car)
        with st.container():
            st.markdown('<div class="car-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 3], gap="large")
            with col1:
                st.markdown(f"<div class='car-title'>🔹 {car_display}</div>", unsafe_allow_html=True)
                st.markdown(f"**Average Price Change:** `{info['avg_change']}`")
                st.markdown(f"**Top Metric:** `{info['top_feature']}`")
                st.markdown(f"**Dominant Sentiment:** `{info['top_sentiment']}`")
                image_path = f"model/images/{car.lower()}.png"
                if os.path.exists(image_path):
                    st.image(image_path, caption=car_display, use_container_width=True)
                else:
                    st.warning("🚫 No image available.")
            with col2:
                car_column = car.lower()
                if car_column in monthly_df_last_year.columns:
                    fig, ax = plt.subplots(figsize=(5, 3), facecolor='#f5f5f5')
                    sns.set_style("whitegrid")
                    sns.lineplot(x=monthly_df_last_year["year"], y=monthly_df_last_year[car_column], ax=ax, marker="o", color="#3366cc")
                    ax.set_title(f"Last Year Price Trend for {car_display}")
                    ax.set_ylabel("Price")
                    ax.set_xlabel("Month")
                    ax.set_facecolor("#f5f5f5")
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    fig.autofmt_xdate()
                    st.pyplot(fig)
                else:
                    st.warning(f"No price data found for {car_display}.")
            st.markdown('</div>', unsafe_allow_html=True)
