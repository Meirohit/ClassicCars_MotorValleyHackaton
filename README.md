# SentimentDrive: Investment Trends in Classic Cars 🚗

This project was developed during the **Motor Valley Fest 2025 Hackathon**.  
The goal was to analyze how **TikTok engagement metrics** (likes, comments, shares, views, engagement rate) on posts with car-related hashtags correlate with **short-term price changes** of classic cars.

---

## Dataset & ETL  
After performing ETL on the raw dataset provided by the organizers, I focused on 5 iconic models with sufficient data:  
- Ferrari Enzo  
- Ferrari F40  
- Ford GT40  
- Lamborghini LM002  
- McLaren F1  

The analysis combines:
- **Engagement metrics** (video views, like/comment/share ratios, sentiment tags)  
- **Monthly average price data** (collected and aligned to the same timeframe)

---

## Model  
A **Ridge Regression model** was trained to predict **1-month price changes**.  

### Key results (`metrics.json`):
- **R² Score:** 0.67  
- **MAE:** \$5,345  
- **Top positive predictors:** Engagement rate, like_per_view  
- **Top negative predictors:** Video views, Share count  

---

## Per-Car Insights  
`model_insights.json` highlights how each model’s price trend correlates with engagement and sentiment. For example:  
- **Ferrari Enzo** → strong growth driven by engagement rate and positive sentiment  
- **McLaren F1** → prices correlate negatively with shares and cons sentiment  

---

## Streamlit Dashboard (`main.py`)  
The repository includes a **Streamlit dashboard** with three sections:

1. **📤 Upload & Predict**  
   Upload TikTok engagement data and predict expected 1-month price changes.  

2. **📊 Model Metrics**  
   Explore regression performance, coefficients, and top predictors.  

3. **🚘 Per-Car Insights**  
   Visualize price trends and sentiment/feature impact for each car.  
