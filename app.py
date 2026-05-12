"""
🛒 Store Sales Demand Forecasting - Streamlit Web Application
==============================================================
A machine learning-powered application for predicting store sales
using XGBoost regression model.

Author: Sales Forecasting Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="🛒 Store Sales Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Custom CSS Styling
# ========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px;
        color: #1E88E5;
        font-size: 16px;
        font-weight: 600;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ========================
# Load Model
# ========================
@st.cache_resource
def load_model():
    """Load the pre-trained XGBoost model."""
    model_path = os.path.join(os.path.dirname(__file__), "model (1).pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

@st.cache_data
def load_stores():
    """Load store metadata for user-friendly dropdowns."""
    stores_path = os.path.join(os.path.dirname(__file__), "store-sales-time-series-forecasting", "stores.csv")
    try:
        stores_df = pd.read_csv(stores_path)
        stores_df['display_name'] = stores_df.apply(
            lambda x: f"Store #{x['store_nbr']} ({x['city']}, {x['state']})", axis=1
        )
        return dict(zip(stores_df['display_name'], stores_df['store_nbr']))
    except Exception:
        return {f"Store #{i}": i for i in range(1, 55)}


# ========================
# Feature Engineering
# ========================
def create_features(df):
    """Create time-series features from a date column."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return df


def prepare_prediction_features(store_nbr, family, date_val, onpromotion,
                                 lag_1, lag_7, rolling_mean_7):
    """Prepare feature vector for prediction."""
    date_val = pd.to_datetime(date_val)

    features = {
        'store_nbr': store_nbr,
        'onpromotion': onpromotion,
        'year': date_val.year,
        'month': date_val.month,
        'day': date_val.day,
        'day_of_week': date_val.dayofweek,
        'is_weekend': 1 if date_val.dayofweek >= 5 else 0,
        'lag_1': lag_1,
        'lag_7': lag_7,
        'rolling_mean_7': rolling_mean_7,
    }

    # Initialize all family dummies to 0
    for fam in PRODUCT_FAMILIES:
        if fam != "AUTOMOTIVE":
            features[f"family_{fam}"] = 0
            
    # Set the selected family to 1
    if family != "AUTOMOTIVE" and family in PRODUCT_FAMILIES:
        features[f"family_{family}"] = 1

    expected_cols = [
        'store_nbr', 'onpromotion', 'year', 'month', 'day', 'day_of_week', 'is_weekend', 
        'lag_1', 'lag_7', 'rolling_mean_7'
    ] + [f"family_{fam}" for fam in PRODUCT_FAMILIES if fam != "AUTOMOTIVE"]
    
    df = pd.DataFrame([features])
    return df[expected_cols]


# ========================
# Product Families
# ========================
PRODUCT_FAMILIES = [
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
    "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
    "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
    "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES",
    "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
    "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
    "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
    "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES",
    "SEAFOOD"
]


# ========================
# Main Application
# ========================
def main():
    # Header
    st.markdown('<div class="main-header">🛒 Store Sales Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict future store sales using Machine Learning (XGBoost)</div>', unsafe_allow_html=True)

    # Load model
    model = load_model()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/shopping-cart--v2.png", width=80)
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "🔮 Single Prediction"],
            label_visibility="collapsed"
        )

    # ========================
    # Home Page
    # ========================
    if page == "🏠 Home":
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">364.78</div>
                <div class="metric-label">RMSE Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">114.02</div>
                <div class="metric-label">MAE Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">~3M</div>
                <div class="metric-label">Training Samples</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This application uses a **pre-trained XGBoost model** to forecast store sales
        based on historical data from the Kaggle Store Sales - Time Series Forecasting competition.

        #### Key Features:
        - 🔮 **Single Prediction:** Predict sales for a specific store, date, and product family
        - 📈 **Batch Prediction:** Upload CSV data for bulk predictions
        - 📊 **Model Performance:** View model metrics and feature importance
        - 🧠 **Advanced ML:** Uses XGBoost with time-series feature engineering
        """)

        st.markdown("### 📋 Feature Engineering Pipeline")
        st.markdown("""
        The model uses the following engineered features:

        | Feature | Description |
        |---------|-------------|
        | `store_nbr` | Store number identifier |
        | `onpromotion` | Number of items on promotion |
        | `year` | Year extracted from date |
        | `month` | Month extracted from date |
        | `day` | Day extracted from date |
        | `day_of_week` | Day of week (0=Monday, 6=Sunday) |
        | `is_weekend` | Weekend flag (1 if Saturday/Sunday) |
        | `lag_1` | Sales value from 1 day ago |
        | `lag_7` | Sales value from 7 days ago |
        | `rolling_mean_7` | 7-day rolling average of sales |
        """)

    # ========================
    # Single Prediction
    # ========================
    elif page == "🔮 Single Prediction":
        st.markdown("### 🔮 Single Sales Prediction")
        st.markdown("Enter the details below to predict sales for a specific store and date.")

        col1, col2 = st.columns(2)

        with col1:
            stores_dict = load_stores()
            store_display = st.selectbox("🏪 Select Store Location", list(stores_dict.keys()))
            store_nbr = stores_dict[store_display]
            
            family = st.selectbox("📦 Product Family", PRODUCT_FAMILIES, index=PRODUCT_FAMILIES.index("GROCERY I"))
            prediction_date = st.date_input("📅 Prediction Date",
                                            value=datetime(2017, 8, 16),
                                            min_value=datetime(2013, 1, 1),
                                            max_value=datetime(2025, 12, 31))
            onpromotion = st.number_input("🏷️ Items on Promotion", min_value=0, max_value=1000, value=0)

        with col2:
            lag_1 = st.number_input("📊 Previous Day Sales (Lag-1)", min_value=0.0, value=500.0, step=10.0)
            lag_7 = st.number_input("📊 Sales 7 Days Ago (Lag-7)", min_value=0.0, value=480.0, step=10.0)
            rolling_mean_7 = st.number_input("📊 7-Day Rolling Mean", min_value=0.0, value=490.0, step=10.0)

        if st.button("🚀 Predict Sales", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    # Prepare features
                    features_df = prepare_prediction_features(
                        store_nbr=store_nbr,
                        family=family,
                        date_val=prediction_date,
                        onpromotion=onpromotion,
                        lag_1=lag_1,
                        lag_7=lag_7,
                        rolling_mean_7=rolling_mean_7
                    )

                    # Make prediction
                    prediction = model.predict(features_df)
                    predicted_sales = max(0, float(prediction[0]))

                    st.markdown("---")
                    st.markdown("### 📊 Prediction Result")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏪 Store", f"#{store_nbr}")
                    with col2:
                        st.metric("📅 Date", str(prediction_date))
                    with col3:
                        st.metric("💰 Predicted Sales", f"{predicted_sales:,.2f}")

                    # Visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=predicted_sales,
                        title={'text': "Predicted Sales Value"},
                        gauge={
                            'axis': {'range': [0, max(predicted_sales * 2, 2000)]},
                            'bar': {'color': "#1E88E5"},
                            'steps': [
                                {'range': [0, predicted_sales * 0.5], 'color': '#E3F2FD'},
                                {'range': [predicted_sales * 0.5, predicted_sales], 'color': '#BBDEFB'},
                                {'range': [predicted_sales, predicted_sales * 2], 'color': '#90CAF9'},
                            ],
                        }
                    ))
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, width='stretch')

                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.exception(e)

    


if __name__ == "__main__":
    main()
