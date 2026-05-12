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

    # The model was trained with features:
    # store_nbr, onpromotion, year, month, day, day_of_week, is_weekend,
    # lag_1, lag_7, rolling_mean_7
    # Note: 'family' was label-encoded during training. We need to handle it.
    # Since the pkl model may have family encoding built in via the training pipeline,
    # we include a family_encoded feature.
    return pd.DataFrame([features])


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
            ["🏠 Home", "🔮 Single Prediction", "📈 Batch Prediction",
             "📊 Model Performance", "ℹ️ About"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("### 🔧 Model Info")
        st.markdown("""
        - **Algorithm:** XGBoost
        - **RMSE:** 364.78
        - **MAE:** 114.02
        - **Training Data:** ~3M rows
        - **Features:** 10
        """)

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
            store_nbr = st.number_input("🏪 Store Number", min_value=1, max_value=54, value=1, step=1)
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
                        family="GROCERY I",  # placeholder
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

    # ========================
    # Batch Prediction
    # ========================
    elif page == "📈 Batch Prediction":
        st.markdown("### 📈 Batch Sales Prediction")
        st.markdown("Upload a CSV file with the required columns for batch predictions.")

        st.markdown("""
        <div class="info-box">
        <strong>📋 Required CSV Columns:</strong><br>
        <code>date</code>, <code>store_nbr</code>, <code>onpromotion</code>,
        <code>lag_1</code>, <code>lag_7</code>, <code>rolling_mean_7</code>
        </div>
        """, unsafe_allow_html=True)

        # Generate sample CSV
        if st.button("📥 Download Sample CSV"):
            sample_data = pd.DataFrame({
                'date': pd.date_range('2017-08-16', periods=10),
                'store_nbr': [1] * 10,
                'onpromotion': [0] * 10,
                'lag_1': np.random.uniform(100, 1000, 10).round(2),
                'lag_7': np.random.uniform(100, 1000, 10).round(2),
                'rolling_mean_7': np.random.uniform(100, 1000, 10).round(2),
            })
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download",
                data=csv,
                file_name="sample_prediction_input.csv",
                mime="text/csv"
            )

        uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown("#### 📋 Input Data Preview")
                st.dataframe(df.head(10), width='stretch')

                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        # Create features
                        df_features = create_features(df)

                        # Select model features
                        feature_cols = ['store_nbr', 'onpromotion', 'year', 'month',
                                       'day', 'day_of_week', 'is_weekend',
                                       'lag_1', 'lag_7', 'rolling_mean_7']

                        missing_cols = [c for c in feature_cols if c not in df_features.columns]
                        if missing_cols:
                            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                        else:
                            X = df_features[feature_cols]
                            predictions = model.predict(X)
                            predictions = np.maximum(0, predictions)

                            df_features['predicted_sales'] = predictions

                            st.markdown("#### 📊 Prediction Results")
                            st.dataframe(df_features[['date', 'store_nbr', 'predicted_sales']].head(20),
                                        width='stretch')

                            # Visualization
                            fig = px.line(df_features, x='date', y='predicted_sales',
                                         title='📈 Predicted Sales Over Time',
                                         labels={'predicted_sales': 'Predicted Sales', 'date': 'Date'})
                            fig.update_layout(template='plotly_white')
                            st.plotly_chart(fig, width='stretch')

                            # Download results
                            csv_result = df_features.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Predictions",
                                data=csv_result,
                                file_name="sales_predictions.csv",
                                mime="text/csv"
                            )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    # ========================
    # Model Performance
    # ========================
    elif page == "📊 Model Performance":
        st.markdown("### 📊 Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">364.78</div>
                <div class="metric-label">RMSE (Root Mean Squared Error)</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">114.02</div>
                <div class="metric-label">MAE (Mean Absolute Error)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div class="success-box">
        <strong>✅ XGBoost Performance</strong><br>
        The XGBoost model successfully captures the complex, non-linear relationships in store sales.
        It benefits heavily from tabular feature engineering (lag features, rolling means, temporal features) 
        which identify the underlying sales patterns at a highly granular store-family level.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔑 Key Model Details")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Algorithm | XGBoost Regressor |
        | Training Data Size | ~3,000,000 rows |
        | Number of Features | 10 |
        | Target Variable | Sales |
        | Seasonality Mode | Handled via feature engineering |
        | Lag Features | 1-day lag, 7-day lag |
        | Rolling Features | 7-day rolling mean |
        """)

    # ========================
    # About Page
    # ========================
    elif page == "ℹ️ About":
        st.markdown("### ℹ️ About This Project")
        st.markdown("""
        #### 📌 Project Description
        This project is a **Store Sales Time Series Forecasting** application that predicts
        the sales for thousands of products across multiple stores.

        #### 📊 Dataset
        - **Source:** Kaggle - Store Sales Time Series Forecasting
        - **Size:** ~3,000,888 rows × 6 columns
        - **Time Period:** 2013-01-01 to 2017-08-15
        - **Stores:** 54 stores
        - **Product Families:** 33 categories

        #### 🧠 Machine Learning Pipeline
        1. **Data Loading & Cleaning** - Handle missing values, parse dates
        2. **Feature Engineering** - Create temporal features, lag features, rolling statistics
        3. **Model Training** - Train XGBoost and Prophet models
        4. **Model Evaluation** - Compare using RMSE and MAE metrics
        5. **Model Deployment** - Serve via Streamlit web application

        #### 🛠️ Tech Stack
        - **Python** - Core programming language
        - **Pandas & NumPy** - Data manipulation
        - **XGBoost** - Primary ML model
        - **Scikit-learn** - Model evaluation metrics
        - **Streamlit** - Web application framework
        - **Plotly** - Interactive visualizations
        - **Pickle** - Model serialization

        #### 👨‍💻 Workflow
        ```
        Raw Data → Data Cleaning → Feature Engineering → Train/Test Split
            → Model Training (XGBoost) → Evaluation → Deployment (Streamlit)
        ```
        """)


if __name__ == "__main__":
    main()
