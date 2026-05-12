# 🛒 Store Sales Demand Forecasting

A machine learning-powered application for predicting store sales using advanced time-series forecasting techniques and an **XGBoost regression model**, deployed with an interactive **Streamlit** dashboard.

## 🌟 Key Features

- **🔮 Single Prediction:** Real-time predictions for specific stores, dates, and product families.
- **📈 Batch Prediction:** Seamlessly upload CSV files for bulk sales forecasting.
- **📊 Model Comparison:** Direct performance comparison between machine learning (XGBoost) and traditional time-series (Prophet) approaches.
- **🧠 Advanced Feature Engineering:** Leverages lag features (`lag_1`, `lag_7`) and rolling statistics (`rolling_mean_7`) to capture temporal patterns.

## 🚀 Model Performance

During experimentation, we tested multiple approaches. The **XGBoost Regressor** vastly outperformed the Prophet model by successfully capturing complex tabular feature relationships.

| Model | RMSE | MAE |
|-------|------|-----|
| **XGBoost Regressor** | **364.78** | **114.02** |
| Prophet (Time Series) | 177,944.26 | 136,023.41 |

*Note: XGBoost benefits significantly from engineered features like 1-day/7-day lags and 7-day rolling means.*

## 🛠️ Tech Stack

- **Python:** Data processing and machine learning
- **XGBoost & Scikit-Learn:** Model building and evaluation
- **Pandas & NumPy:** Data manipulation and feature engineering
- **Streamlit:** Web application framework
- **Plotly:** Interactive data visualization

## 💻 How to Run Locally

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd Sales-Forecasting
   ```

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
   *(Note for Mac users: If you encounter XGBoost runtime errors, run `brew install libomp`)*

4. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
