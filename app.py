Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Rainfall Analysis & Forecast", layout="wide")

st.title("Rainfall Analysis & Forecast Dashboard (2015–2023)")

# Loading DataSet
uploaded = st.file_uploader("Upload rainfall CSV (columns: date,rainfall_mm)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV to begin or ensure it's named 'rainfall_2015_2023.csv'")
    st.stop()

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df['rainfall_mm'] = df['rainfall_mm'].interpolate(method='time')
df['month'] = df.index.month
df['year'] = df.index.year
df['rolling_7d'] = df['rainfall_mm'].rolling(7).mean()
df['rolling_30d'] = df['rainfall_mm'].rolling(30).mean()


# Data Overview
st.subheader("Data Sample & Info")
st.dataframe(df.head(10))
st.write(df.describe())


# Interactive Rainfall Trend
st.subheader("Daily Rainfall Trend + 30-Day Rolling Average")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['rainfall_mm'], mode='lines', name='Daily Rainfall', line=dict(color='blue', width=1), opacity=0.3))
fig.add_trace(go.Scatter(x=df.index, y=df['rolling_30d'], mode='lines', name='30-Day Rolling Avg', line=dict(color='red', width=3)))
fig.update_layout(xaxis_title="Date", yaxis_title="Rainfall (mm)", height=500)
st.plotly_chart(fig, use_container_width=True)


# Seasonal Decomposition
st.subheader("Seasonal Decomposition")
try:
    decomposition = seasonal_decompose(df['rainfall_mm'], model='additive', period=365)
    fig, axes = plt.subplots(4,1, figsize=(12,8), sharex=True)
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Decomposition failed: {e}")

st.markdown("""
**Insights:**
- Trend: Shows the overall long-term direction (mostly flat, as observed above).
- Seasonal: Shows a highly consistent annual cycle, peaking in the middle of the year (May-July )follwed by September -October and dipping at the start/end of the year(December- February).
- Residual: What is left after accounting for trend and seasonality (the unpredictable daily variability).
- Noise :Daily rainfall has irregular short-term variations.
""")

# Monthly Distribution
st.subheader("Monthly Rainfall Distribution")
fig = px.box(df.reset_index(), x='month', y='rainfall_mm', points="all",
             labels={'month':'Month','rainfall_mm':'Rainfall (mm)'}, title="Monthly Rainfall Boxplot")
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
**Insights:**
- Some months show higher spread, indicating unreliable rainfall patterns.
- Peak rainfall months show frequent extreme values.
- The boxplot clearly shows that Months 4 through 8 have the highest average rainfall and the largest extremes (outliers), indicating the wet season.
""")

# Heatmap: Month vs Year
st.subheader("Monthly Rainfall Heatmap (mm)")
monthly_totals = df.pivot_table(values='rainfall_mm', index='year', columns='month', aggfunc='sum')
fig = px.imshow(monthly_totals, text_auto=True, labels=dict(x='Month', y='Year', color='Rainfall (mm)'),
                title='Monthly Rainfall Heatmap')
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
**Insights:**
- The heatmap shows the total rainfall for each month/year combination. The darker blues consistently fall in the mid-year months, confirming the same seasonal pattern holds true across every year.
""")

# Monthly Aggregation & Lag Features
df_monthly = df['rainfall_mm'].resample('M').sum().to_frame(name='rainfall')
df_lags = df_monthly.copy()
for i in range(1,13):
    df_lags[f'lag_{i}'] = df_lags['rainfall'].shift(i)
df_lags.dropna(inplace=True)

train_size = int(len(df_lags)*0.8)
train, test = df_lags.iloc[:train_size], df_lags.iloc[train_size:]

X_train, y_train = train.drop('rainfall', axis=1), train['rainfall']
X_test, y_test = test.drop('rainfall', axis=1), test['rainfall']

def evaluate(y_true, y_pred, model_name):
    return {"Model": model_name,
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


# Train Models
st.subheader("Model Training & Comparison")

# ARIMA
try:
    arima_model = ARIMA(train['rainfall'], order=(5,1,0)).fit()
    arima_pred = pd.Series(arima_model.forecast(len(test)), index=test.index)
    arima_res = evaluate(y_test, arima_pred, 'ARIMA')
except Exception:
    arima_res = {"Model":"ARIMA","RMSE":1e9,"MAE":1e9,"R2":-999}

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = pd.Series(rf.predict(X_test), index=test.index)
rf_res = evaluate(y_test, rf_pred, 'Random Forest')

# Holt-Winters
hw = ExponentialSmoothing(train['rainfall'], seasonal='add', seasonal_periods=12).fit()
hw_pred = pd.Series(hw.forecast(len(test)), index=test.index)
hw_res = evaluate(y_test, hw_pred, 'Holt-Winters')

results = pd.DataFrame([arima_res, rf_res, hw_res])
st.dataframe(results)

best_model = results.sort_values('RMSE').iloc[0]['Model']
st.success(f"Best model based on RMSE: {best_model}")


# Forecasting
st.subheader("12-Month Forecast")
final_model = ExponentialSmoothing(df_monthly['rainfall'], seasonal='add', seasonal_periods=12).fit()
forecast_values = final_model.forecast(12)
forecast_dates = pd.date_range(df_monthly.index[-1]+pd.DateOffset(months=1), periods=12, freq='M')
forecast_series = pd.Series(forecast_values, index=forecast_dates)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly['rainfall'], mode='lines', name='Historical'))
fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines+markers', name='Forecast'))
fig.update_layout(xaxis_title='Date', yaxis_title='Rainfall (mm)', title='12-Month Rainfall Forecast', height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Insights:**
- The final 12-month forecast, based on the Holt-Winters model, predicts future monthly rainfall based on the consistent historical seasonality.
- Reliable Predictions: The forecast is highly reliable for predicting the timing and magnitude of the upcoming wet and dry seasons.
** Actionable Insight:** This forecast is most useful for operational planning, such as:

- Agriculture: Timing planting and harvesting based on predicted wet season start/peak.
- Water Management: Planning reservoir levels and conservation efforts ahead of the dry season.
- Infrastructure: Preparing for periods of high flow (peak wet season) to prevent flooding.
""")
