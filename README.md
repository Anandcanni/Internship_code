# Internship_code
This project predicts future diamond sales demand based on historical transaction data using Facebook Prophet. The model analyzes sales trends for different diamond shapes, performs data preprocessing, outlier treatment, and logarithmic transformation to improve forecasting accuracy.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# STEP 1: Load and prepare the dataset
df = pd.read_csv('Data/t_dmd_entry.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# STEP 2: Map shape codes to full shape names
shape_map = {
    'OV': 'Oval', 'RD': 'Round', 'BR': 'Round12', 'PR': 'Princess',
    'HS': 'Heart', 'EM': 'Emerald12', 'EC': 'Emerald', 'CU': 'Cushion',
    'CUS': 'Cushion', 'MQ': 'Marquise', 'PS': 'Pear', 'AS': 'Asscher',
    'TR': 'Trapezoid', 'TRAP': 'Trapezoid12', 'SH': 'Shield',
    'RAD': 'Radiant', 'POR': 'Portuguese', 'OMC': 'Old Mine Cut',
    'MOV': 'Modified Oval', 'HEX': 'Hexagon'
}
df['ShapeFull'] = df['Shape'].map(shape_map)

# STEP 3: Select shape to forecast
selected_shape = 'Round12'
filtered_df = df[df['ShapeFull'] == selected_shape].copy()

# STEP 4: Group by day and sum sales
daily_sales = filtered_df.groupby('InvoiceDate')['Sold'].sum().reset_index()
daily_sales = daily_sales.rename(columns={'InvoiceDate': 'ds', 'Sold': 'y'})
daily_sales = daily_sales.sort_values('ds')

# STEP 5: Clean outliers using rolling window
rolling_mean = daily_sales['y'].rolling(window=7, center=True).mean()
rolling_std = daily_sales['y'].rolling(window=7, center=True).std()

upper = rolling_mean + 2 * rolling_std
lower = rolling_mean - 2 * rolling_std

daily_sales['y_cleaned'] = daily_sales['y'].where(
    (daily_sales['y'] <= upper) & (daily_sales['y'] >= lower),
    rolling_mean
)
daily_sales['y_cleaned'].fillna(daily_sales['y'], inplace=True)

# STEP 6: Log transformation
daily_sales['y_log'] = np.log1p(daily_sales['y_cleaned'])

# STEP 7: Prophet Model with enhanced seasonality
model = Prophet(
    changepoint_prior_scale=0.1,
    seasonality_mode='additive',
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.add_seasonality(name='weekly', period=7, fourier_order=3)

# STEP 8: Fit model
full_data = daily_sales[['ds', 'y_log']].rename(columns={'y_log': 'y'})
model.fit(full_data)

# STEP 9: Forecast 90 days
future = model.make_future_dataframe(periods=90, freq='D')
forecast = model.predict(future)
forecast['yhat'] = np.expm1(forecast['yhat'])  # Reverse log1p

# STEP 10: Evaluation on last 90 days
eval_window = 90
actual = daily_sales[['ds', 'y_cleaned']].tail(eval_window).reset_index(drop=True)
predicted = forecast[['ds', 'yhat']][forecast['ds'].isin(actual['ds'])].reset_index(drop=True)

mae = mean_absolute_error(actual['y_cleaned'], predicted['yhat'])
rmse = np.sqrt(mean_squared_error(actual['y_cleaned'], predicted['yhat']))
mape = np.mean(np.abs((actual['y_cleaned'] - predicted['yhat']) / actual['y_cleaned'].clip(lower=1))) * 100

# STEP 11: Plot Forecast
plt.figure(figsize=(14, 6))
plt.plot(daily_sales['ds'], daily_sales['y'], label='Original Sales', alpha=0.4)
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast (Next 90 Days)', color='blue')
plt.axvline(x=daily_sales['ds'].max(), color='gray', linestyle='--', label='Forecast Start')
plt.title(f'\U0001F4C8 Diamond Sales Forecast for Shape: {selected_shape}')
plt.xlabel('Date')
plt.ylabel('Items Sold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 12: Plot Seasonality Components
model.plot_components(forecast)
plt.tight_layout()
plt.show()

# STEP 13: Print Forecast + Metrics
print("\n\U0001F52E Forecast for Next 90 Days (Last 15 Days Preview):\n")
print(forecast[['ds', 'yhat']].tail(15).to_string(index=False))

print("\n\U0001F4CA Evaluation on Last 90 Days of Actual Sales:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAPE = {mape:.2f}%")
