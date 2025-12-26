# ================================================
# 📌 Retail Sales Forecasting with Prophet
# ================================================

# Install dependencies in case not installed
# !pip install pandas matplotlib prophet

# --------------------------------
# STEP 1: IMPORT LIBRARIES
# --------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# --------------------------------
# STEP 2: LOAD DATA
# --------------------------------
df = pd.read_csv("sales.csv")
print("Dataset loaded successfully.")
print(df.head())
print(df.info())

# --------------------------------
# STEP 3: CLEAN DATA
# --------------------------------
# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Drop missing values if found
df = df.dropna()

# --------------------------------
# STEP 4: MONTHLY AGGREGATION
# --------------------------------
df['month'] = df['date'].dt.to_period('M')
monthly_sales = df.groupby('month')['sum_total'].sum().reset_index()
monthly_sales['month'] = monthly_sales['month'].dt.to_timestamp()

# Rename for Prophet
df_prophet = monthly_sales.rename(columns={"month":"ds", "sum_total":"y"})

print("\nPrepared dataset for Prophet:")
print(df_prophet.head())

# --------------------------------
# STEP 5: TRAIN MODEL
# --------------------------------
model = Prophet()
model.fit(df_prophet)

# --------------------------------
# STEP 6: FORECAST FUTURE
# --------------------------------
future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

print("\nForecast result:")
print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())

# --------------------------------
# STEP 7: VISUALIZE RESULTS
# --------------------------------
fig1 = model.plot(forecast)
plt.title("Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

fig2 = model.plot_components(forecast)
plt.show()

# --------------------------------
# STEP 8: EXPORT FORECAST
# --------------------------------
forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv("sales_forecast_output.csv", index=False)
print("\nForecast exported to: sales_forecast_output.csv")

print("\n🎉 Forecasting Complete!")
