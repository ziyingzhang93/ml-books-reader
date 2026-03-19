import pandas as pd
import matplotlib.pyplot as plt

# Load time series
df = pd.read_csv("ad_viz_plotval_data.csv", parse_dates=[0])
print("Input data:")
print(df)

# Set date index
df_pm25 = df.set_index("Date")
print("\nUsing date index:")
print(df_pm25)
print(df_pm25.index)

# 2021 daily
df_2021 = ( df[["Date", "Daily Mean PM2.5 Concentration", "Site Name"]]
            .pivot_table(index="Date",
                         columns="Site Name",
                         values="Daily Mean PM2.5 Concentration")
          )
print("\nUsing date index:")
print(df_2021)
print(df_2021.index.is_unique)

# Time interval
df_3mon = df_2021["2021-04-01":"2021-07-01"]
print("\nInterval selection:")
print(df_3mon)

# Resample
print("\nResampling dataframe:")
df_resample = df_2021.resample("W-SUN").first()
print(df_resample)
print("\nResampling series for OHLC:")
df_ohlc = df_2021["San Antonio Interstate 35"].resample("W-SUN").ohlc()
print(df_ohlc)
print("\nResampling series with forward fill:")
series_ffill = df_2021["San Antonio Interstate 35"].resample("H").ffill()
print(series_ffill)

# rolling
print("\nRolling mean:")
df_mean = df_2021["San Antonio Interstate 35"].rolling(10).mean()
print(df_mean)

# Plot moving average
fig = plt.figure(figsize=(12,6))
plt.plot(df_2021["San Antonio Interstate 35"], label="daily")
plt.plot(df_2021["San Antonio Interstate 35"].rolling(10, min_periods=5).mean(),
         label="10-day MA")
plt.legend()
plt.ylabel("PM 2.5")
plt.show()
