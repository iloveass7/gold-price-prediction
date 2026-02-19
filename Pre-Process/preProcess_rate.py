import pandas as pd
import numpy as np

# 1. Load your USD-BDT Excel file
# Replace 'usd_bdt_rate.xlsx' with your actual filename
file_path = r'D:\Gold Price Predictor\Dataset\export-Bangladesh-Official.csv'
df = pd.read_csv(file_path)
df = df[['Date', 'Value']]  # Keep only relevant columns

# 2. Standardize the Date column
# In your screenshot, 'Date' is likely the 5th column
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# 3. Extend data back to 2007
# The CSV starts from Aug 2008. We add a starting point at Jan 2007
# using the earliest known rate (~68.50 BDT/USD, which was stable then)
earliest_rate = df['Value'].iloc[0]
start_2007 = pd.DataFrame({'Value': [earliest_rate]}, index=pd.to_datetime(['2007-01-01']))
start_2007.index.name = 'Date'
df = pd.concat([start_2007, df])

# 4. Handle Duplicate Dates
# Currency data sometimes has multiple entries for one day. We take the daily mean.
df = df.resample('D').mean()

# 5. Fill the Gaps (The 2-month breaks)
# resample('D').asfreq() inserts NaN rows for every single missing day
df_daily = df.resample('D').asfreq()

# 6. Apply Cubic Spline Interpolation for gaps within real data,
# and forward-fill for the 2007 backfill period (rate was stable ~68.5)
df_daily['Value'] = df_daily['Value'].interpolate(method='linear')
df_daily['Value'] = df_daily['Value'].ffill().bfill()

# 6. Calculate Log Returns for Stationarity
# This ensures the model learns 'change' rather than absolute numbers
df_daily['Rate_Log_Returns'] = np.log(df_daily['Value'] / df_daily['Value'].shift(1))

# 7. Final Cleanup and Export
df_daily.dropna(inplace=True)
df_daily.to_csv(r'D:\Gold Price Predictor\Dataset\cleaned_daily_usd_bdt.csv')

print("Successfully created cleaned_daily_usd_bdt.csv")
print(df_daily.head())