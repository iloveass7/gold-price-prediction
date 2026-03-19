import pandas as pd
import numpy as np

# 1. Load the inflation data
file_path = r'D:\Gold Price Predictor\Dataset\pattern_INFLATION_2.csv'
df = pd.read_csv(file_path)

# 2. Extract Year and Inflation Rate
# The column containing inflation values is the last column
df = df[['Year', df.columns[-1]]].copy()
df.columns = ['Year', 'Inflation_Rate']
df['Inflation_Rate'] = pd.to_numeric(df['Inflation_Rate'], errors='coerce')
df.dropna(inplace=True)

# 3. Create a date index (Jan 1 of each year)
df['Date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-01-01')
df.set_index('Date', inplace=True)
df = df[['Inflation_Rate']]

print(f"Annual inflation data: {len(df)} years")
print(f"Range: {df.index.min().date()} to {df.index.max().date()}")

# 4. Resample to daily frequency and interpolate
# This converts annual inflation to a smooth daily series
daily_df = df.resample('D').asfreq()
daily_df['Inflation_Rate'] = daily_df['Inflation_Rate'].interpolate(method='cubic')
daily_df['Inflation_Rate'] = daily_df['Inflation_Rate'].ffill().bfill()

# 5. Scale to a daily magnitude
# Annual inflation ~10% means ~0.027% per day (10/365)
# We scale it so the LSTM sees reasonable daily-scale values
daily_df['Daily_Inflation_Scaled'] = daily_df['Inflation_Rate'] / 100

# 6. Save
daily_df.to_csv(r'D:\Gold Price Predictor\Dataset\cleaned_daily_inflation.csv')
print(f"Created cleaned_daily_inflation.csv with {len(daily_df)} daily entries")
print(daily_df.head())
