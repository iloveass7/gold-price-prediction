import pandas as pd
import numpy as np

# 1. Load the cleaned datasets
# Ensure Date is parsed correctly and set as the index
gold_df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\cleaned_daily_gold.csv', parse_dates=['date'], index_col='date')
gold_df.index.name = 'Date'  # Standardize index name
usd_df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\cleaned_daily_usd_bdt.csv', parse_dates=['Date'], index_col='Date')
inf_df = pd.read_csv(r'D:\Gold Price Predictor\Dataset\cleaned_daily_inflation.csv', parse_dates=[0], index_col=0)
inf_df.index.name = 'Date'  # Standardize index name

# 2. Determine the starting point
# As per your requirement, we find the first date where gold prices exist
gold_start_date = gold_df.index.min()
print(f"Aligning dataset to start date: {gold_start_date.date()}")

# 3. Synchronize and Merge
# We use an inner join to ensure every date has all four features
# We then filter to ensure we start exactly where your gold data begins
merged_df = gold_df.join([usd_df, inf_df], how='inner')
merged_df = merged_df[merged_df.index >= gold_start_date]

# 4. Engineer the Wedding Season Flag (Nov-Feb)
# This captures the localized demand shocks mentioned in your methodology
def is_wedding_season(date):
    if date.month in [11, 12, 1, 2]:
        return 1
    return 0

merged_df['Wedding_Season_Flag'] = merged_df.index.to_series().apply(is_wedding_season)

# 5. Select Final Features for the Multivariate LSTM
# We use Log Returns for prices and rates to ensure stationarity
final_features = [
    'Log_Returns',            # Gold price volatility
    'Rate_Log_Returns',       # Currency exchange volatility
    'Daily_Inflation_Scaled', # Daily interpolated inflation
    'Wedding_Season_Flag'     # Seasonal demand shock
]

training_df = merged_df[final_features].copy()

# 6. Final Data Integrity Check
# Drop any remaining NaNs (usually the very first row due to Log Returns)
training_df.dropna(inplace=True)

# 7. Export the Master Dataset
training_df.to_csv(r'D:\Gold Price Predictor\Dataset\master_training_data.csv')

print("Successfully created master_training_data.csv")
print(f"Total entries from {training_df.index.min().date()} to {training_df.index.max().date()}: {len(training_df)}")