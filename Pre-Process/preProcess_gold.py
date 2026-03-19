import pandas as pd
import numpy as np

# 1. Load your Excel file
# Replace 'price.xlsx' with your actual filename
file_path = r'D:\Gold Price Predictor\Dataset\prices.csv'
df = pd.read_csv(file_path)

# 2. Standardize the Date column
# This ensures VS Code treats the 'Date' column as actual time objects
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)

# 3. Split the data at the transition point (April 4, 2024)
# We isolate the 'messy' part to apply the spline specifically there
historical_segment = df[:'2024-04-03']
modern_segment = df['2024-04-04':]

# 4. Resample to Daily ('D')
# This forces VS Code to create rows for every missing day in your 2-month gaps
historical_resampled = historical_segment.resample('D').asfreq()

# 5. Apply Cubic Spline Interpolation
# This fulfills your objective of using advanced interpolation for alignment [cite: 49, 71]
historical_daily = historical_resampled.interpolate(method='cubic')

# 6. Recombine and Cleanup
# Merge the reconstructed history with your authentic daily data
final_df = pd.concat([historical_daily, modern_segment])

# Drop any NaNs that might exist before your first recorded price in 2007
final_df.dropna(inplace=True)

# 7. Calculate Log Returns (as per your methodology) [cite: 72, 105]
final_df['Log_Returns'] = np.log(final_df['traditional'] / final_df['traditional'].shift(1))

# 8. Remove Non-Trading Days (weekends, holidays, unchanged prices)
# Days where price = previous day's price produce Log_Return = 0, which is noise.
# These make up ~40% of the dataset and cause the model to learn "predict zero".
before_count = len(final_df)
final_df = final_df[final_df['Log_Returns'] != 0]
after_count = len(final_df)
print(f"Removed {before_count - after_count} non-trading days ({(before_count - after_count)/before_count*100:.1f}%)")
print(f"Remaining trading days: {after_count}")

# Save the cleaned daily data for your LSTM training
final_df.to_csv(r'D:\Gold Price Predictor\Dataset\cleaned_daily_gold.csv')
print("Successfully created cleaned_daily_gold.csv with trading days only.")