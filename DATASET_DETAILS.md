# Gold Price Predictor: Dataset Details

This document provides a technical breakdown of the datasets used in this project, from raw collection to the final processed features.

---

## 1. Raw Gold Prices (`Dataset/prices.csv`)
This dataset contains the historical price of gold in Bangladesh, categorized by purity.

- **Columns**:
  - `date`: The date of the price record.
  - `k18`, `k21`, `k22`: Price per bhori for 18, 21, and 22 Karat gold respectively.
  - `traditional`: Price per bhori for traditional (Sanatan) gold.
- **Example**:
  ```csv
  date,k18,k21,k22,traditional
  2007-03-07,1326,1547,1620,1372
  2007-04-19,1398,1632,1709,1457
  ```

---

## 2. Raw Exchange Rates (`Dataset/export-Bangladesh-Official.csv`)
Historical USD to BDT exchange rates. Gold prices are highly sensitive to currency fluctuations as gold is traded globally in USD.

- **Columns**:
  - `Date`: Standardized date format.
  - `Value`: The exchange rate (how many BDT per 1 USD).
- **Example**:
  ```csv
  Date,Value
  04/08/2008,68.5
  07/08/2008,68.45
  ```

---

## 3. Master Training Data (`Dataset/master_training_data.csv`)
The final, processed dataset used for training the Machine Learning models. All values here are **Scaled** (using `StandardScaler`) and **Stationary** (using Log Returns).

### Key Features Explained:
- **`Log_Returns`**: The target variable. Calculated as $ln(Price_t / Price_{t-1})$. This represents the daily percentage change in gold price.
- **`Rate_Log_Returns`**: Daily change in the USD-BDT exchange rate.
- **`Daily_Inflation_Scaled`**: Interpolated daily inflation rate, normalized for the model.
- **`Wedding_Season_Flag`**: Binary feature (1 or 0) indicating high-demand months (Nov-Feb).
- **`Log_Returns_Lag1/2/3`**: The price changes from 1, 2, and 3 days ago. This helps the model identify short-term momentum.
- **`Rolling_Mean_7d/14d`**: The average price change over the last week/two weeks. Captures medium-term trends.
- **`Rolling_Volatility_7d`**: The standard deviation of price changes over the last 7 days. Represents market risk/stability.

### Example (Scaled Values):
| Date | Log_Returns | Rate_Log_Returns | Wedding_Season_Flag | Rolling_Mean_7d |
| :--- | :--- | :--- | :--- | :--- |
| 2007-03-21 | 0.076198 | -0.028747 | -0.700699 | 0.058393 |
| 2007-03-22 | 0.083315 | -0.028747 | -0.700699 | 0.066951 |

---

## Data Flow Summary
1. **Raw Data**: Scraped or collected into `prices.csv` and `export-Bangladesh-Official.csv`.
2. **Preprocessing**: 
   - Missing dates are filled using **Cubic Spline Interpolation**.
   - Absolute prices are converted to **Log Returns** to remove trends (making data stationary).
3. **Feature Engineering**: Lags and rolling windows are calculated.
4. **Standardization**: All features are scaled to a mean of 0 and standard deviation of 1, allowing the LSTM to converge efficiently.
