import yfinance as yf
import pandas as pd
import numpy as np
import time
from pandas.tseries.offsets import BDay
from curl_cffi import requests # Use the same browser session

# --- 1. DEFINE PARAMETERS ---

# This is the lookback window for the risk matrix, as per your plan
RISK_WINDOW = 60 # 60-day rolling covariance

# --- 2. LOAD HELPER FILES ---
# This guarantees your Sigma, X, and y tensors are all aligned.
print("Loading helper files...")
try:
    # Load the "master key" for dates
    dates_df = pd.read_csv("train_dates_2018.csv", header=0, index_col=0)
    dates_index = dates_df.iloc[:, 0].astype('datetime64[ns]') # Ensure correct type
    print(f"Loaded {len(dates_index)} trading dates from 'train_dates_2018.csv'.")
    
    # Load the "master key" for tickers
    tickers = [line.strip() for line in open("tensor_tickers.txt")]
    print(f"Loaded {len(tickers)} tickers from 'tensor_tickers.txt'.")

except FileNotFoundError:
    print("ERROR: 'train_dates_2018.csv' or 'tensor_tickers.txt' not found.")
    print("Please run your 'build_training_data.py' script first.")
    exit()

# --- 3. GET DATA FOR COVARIANCE ---
# We need data from *before* our first date (e.g., 2018-01-22)
# to calculate its 60-day covariance.
first_date = dates_index[0]
# Calculate download start date (60 business days + buffer)
download_start = first_date - BDay(RISK_WINDOW + 20) # ~4 months before
download_end = dates_index[-1] # The last date in our training set

print(f"Downloading price data from {download_start.date()} to {download_end.date()}...")

# Use the same browser session to avoid rate limits
session = requests.Session(impersonate="chrome")
session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'

data = yf.download(
    tickers, 
    start=download_start, 
    end=download_end, 
    session=session
)

# We only need 'Close' for this
close_prices = data['Close']

# Calculate daily returns (percent change) for all stocks
returns = close_prices.pct_change()

print("Data download and returns calculation complete.")

# --- 4. CREATE THE DYNAMIC SIGMA (Risk) TENSOR ---
# This list will hold each day's (10, 10) covariance matrix
all_risk_matrices = []

print(f"Calculating {len(dates_index)} daily covariance matrices...")

# Loop through every single date from your 'train_dates_2018.csv' file
for date in dates_index:
    
    # Define the 60-day window for *this* date
    end_window = date - BDay(1) # End one business day before
    start_window = date - BDay(RISK_WINDOW) # Start 60 business days before
    
    # Get the daily returns for this 60-day window
    window_returns = returns.loc[start_window:end_window]
    
    # Calculate the COVARIANCE matrix for this window
    # This is the key change: .cov() instead of .corr()
    cov_matrix = window_returns.cov()
    
    # Handle NaNs: If a stock didn't move, fill with 0
    cov_matrix.fillna(0, inplace=True)
    
    # Add the (10, 10) NumPy array to our list
    all_risk_matrices.append(cov_matrix.values)

# --- 5. STACK AND SAVE THE FINAL TENSOR ---

# Stack all the 2D matrices (one for each day) into a single 3D tensor
final_tensor_SIGMA = np.stack(all_risk_matrices, axis=0)

# Final check for any NaNs
final_tensor_SIGMA = np.nan_to_num(final_tensor_SIGMA, nan=0.0)

print(f"\nFinal Input SIGMA tensor shape: {final_tensor_SIGMA.shape}")

# --- 6. SAVE YOUR WORK ---
np.save("input_SIGMA_2018_tensor.npy", final_tensor_SIGMA)
print("\n--- Success! ---")
print("Saved: input_SIGMA_2018_tensor.npy (Your Risk Matrix for Stage 4)")