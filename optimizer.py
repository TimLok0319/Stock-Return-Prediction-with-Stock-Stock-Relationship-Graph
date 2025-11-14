import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import objective_functions
import os
import cvxpy as cp 

# --- 1. DEFINE PARAMETERS ---

# This is the name of the file from your partner.
OUTPUT_FILE = "predictions.npy" # [cite: predictions.npy]

# Your data tensors
RISK_MATRIX_FILE = "input_SIGMA_2018_tensor.npy" # [cite: input_SIGMA_2018_tensor.npy]
FEATURE_MATRIX_FILE = "input_X_train_2018.npy" # [cite: input_X_train_2018.npy]

# Helper files
DATES_FILE = "train_dates_2018.csv" # [cite: train_dates_2018.csv]
TICKERS_FILE = "tensor_tickers.txt" # [cite: tensor_tickers.txt]

# --- 2. LOAD ALL DATA ---
print("Loading all data for Stage 4...")
try:
    # Load your partner's REAL model output
    mu_tensor = np.load(OUTPUT_FILE) # (Shape must be 104, 10)
    
    # Load your Risk Matrix
    S_tensor = np.load(RISK_MATRIX_FILE) # (Shape 104, 10, 10)
    
    # Load your Feature Matrix (to get the risk-free rate)
    X_tensor = np.load(FEATURE_MATRIX_FILE) # (Shape 104, 10, 8)
    
    # Load helper files
    dates = pd.read_csv(DATES_FILE, header=0, index_col=0).iloc[:, 0]
    dates.index = pd.to_datetime(dates.index)
    tickers = [line.strip() for line in open(TICKERS_FILE)]

except FileNotFoundError as e:
    print(f"ERROR: Missing a file: {e}")
    print("Please make sure all your .npy files and helper files are here.")
    exit()

# --- VALIDATION STEP ---
if not (len(mu_tensor) == len(S_tensor) == len(X_tensor) == len(dates)):
    print("\n---!!! CRITICAL ERROR !!! ---")
    print(f"Your files are not aligned:")
    print(f"  Dates:       {len(dates)} samples")
    print(f"  Input X:     {len(X_tensor)} samples")
    print(f"  Input Sigma: {len(S_tensor)} samples")
    print(f"  Predictions: {len(mu_tensor)} samples  <-- THIS IS THE PROBLEM")
    print("\nTell your partner to regenerate 'predictions.npy' with the correct shape.")
    exit()

print("Data load complete and all files are aligned.")

# --- 3. RUN THE DAILY OPTIMIZATION LOOP ---
all_weights = []
all_dates = []

print(f"Running optimizer for {len(dates)} trading days...")

for i in range(len(dates)):
    # Get the data for this one day
    date = dates.index[i]
    
    # Get the 10x1 predicted returns (mu) for this day
    mu = pd.Series(mu_tensor[i], index=tickers)
    
    # Get the 10x10 risk matrix (S) for this day
    S = pd.DataFrame(S_tensor[i], index=tickers, columns=tickers)
    
    # --- DYNAMIC RISK-FREE RATE ---
    X_day_features = X_tensor[i] 
    todays_risk_free_rate = X_day_features[0, 7] / 100.0
    
    # --- This is the core of Stage 4 ---
    try:
        # 1. Create the optimizer object
        ef = EfficientFrontier(mu, S)

        # 2. (Optional) Add constraints. 
        ef.add_constraint(lambda w: w <= 0.3) # Max 30% in any stock
        ef.add_constraint(lambda w: w >= -0.3) # Max 30% short
        ef.add_constraint(lambda w: cp.sum(cp.abs(w)) <= 1.0) # Net leverage = 1

        # 3. Find the portfolio that maximizes the Sharpe Ratio
        weights = ef.max_sharpe(risk_free_rate=todays_risk_free_rate)
        
        # 4. Clean the weights (removes tiny values)
        cleaned_weights = ef.clean_weights()

    except Exception as e:
        # If optimizer fails (e.g., matrix not invertible), hold 0%
        print(f"  - WARNING: Optimizer failed on {date.date()}: {e}")
        cleaned_weights = {ticker: 0.0 for ticker in tickers}
    
    all_weights.append(cleaned_weights)
    all_dates.append(date)

print("Optimization loop complete.")

# --- 4. SAVE YOUR FINAL OUTPUT ---
# This DataFrame is your final "decision" for every day
weights_df = pd.DataFrame(all_weights, index=all_dates)

# Save it to a CSV. This file is the input for Stage 5 (Backtesting)
try:
    weights_df.to_csv("portfolio_weights_2018.csv")
    print("\n--- Success! ---")
    print("Saved daily portfolio weights to: portfolio_weights_2018.csv")
    print("\n--- Example Weights (First 5 Days) ---")
    print(weights_df.head())

except PermissionError:
    print("\n---!!! FAILED TO SAVE !!! ---")
    print("PERMISSION ERROR: Please CLOSE 'portfolio_weights_2018.csv' in Excel and run the script again.")
    print("-------------------------------")