import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple, Set

"This file is use to combine the LLM scored file into input matrix .npy file"

# --- Configuration: Date Filters ---
PROJECT_START_DATE = '2018-01-01'
PROJECT_END_DATE = '2018-05-31' # Your target window

# --- Configuration: 10 Target Stocks (Final List) ---
# --- THIS IS THE UPDATED LIST IN YOUR SPECIFIED ORDER ---
TICKERS = [
    "apple", "microsoft", "google", "amazon", "nvidia", 
    "jpmorgan", "berkshire hathaway", "johnson & johnson", "walmart", "exxon"
]
N_STOCKS = len(TICKERS) # Now 10
TICKER_TO_INDEX = {ticker: i for i, ticker in enumerate(TICKERS)} 

# --- File Names ---
TRADING_DAY_FILE = 'data_preprocessing/trading_days_2018.csv' # Your CSV file containing trading dates
SINGLE_ENTITY_FILE = 'data_preprocessing/single_entity_results.json'
RELATIONSHIP_FILE = 'data_preprocessing/relationship_results.json'
OUTPUT_CSV = 'final_sentiment_vector_S.csv'
OUTPUT_NPY = 'relationship_matrix_A.npy'


# --- Helper Functions ---

def load_json_data(filename: str) -> List[Dict[str, Any]]:
    """Loads JSON data from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Input file '{filename}' not found. Returning empty data.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Please check file integrity.")
        return []

def load_trading_days(filename: str) -> Set[str]:
    """Reads a CSV of trading days and returns a set of YYYY-MM-DD strings."""
    try:
        df = pd.read_csv(filename)
        # Assumes the first column is the date column; adjust this name if necessary.
        date_column_name = df.columns[0]
        
        # Convert all dates to standard string format (YYYY-MM-DD) and return as a set for fast lookup
        trading_days = pd.to_datetime(df[date_column_name]).dt.strftime('%Y-%m-%d').tolist()
        
        print(f"Loaded {len(trading_days)} trading days from {filename}.")
        
        return set(trading_days)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Trading day file '{filename}' not found. Cannot proceed with forward-filling.")
        return set()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not parse trading day file: {e}")
        return set()

def is_trading_day(date_str: str, trading_day_set: Set[str]) -> bool:
    """Checks if a date is a trading day using the loaded set."""
    return date_str in trading_day_set


def aggregate_and_forward_fill(all_data: List[Dict[str, Any]], trading_days: Set[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Performs the core aggregation and forward-filling logic to align news to trading days.
    """
    if not all_data or not trading_days:
        # Return empty structures if critical data is missing
        empty_df = pd.DataFrame(index=[], columns=TICKERS)
        empty_np = np.empty((0, N_STOCKS, N_STOCKS))
        return empty_df, empty_np

    # --- Step 1: Prepare the Data ---
    df = pd.DataFrame(all_data)
    # Filter to only include the target project window and sort by date
    df = df[
        (df['published_date'] >= PROJECT_START_DATE) & 
        (df['published_date'] <= PROJECT_END_DATE)
    ].sort_values(by='published_date')

    # Create the full calendar range for iteration
    date_range = pd.date_range(start=PROJECT_START_DATE, end=PROJECT_END_DATE, freq='D')
    
    # --- Step 2: Initialize Accumulators and Final Storage ---
    daily_S_accumulator = {ticker: [] for ticker in TICKERS} 
    daily_A_accumulator = {} 
    
    final_S_data = [] #--Sentiment Vector
    final_A_matrices = []  #--Relationship matrix
    final_dates = [] 

    # --- Step 3: Iterate through ALL calendar days (Jan 1 to May 31) ---
    for date_obj in date_range:
        current_date = date_obj.strftime('%Y-%m-%d')
        day_data = df[df['published_date'] == current_date]
        
        # A. Accumulate Scores for the day (if any news exists)
        if not day_data.empty:
            for _, row in day_data.iterrows():
                score = row['sentiment_score']
                
                # Single-Entity Accumulation (S)
                if 'processed_company' in row:
                    company = row['processed_company']
                    if company in TICKERS:
                        daily_S_accumulator[company].append(score)
                        
                # Relationship Accumulation (A)
                if 'processed_pair' in row:
                    # Check if 'processed_pair' is valid before processing
                    if isinstance(row['processed_pair'], list) and len(row['processed_pair']) == 2:
                        pair = tuple(sorted(row['processed_pair']))
                        # We must check if the pair members are in our 10 target tickers
                        if pair[0] in TICKERS and pair[1] in TICKERS:
                            if pair not in daily_A_accumulator:
                                daily_A_accumulator[pair] = []
                            daily_A_accumulator[pair].append(score)
        
        # B. DUMP ACCUMULATORS ON TRADING DAY (Forward-Fill Logic)
        if is_trading_day(current_date, trading_days):
            
            # i. Build Final S Score (Node Feature Vector)
            # Calculate mean of all accumulated scores, filling 0.0 if no news since last trade day
            s_row = [np.mean(daily_S_accumulator[ticker]) if daily_S_accumulator[ticker] else 0.0 for ticker in TICKERS]
            
            # ii. Build Final A Matrix (Edge Feature)
            matrix_A = np.zeros((N_STOCKS, N_STOCKS))
            for pair, scores in daily_A_accumulator.items():
                if scores:
                    avg_score = np.mean(scores)
                    idx_a = TICKER_TO_INDEX.get(pair[0])
                    idx_b = TICKER_TO_INDEX.get(pair[1])
                    
                    if idx_a is not None and idx_b is not None:
                        matrix_A[idx_a, idx_b] = avg_score
                        matrix_A[idx_b, idx_a] = avg_score # Symmetrical fill

            # iii. Save the final data and RESET ACCUMULATORS for the next cycle
            final_S_data.append(s_row)
            final_A_matrices.append(matrix_A)
            final_dates.append(current_date)
            
            daily_S_accumulator = {ticker: [] for ticker in TICKERS}
            daily_A_accumulator = {}

    # --- Step 4: Final Consolidation ---
    S_vector_df = pd.DataFrame(final_S_data, index=final_dates, columns=TICKERS)
    A_tensor = np.stack(final_A_matrices, axis=0) if final_A_matrices else np.empty((0, N_STOCKS, N_STOCKS))
    
    return S_vector_df, A_tensor


# --- Main Execution ---
print("Starting aggregation script...")

# Load trading days (THE NEW CRITICAL INPUT)
trading_day_set = load_trading_days(TRADING_DAY_FILE)

if not trading_day_set:
    print("No trading days loaded. Exiting.")
    exit()

# Load LLM data from the two pipeline outputs
single_entity_data = load_json_data(SINGLE_ENTITY_FILE)
relationship_data = load_json_data(RELATIONSHIP_FILE)

# Combine both JSON outputs for unified processing
combined_data = single_entity_data + relationship_data

if not combined_data:
    print("No LLM data found. Exiting.")
    exit()

# 1. Aggregate and Forward-Fill
sentiment_vector_S, relationship_matrix_A_tensor = aggregate_and_forward_fill(combined_data, trading_day_set)


# --- Final Save ---
print("===================================================================")
print("✅ Step 3: Aggregation Complete. Data is aligned to Trading Days.")
print(f"Target Window: {PROJECT_START_DATE} to {PROJECT_END_DATE}")
print("===================================================================")

# Save the Sentiment Vector (S) as a CSV file
sentiment_vector_S.to_csv(OUTPUT_CSV)
print(f"-> Saved Sentiment Vector (S) to '{OUTPUT_CSV}'. Shape: {sentiment_vector_S.shape}")

# Save the Relationship Matrix (A) as a NumPy binary file
np.save(OUTPUT_NPY, relationship_matrix_A_tensor)
print(f"-> Saved Relationship Matrix (A) to '{OUTPUT_NPY}'. Shape: {relationship_matrix_A_tensor.shape}")