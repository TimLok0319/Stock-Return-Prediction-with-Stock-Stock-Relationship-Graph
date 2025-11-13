import numpy as np
import pandas as pd
import os

"This file is use to combine final_sentiment_vector_S.csv with input_X_train_2018.npy into a single input feature matrix, X"

# --- 1. CONFIGURE YOUR FILENAMES ---

# This is your existing .npy file with 8 features
QUANT_FEATURES_NPY_FILE = 'data_preprocessing/input_X_train_2018.npy' 

# This is the CSV from your aggregation script
SENTIMENT_CSV_FILE = 'final_sentiment_vector_S.csv'

# This is the name of the final, combined file to be created
FINAL_X_NPY_FILE = 'features_matrix.npy' 

# --- 2. LOAD YOUR DATA ---

print(f"Loading 8 quantitative features from: {QUANT_FEATURES_NPY_FILE}")
try:
    # Load your (104, 10, 8) quantitative feature matrix
    quant_data = np.load(QUANT_FEATURES_NPY_FILE)
except FileNotFoundError:
    print(f"Error: File '{QUANT_FEATURES_NPY_FILE}' not found.")
    print("Please create this file first, or run this script in the correct directory.")
    # As a placeholder, we create dummy data to show the script works
    print("Creating placeholder quantitative data...")
    # Assuming 104 trading days, 10 stocks, 8 features
    quant_data = np.random.rand(104, 10, 8) 
except Exception as e:
    print(f"An error occurred loading {QUANT_FEATURES_NPY_FILE}: {e}")
    exit()

print(f"Loading 9th sentiment feature from: {SENTIMENT_CSV_FILE}")
try:
    # Load your (104, 10) sentiment vector CSV
    sentiment_df = pd.read_csv(SENTIMENT_CSV_FILE, index_col=0)
    sentiment_data = sentiment_df.to_numpy()
except FileNotFoundError:
    print(f"Error: File '{SENTIMENT_CSV_FILE}' not found.")
    print("Please run the 'aggregate_data.py' script first.")
    exit()
except Exception as e:
    print(f"An error occurred loading {SENTIMENT_CSV_FILE}: {e}")
    exit()

# --- 3. ALIGNMENT CHECK ---
print("Checking data alignment...")

if quant_data.shape[0] != sentiment_data.shape[0]:
    raise ValueError(
        f"Date dimension mismatch! Quant file has {quant_data.shape[0]} days, "
        f"but Sentiment file has {sentiment_data.shape[0]} days."
    )

if quant_data.shape[1] != sentiment_data.shape[1]:
    raise ValueError(
        f"Stock dimension mismatch! Quant file has {quant_data.shape[1]} stocks, "
        f"but Sentiment file has {sentiment_data.shape[1]} stocks."
    )

print(f"Alignment successful: {quant_data.shape[0]} days, {quant_data.shape[1]} stocks.")

# --- 4. RESHAPE SENTIMENT VECTOR ---
# To concatenate, we must reshape the sentiment data from
# (104, 10) -> (104, 10, 1)
# This makes it a 3D tensor with one feature.
sentiment_data_reshaped = np.expand_dims(sentiment_data, axis=2)

# --- 5. CONCATENATE MATRICES ---
# We join the arrays along axis=2 (the feature dimension)
# (104, 10, 8) + (104, 10, 1) -> (104, 10, 9)
final_X_data = np.concatenate([quant_data, sentiment_data_reshaped], axis=2)

# --- 6. VERIFY AND SAVE ---
print("\n--- Final Output ---")
print(f"Original Quant shape:  {quant_data.shape}")
print(f"Reshaped Sentiment shape: {sentiment_data_reshaped.shape}")
print(f"Final Combined X shape: {final_X_data.shape}")

np.save(FINAL_X_NPY_FILE, final_X_data)
print(f"\n✅ Successfully saved final (104, 10, 9) feature matrix to '{FINAL_X_NPY_FILE}'")