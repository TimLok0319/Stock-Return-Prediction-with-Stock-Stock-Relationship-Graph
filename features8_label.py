import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import json 
import time
from curl_cffi import requests # Import the special requests

# --- HELPER CLASS ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# --- END OF HELPER CLASS ---


# --- 1. DEFINE PARAMETERS ---
stock_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
    'JPM', 'BRK-B', 'JNJ', 'WMT', 'XOM'
]
macro_tickers = ['^GSPC', '^TNX']
all_tickers = stock_tickers + macro_tickers

feature_columns = [
    'Daily_Return', 'Volatility', 'RSI', 'MACD', 'Norm_Volume',
    'GSPC_Return', 'TNX_Return', 'TNX_Yield'
]
N_DAYS = 5
DOWNLOAD_START = "2017-09-01" 
DOWNLOAD_END = "2018-06-15"   
FINAL_START = "2018-01-16"
FINAL_END = "2018-05-31"
DAY_TO_INSPECT = 0

print("--- Starting Data Build ---")

# --- 2. CREATE BROWSER IMPERSONATION SESSION ---
print("Creating Chrome browser session...")
session = requests.Session(impersonate="chrome")
session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'


# --- 3. GATHER ALL RAW DATA ---
print(f"Downloading {len(all_tickers)} tickers using browser session...")

try:
    data = yf.download(
        all_tickers, 
        start=DOWNLOAD_START, 
        end=DOWNLOAD_END, 
        session=session 
    )
    if data.empty:
        raise Exception("No data returned")
        
except Exception as e:
    print(f"!!! FAILED TO DOWNLOAD: {e}")
    print("Stopping script. Please check your connection or ticker symbol.")
    exit()

print("Download complete.")

# --- 4. ENGINEER LABELS (y) ---
print(f"Engineering {N_DAYS}-day future return label (y)...")
# FIX 1: Use 'Close' instead of 'Adj Close'
adj_close = data['Close'][stock_tickers]
future_price = adj_close.shift(-N_DAYS)
y_labels = (future_price - adj_close) / adj_close

# --- 5. ENGINEER MACRO FEATURES ---
print("Engineering macro features...")
macro_features = pd.DataFrame(index=data.index)
# FIX 2: Use 'Close'
macro_features['GSPC_Return'] = data['Close']['^GSPC'].pct_change()
# FIX 3: Use 'Close'
macro_features['TNX_Yield'] = data['Close']['^TNX']
# FIX 4: Use 'Close'
macro_features['TNX_Return'] = data['Close']['^TNX'].pct_change()

# --- 6. ENGINEER NODE FEATURES (Input X) ---
print(f"Engineering 8 features for {len(stock_tickers)} stocks...")
all_stock_dfs = [] 

for ticker in stock_tickers:
    df = pd.DataFrame(index=data.index)
    
    # FIX 5: Use 'Close'
    stock_price = data['Close'][ticker]
    stock_volume = data['Volume'][ticker]
    
    df['Daily_Return'] = stock_price.pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df['RSI'] = ta.rsi(stock_price, length=14)
    macd = ta.macd(stock_price)
    
    if macd is not None and 'MACD_12_26_9' in macd.columns:
        df['MACD'] = macd['MACD_12_26_9']
    else:
        df['MACD'] = 0 
    
    turnover = stock_volume * stock_price
    turnover_mean = turnover.rolling(window=60).mean()
    turnover_std = turnover.rolling(window=60).std()
    df['Norm_Volume'] = (turnover - turnover_mean) / turnover_std
    
    df = df.join(macro_features)
    final_df = df[feature_columns]
    all_stock_dfs.append(final_df)

# --- 7. ASSEMBLE, ALIGN, AND FILTER ---
print("Assembling and aligning X and y tensors...")

panel_X = pd.concat(all_stock_dfs, keys=stock_tickers, axis=1)

common_index = panel_X.dropna().index.intersection(y_labels.dropna().index)

panel_X_clean = panel_X.loc[common_index]
y_labels_clean = y_labels.loc[common_index]

panel_X_train = panel_X_clean.loc[FINAL_START:FINAL_END]
y_labels_train = y_labels_clean.loc[FINAL_START:FINAL_END]

print(f"Found {len(panel_X_train)} clean, aligned trading days.")

# --- 8. RESHAPE TENSORS (IN MEMORY) ---
X_tensor = np.stack(
    [panel_X_train.loc[:, (ticker, slice(None))].values for ticker in stock_tickers],
    axis=1 
)
y_tensor = y_labels_train.values

X_tensor = np.nan_to_num(X_tensor, nan=0.0)
y_tensor = np.nan_to_num(y_tensor, nan=0.0)

print(f"Final Input X shape: {X_tensor.shape}")
print(f"Final y Label shape: {y_tensor.shape}")

# --- 9. CREATE JSON VISUALIZATION FILE ---
print(f"--- Creating JSON snapshot for Day {DAY_TO_INSPECT} ---")

try:
    date_str = panel_X_train.index[DAY_TO_INSPECT].strftime('%Y-%m-%d')
    X_day = X_tensor[DAY_TO_INSPECT] 
    y_day = y_tensor[DAY_TO_INSPECT]
    
    output_json = {"date": date_str, "day_index": DAY_TO_INSPECT, "stocks": {}}
    
    for i, ticker in enumerate(stock_tickers):
        output_json["stocks"][ticker] = {
            "label_5_day_future_return": y_day[i],
            "features": {name: X_day[i, j] for j, name in enumerate(feature_columns)}
        }
        
    output_filename = f"snapshot_day_{DAY_TO_INSPECT}.json"
    
    output_json_serializable = json.loads(json.dumps(output_json, cls=NumpyEncoder))

    with open(output_filename, "w") as f:
        json.dump(output_json_serializable, f, indent=4)
    print(f"Successfully saved visualization file: {output_filename}")

except IndexError:
    print(f"ERROR: Could not inspect Day {DAY_TO_INSPECT}.")
    print(f"Your data only has {len(X_tensor)} days.")
except Exception as e:
    print(f"An error occurred during JSON creation: {e}")

# --- 10. SAVE .NPY FILES FOR MODEL ---
print("--- Saving .npy files for model training ---")

np.save("input_X_train_2018.npy", X_tensor)
np.save("y_labels_train_2018.npy", y_tensor)

panel_X_train.index.to_series().to_csv("train_dates_2018.csv")
with open("tensor_tickers.txt", "w") as f:
    f.write("\n".join(stock_tickers))

print("\n--- Success! ---")
print("Model files (2): input_X_train_2018.npy, y_labels_train_2018.npy")
print("Helper files (2): train_dates_2018.csv, tensor_tickers.txt")
print(f"Viz file (1): {output_filename}")