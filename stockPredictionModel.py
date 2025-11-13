from STGAT import * #import from the model definition file
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Hyperparameter
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.006
NUM_STOCKS = 10
STOCKS_FEATURES = 9
GAT_OUT = 8
LSTM_HIDDEN = 16
PATIENCE = 10
WEIGHT_DECAY = 1e-3
LOOKBACK = 10

#Function for inpuit
def create_sliding_windows(X, A, y, lookback_period):
    
    total_days = X.shape[0]
    
    # We can't start making predictions until we have a full lookback period
    # e.g., if lookback=30, the first prediction is on day 30 (index 29)
    first_predict_day_index = lookback_period - 1
    
    X_windows = []
    A_windows = []
    y_labels = []

    # Loop from the first day we can make a prediction to the last day
    for i in range(first_predict_day_index, total_days):
        
        # --- 1. Find the window start and end ---
        # e.g., if i=29 (day 30), start=0, end=30. 
        # This gives us indices 0-29 (30 days)
        start_index = i - first_predict_day_index
        end_index = i + 1 # (slicing is exclusive, so i+1 includes index i)
        
        # --- 2. Slice the input sequences ---
        # These are the 30 days of history *before* the prediction
        X_window = X[start_index : end_index]
        A_window = A[start_index : end_index]
        
        # --- 3. Get the corresponding label ---
        # The label is the 'y' value for day 'i'
        y_label = y[i]
        
        # --- 4. Store the (Input, Label) pair ---
        X_windows.append(X_window)
        A_windows.append(A_window)
        y_labels.append(y_label)

    # Convert the lists of windows into single, large NumPy arrays
    return np.array(X_windows), np.array(A_windows), np.array(y_labels)

class StockDataset(Dataset):
    """
    A custom PyTorch Dataset for the spatio-temporal stock data.
    It takes the windowed NumPy arrays, converts them to tensors,
    and serves them up one sample at a time.
    """
    def __init__(self, X_windows, A_windows, y_labels):
        # Convert NumPy arrays to PyTorch tensors
        # We use .float() because neural networks work with floating-point numbers
        self.X = torch.tensor(X_windows, dtype=torch.float32)
        self.A = torch.tensor(A_windows, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of samples (e.g., 75)"""
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Gets one sample (one window and its label) by index.
        This is what the DataLoader calls.
        """
        return self.X[idx], self.A[idx], self.y[idx]
    
def load_trained_model(model_path):
    """
    Loads the saved model weights into a new model instance.
    """
    print(f"Loading model from {model_path}...")
    
    # 1. Re-create the empty 'shell' of your model
    model = SpatioTemporalGAT(
        num_nodes=NUM_STOCKS,
        node_features_in=STOCKS_FEATURES,
        gat_features_out=GAT_OUT,
        lstm_hidden_dim=LSTM_HIDDEN
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    print("Model loaded successfully.")
    return model

def get_all_data():
    """
    Loads all data and returns a loader for the *entire* dataset
    and the *actual* labels for all samples.
    """
    X_all = np.load("data/input/features_matrix_X.npy")
    A_all = np.load("data/input/relationship_matrix_A.npy")
    y_all = np.load("data/label/label.npy")
    
    # ---
    # We still need to scale our features just like in training!
    # This is critical for the model to understand the data.
    # ---
    X_reshaped = X_all.reshape(-1, X_all.shape[-1])
    # DANGER: We should use a *saved* scaler from training.
    # But for now, let's just re-fit to all data.
    # (This is a small "leak" but okay for visualization)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled_2d = scaler.fit_transform(X_reshaped)
    X_all = X_scaled_2d.reshape(X_all.shape)

    # ---
    
    X_seq, A_seq, y_seq = create_sliding_windows(
        X_all, 
        A_all, 
        y_all, 
        LOOKBACK
    )
    
    # --- NO SPLIT ---
    # We are using the *entire* sequence
    
    all_dataset = StockDataset(X_seq, A_seq, y_seq)
    all_loader = DataLoader(
        dataset=all_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False # Critically important to keep the order
    )
    
    # Return the loader and the *full* set of labels
    return all_loader, y_seq


def main():
    X_all = np.load("data/input/features_matrix_X.npy")
    A_all = np.load("data/input/relationship_matrix_A.npy")
    y_all = np.load("data/label/label.npy")

    X_reshaped = X_all.reshape(-1, X_all.shape[-1])
    scaler = StandardScaler()
    X_scaled_2d = scaler.fit_transform(X_reshaped)
    X_all = X_scaled_2d.reshape(X_all.shape)

    print(f"Loading input and label................")
    print(f"X shape: {X_all.shape}")
    print(f"A shape: {A_all.shape}")
    print(f"y shape: {y_all.shape}")
    print(f"Input and label loaded")
    print(f"--------------------------------------------------------------------------------")

    
    X_seq, A_seq, y_seq = create_sliding_windows(
        X_all, 
        A_all, 
        y_all, 
        LOOKBACK
    )


    #Train/Test Split (80/20)
    total_samples = X_seq.shape[0] # e.g., 75
    split_index = int(total_samples * 0.8) # e.g., 60

    X_train = X_seq[:split_index]
    A_train = A_seq[:split_index]
    y_train = y_seq[:split_index]

    X_test = X_seq[split_index:]
    A_test = A_seq[split_index:]
    y_test = y_seq[split_index:]

    train_dataset = StockDataset(X_train, A_train, y_train)
    test_dataset = StockDataset(X_test, A_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    model = SpatioTemporalGAT(
        num_nodes=NUM_STOCKS,
        node_features_in=STOCKS_FEATURES,
        gat_features_out=GAT_OUT,
        lstm_hidden_dim=LSTM_HIDDEN
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )

    print("\nStarting training loop...")

    #--------------------------------------Training Loop-----------------------------------------------#
    best_loss = 1000
    no_improve = 0
    epoch_progress_bar = tqdm(range(NUM_EPOCHS), desc="Epochs")
    train_loss_history = []
    test_loss_history = []

    for epoch in epoch_progress_bar:
        model.train()
        train_loss = 0

        for x_batch, a_batch, y_batch in train_loader:
            x_batch, a_batch, y_batch = x_batch.to(device), a_batch.to(device),y_batch.to(device)

            predictions = model(x_batch, a_batch)
            loss = criterion(predictions, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            

        avg_train_loss = train_loss / len(train_loader)
        epoch_progress_bar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        model.eval()
        test_loss = 0

        with torch.no_grad():

            for x_batch, a_batch, y_batch in test_loader:
                x_batch, a_batch, y_batch = x_batch.to(device), a_batch.to(device),y_batch.to(device)
                
                predictions = model(x_batch, a_batch)
                loss = criterion(predictions, y_batch)
                test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS} --- Train Loss: {avg_train_loss:.4f} --- Test Loss: {avg_test_loss:.4f}")

        #Early Stopping
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        
        else:
            no_improve += 1

        if no_improve >= PATIENCE :
            tqdm.write(f"  No improvement for {PATIENCE} epoch. Initiate early stopping at epoch {epoch+1} ")
            break

        train_loss_history.append(avg_train_loss)
        test_loss_history.append(avg_test_loss)

    print("Training finished.")
    print(f"--------------------------------------------------------------------------------")

    #--------------------------------------Loss Graph Printing-----------------------------------------------#
    print("Plotting loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(test_loss_history, label="Test Loss")
    plt.title("Model Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    
    plot_filename = "loss_curve.png"
    plt.savefig(plot_filename)
    print(f"Loss curve saved as {plot_filename}")

    #--------------------------------------Prediction Graph Printing-----------------------------------------------#
    model = load_trained_model("best_model.pth")
    all_loader, actual_labels = get_all_data()
    
    # --- 2. Get All Predictions ---
    all_predictions = []
    print("Running model on *all* data (train + test)...")
    
    with torch.no_grad():
        for x_batch, a_batch, y_batch in all_loader:
            x_batch = x_batch.to(device)
            a_batch = a_batch.to(device)
            predictions = model(x_batch, a_batch)
            all_predictions.append(predictions.cpu())
            
    predicted_labels = torch.cat(all_predictions, dim=0).numpy()
    
    print(f"Shape of actual labels: {actual_labels.shape}")
    print(f"Shape of predicted labels: {predicted_labels.shape}")
    np.save("predictions.npy", predicted_labels)
    print(f"Predictions saved to predictions.npy")
    
    # --- 3. Plot the Results ---
    
    # Calculate the split point for our visualization
    total_samples = actual_labels.shape[0] # e.g., 75
    split_index = int(total_samples * 0.8) # e.g., 60
    
    stock_to_plot = 0
    actuals = actual_labels[:, stock_to_plot]
    predictions = predicted_labels[:, stock_to_plot]
    
    
    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label="Actual Future Returns", color='blue', alpha=0.7)
    plt.plot(predictions, label="Predicted Returns (Model)", color='red', linestyle='--')
    
    plt.axvline(x=split_index, color='green', linestyle=':', label='Train/Test Split')
    
    plt.title(f"Prediction vs. Actual (Stock {stock_to_plot}) - FULL DATASET")
    plt.xlabel("Sample (Day)")
    plt.ylabel("Future Return (n-day)")
    plt.axhline(y=best_loss, color='red', linestyle='--', 
                label=f"Best Test Loss: {best_loss:.4f}")
    plt.ylim(top=plt.ylim()[1] * 1.1)
    
    plt.text(split_index/2, np.max(actuals), "<- Model trained on this data (Misleading)", 
             horizontalalignment='center', color='gray')
    plt.text(split_index + (total_samples - split_index)/2, np.max(actuals), 
             "Model tested on this data (Real) ->", 
             horizontalalignment='center', color='green')
             
    plt.legend()
    plt.grid(True)
    
    plot_filename = "prediction_vs_actual_ALL.png"
    plt.savefig(plot_filename)
    print(f"Full plot saved as {plot_filename}")
if __name__ == "__main__":
    main()

   

