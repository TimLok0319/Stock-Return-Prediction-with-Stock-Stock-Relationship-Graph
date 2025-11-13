import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


class SpatioTemporalGAT(nn.Module):
    """
    A Spatio-Temporal GAT model for stock prediction based on the project methodology.
    
    This model combines:
    1. A GAT layer to process "spatial" graph relationships (Stage 2)
    2. An LSTM layer to process "temporal" sequences (Stage 2)
    3. A final Dense layer for regression output (Stage 3)
    """
    def __init__(self, num_nodes, node_features_in, gat_features_out, lstm_hidden_dim):
        """
        Initialize the 3-layer model.
        
        Args:
            num_nodes (int): Your number of stocks (e.g., 10).
            node_features_in (int): Your number of input features (e.g., 9).
            gat_features_out (int): The hidden dimension of the GAT layer (e.g., 16).
            lstm_hidden_dim (int): The hidden dimension of the LSTM (e.g., 64).
        """
        super(SpatioTemporalGAT, self).__init__()
        
        self.num_nodes = num_nodes
        self.gat_features_out = gat_features_out
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # ---
        # Layer 1: GAT (Graph Attention Network)
        # heads=1 means we have one attention mechanism
        #
        # This is the "explainable" engine.
        # ---
        self.gat_layer = GATConv(
            in_channels=node_features_in,
            out_channels=gat_features_out,
            heads=1,
            concat=False, # False = averages attention heads, True = concatenates
            dropout=0.2
        )
        
        # ---
        # Layer 2: Temporal Layer (LSTM)
        #
        # The input size is (num_nodes * gat_features_out) because we flatten
        # the (10, 16) enhanced feature matrix from the GAT into a
        # single (160,) vector for each day.
        # ---
        self.lstm_layer = nn.LSTM(
            input_size=num_nodes * gat_features_out,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True # Expects input shape (Batch_Size, Seq_Len, Features)
        )
        
        # ---
        # Layer 3: Prediction Output (Dense Layer)
        #
        # Takes the final LSTM output and maps it to 10 stock predictions.
        # It has a linear activation by default.
        # ---
        self.output_layer = nn.Linear(lstm_hidden_dim, num_nodes) #

    
    def forward(self, x, adj, return_explanation=False):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The Node Feature Matrix (Input X).
                Shape: (Batch_Size, Lookback_Period, Num_Nodes, Num_Features)
                e.g., (1, 30, 10, 9)
                
            adj (torch.Tensor): The Dynamic Relationship Matrix (Input A).
                Shape: (Batch_Size, Lookback_Period, Num_Nodes, Num_Nodes)
                e.g., (1, 30, 10, 10)
                
            return_explanation (bool): Set to True to extract the GAT
                attention weights for Stage 6.
        
        Returns:
            torch.Tensor: The model's prediction, shape (Batch_Size, Num_Nodes).
            (optional) tuple: The explanation (attention_weights) for the *last day*
                             in the sequence.
        """
        
        batch_size, lookback_period, _, _ = x.shape
        
        # This list will store the GAT's output for each day in the sequence
        gat_sequence_outputs = []
        
        # This will store the explanation from the *last day* if requested
        last_day_attention = None

        # ---
        # 1. GAT Layer: Process each day in the sequence
        # We loop through the "lookback_period" (e.g., 30 days)
        # ---
        for t in range(lookback_period):
            # Get the features and graph for this one day
            x_t = x[:, t, :, :]       # Shape: (Batch_Size, 10, 9)
            adj_t = adj[:, t, :, :]     # Shape: (Batch_Size, 10, 10)
            
            # GATConv requires an `edge_index` (sparse format) not a dense matrix.
            # We convert our (10, 10) matrix A_t into this format.
            # We also get `edge_weight`, which GATConv can use to scale attention.
            # We assume a non-batched conversion for clarity, processing 1 graph at a time.
            
            # Note: This loop processes one item in the batch at a time.
            # A batched implementation is more complex but faster.
            enhanced_features_batch = []
            
            for b in range(batch_size):
                # Convert the (10, 10) adjacency matrix to sparse (2, E) edge_index
                edge_index_t, edge_weight_t = dense_to_sparse(adj_t[b])
                
                # ---
                # THE GAT LAYER CALL + EXPLANATION EXTRACTION
                # ---
                # We ask the GAT layer to return its internal alpha weights
                enhanced_features_t, attention_tuple = self.gat_layer(
                    x[b, t],                     # Node features (10, 9)
                    edge_index_t,                # Graph structure (2, E)
                    return_attention_weights=True # The magic flag!
                )
                
                enhanced_features_batch.append(enhanced_features_t)
                
                # If this is the LAST day of the sequence AND we want the explanation
                if t == (lookback_period - 1) and return_explanation:
                    # attention_tuple is (edge_index, alpha_scores)
                    # We save this for the final output
                    last_day_attention = attention_tuple 

            # Stack the batch results for this day
            gat_day_output = torch.stack(enhanced_features_batch) # (Batch_Size, 10, gat_features_out)
            gat_sequence_outputs.append(gat_day_output)

        # At this point, `gat_sequence_outputs` is a list of 30 tensors,
        # each of shape (Batch_Size, 10, gat_features_out)

        # Stack all day outputs to create the full sequence for the LSTM
        # Shape: (Batch_Size, 30, 10, gat_features_out)
        lstm_input_sequence = torch.stack(gat_sequence_outputs, dim=1)
        
        # ---
        # 2. LSTM Layer: Process the full sequence
        # ---
        # We flatten the (10, gat_features_out) into a single vector
        # Input shape becomes: (Batch_Size, 30, 10 * gat_features_out)
        lstm_input_flattened = lstm_input_sequence.view(
            batch_size, 
            lookback_period, 
            -1
        )
        
        # The LSTM processes the sequence
        # lstm_out shape: (Batch_Size, Seq_Len, lstm_hidden_dim)
        lstm_out, (hidden_state, cell_state) = self.lstm_layer(lstm_input_flattened)
        
        # We only care about the output from the *very last time step*
        last_time_step_output = lstm_out[:, -1, :] # Shape: (Batch_Size, lstm_hidden_dim)

        # ---
        # 3. Output Layer: Make the final prediction
        # ---
        # Feed the LSTM's final output to the Dense layer
        prediction = self.output_layer(last_time_step_output) # Shape: (Batch_Size, 10)
        
        if return_explanation:
            return prediction, last_day_attention
        else:
            return prediction