import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import re
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
import math
from math import sqrt
import pandas as pd
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

# Set seeds for reproducibility
SEED = 99
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fixed validation sample indices for consistent comparisons
VALIDATION_SAMPLE_INDICES = [5, 15, 25, 35, 45]

# Model Architectures
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.0, output_size=None):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        # Reshape input for LSTM: [batch, features] -> [batch, features, 1]
        x = x.unsqueeze(2)
        
        # Apply LSTM layers with dropout
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Apply fully connected layer
        x = self.fc(x)
        return x 

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.0, output_size=None):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size=1, hidden_size=hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        # Reshape input for GRU: [batch, features] -> [batch, features, 1]
        x = x.unsqueeze(2)
        
        # Apply GRU layers with dropout
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Apply fully connected layer
        x = self.fc(x)
        return x

# Informer model components
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, u, U_part)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.transpose(2, 1).contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(x, x, x)[0])
        new_x, attn = self.attention(
            x, x, x,
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class InformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, n_heads=8, e_layers=3, d_ff=256, dropout=0.0, output_size=None):
        super(InformerModel, self).__init__()
        
        # Processing input: expand to match d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(False, 5, attention_dropout=dropout, output_attention=False),
                    d_model, n_heads),
                d_model,
                d_ff,
                dropout=dropout,
                activation="gelu"
            ) for _ in range(e_layers)
        ])
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # Reshape input: [batch, features] -> [batch, features, 1]
        x = x.unsqueeze(2)
        
        # Apply embedding
        x = self.embedding(x)
        
        # Add positional embedding
        x += self.pos_emb(x)
        
        # Apply Informer encoder layers
        attns = []
        for enc_layer in self.encoder_layers:
            x, attn = enc_layer(x)
            attns.append(attn)
        
        # Final processing
        x = self.layer_norm(x)
        
        # Average pooling across the sequence dimension
        x = torch.mean(x, dim=1)
        
        # Apply final linear layer
        x = self.fc(x)
        
        return x

# Function to create a model based on architecture choice
def create_model(model_type, input_size, output_size, hidden_size1=128, hidden_size2=64, dropout=0.2):
    model_classes = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'informer': InformerModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}")
    
    model_class = model_classes[model_type]
    
    if model_type == 'informer':
        model = model_class(
            input_size=input_size,
            d_model=hidden_size1,
            n_heads=4,
            e_layers=2,
            d_ff=hidden_size2*4,
            dropout=dropout,
            output_size=output_size
        )
    else:
        model = model_class(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            dropout=dropout,
            output_size=output_size
        )
    
    return model

# Function to read EDI file and extract frequencies, apparent resistivity, and phase
def read_edi_file(edi_file):
    with open(edi_file, 'r') as f:
        content = f.read()
    
    # Extract frequencies
    freq_match = re.search(r'>FREQ //(\d+)(.*?)>!', content, re.DOTALL)
    if freq_match:
        num_freqs = int(freq_match.group(1))
        freq_text = freq_match.group(2)
        frequencies = np.array([float(x) for x in freq_text.split() if x.strip()])
        periods = 1.0 / frequencies
        periods = np.sort(periods)  # Sort periods in ascending order
    else:
        raise ValueError("Could not extract frequencies from EDI file")
    
    # Extract apparent resistivity components
    zxyr_match = re.search(r'>ZXYR ROT=ZROT //(\d+)(.*?)>', content, re.DOTALL)
    if zxyr_match:
        zxyr_text = zxyr_match.group(2)
        zxyr = np.array([float(x) for x in zxyr_text.split() if x.strip()])
    else:
        raise ValueError("Could not extract ZXYR from EDI file")
    
    zxyi_match = re.search(r'>ZXYI ROT=ZROT //(\d+)(.*?)>', content, re.DOTALL)
    if zxyi_match:
        zxyi_text = zxyi_match.group(2)
        zxyi = np.array([float(x) for x in zxyi_text.split() if x.strip()])
    else:
        raise ValueError("Could not extract ZXYI from EDI file")
    
    # Calculate phase and apparent resistivity
    mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability of free space
    phase = np.arctan2(zxyi, zxyr) * 180 / np.pi
    rho_apparent = (zxyr**2 + zxyi**2) / (2 * np.pi * mu_0 * frequencies)
    
    # Sort data by periods
    idx = np.argsort(periods)
    periods = periods[idx]
    rho_apparent = rho_apparent[idx]
    phase = phase[idx]
    
    return periods, rho_apparent, phase

# Function to read layered model data
def read_layered_model(layered_file):
    with open(layered_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines and read the data
    data_start = 5  # Typically line 6 (index 5) is where data starts
    depths = []
    thicknesses = []
    resistivities = []
    
    for line in lines[data_start:]:
        if line.strip():
            parts = line.strip().split()
            depths.append(float(parts[0]))
            thicknesses.append(float(parts[1]))
            resistivities.append(float(parts[2]))
    
    return np.array(depths), np.array(thicknesses), np.array(resistivities)

# Generate synthetic data based on parameters from real data
def generate_synthetic_data(num_samples=5000, num_periods=25, num_layers=50, seed=SEED):
    # Set seed for this function
    np.random.seed(seed)
    
    print("Reading real data to derive parameters...")
    
    # Read real data to derive parameters
    edi_files = [f for f in os.listdir('real_data') if f.endswith('.edi')]
    all_periods = []
    all_rho_app = []
    all_phase = []
    all_resistivities = []
    
    for edi_file in edi_files:
        site_name = edi_file.split('.')[0]
        layered_file = f"real_data/{site_name}Layered.txt"
        
        if os.path.exists(layered_file):
            periods, rho_app, phase = read_edi_file(f"real_data/{edi_file}")
            _, _, resistivities = read_layered_model(layered_file)
            
            all_periods.append(periods)
            all_rho_app.append(rho_app)
            all_phase.append(phase)
            all_resistivities.append(resistivities)
    
    # Define log-spaced periods for synthetic data
    log_periods = np.logspace(-3, 3, num_periods)
    
    # Calculate mean and std of log-resistivities from real data
    log_resistivities = [np.log10(res) for res in all_resistivities]
    mean_log_res = np.mean([np.mean(lr) for lr in log_resistivities])
    std_log_res = np.mean([np.std(lr) for lr in log_resistivities])
    
    min_log_res = min([np.min(lr) for lr in log_resistivities])
    max_log_res = max([np.max(lr) for lr in log_resistivities])
    
    print(f"Real data log-resistivity statistics:")
    print(f"  Mean: {mean_log_res:.2f}, Std: {std_log_res:.2f}")
    print(f"  Min: {min_log_res:.2f}, Max: {max_log_res:.2f}")
    
    # Generate layer thicknesses based on skin depth formula
    # Skin depth = 500 * sqrt(rho * T)
    rho_ref = 10**mean_log_res
    min_period = log_periods.min()
    max_period = log_periods.max()
    min_skin_depth = 500 * np.sqrt(rho_ref * min_period) / 4
    max_skin_depth = 500 * np.sqrt(rho_ref * max_period)
    
    print(f"Reference resistivity for layer generation: {rho_ref:.2f} Ohm-m")
    print(f"Minimum skin depth: {min_skin_depth:.2f} m")
    print(f"Maximum skin depth: {max_skin_depth:.2f} m")
    
    # Generate layer thicknesses with logarithmic spacing
    layer_thicknesses = [min_skin_depth]
    while layer_thicknesses[-1] * 1.2 < max_skin_depth and len(layer_thicknesses) < num_layers - 1:
        layer_thicknesses.append(layer_thicknesses[-1] * 1.2)
    layer_thicknesses = np.array(layer_thicknesses)
    
    print(f"Generated {len(layer_thicknesses)} layers with thicknesses from {layer_thicknesses[0]:.2f}m to {layer_thicknesses[-1]:.2f}m")
    
    # Function to generate random resistivity model
    def generate_random_model():
        # Number of layers to generate (one more than the number of thicknesses)
        n_layers = len(layer_thicknesses) + 1
        
        # Generate logarithmically distributed resistivities
        log_res = np.random.normal(mean_log_res, std_log_res * 1.5, n_layers)
        
        # Ensure resistivities are within realistic bounds based on real data
        log_res = np.clip(log_res, min_log_res * 0.8, max_log_res * 1.2)
        
        # Ensure some correlation between adjacent layers (smoothness constraint)
        for i in range(1, len(log_res)):
            log_res[i] = log_res[i-1] * 0.6 + log_res[i] * 0.4
        
        resistivity = 10 ** log_res
        return resistivity
    
    # Function to compute forward response (apparent resistivity and phase)
    def forward_model(resistivity_model):
        # In a real scenario, you would use a proper MT forward modeling code here
        # For this simplified example, we'll create a synthetic response with:
        # 1. Smooth variations of apparent resistivity related to the resistivity model
        # 2. Phase values that are physically consistent with the resistivity
        
        # Compute cumulative depths
        depths = np.cumsum(layer_thicknesses)
        
        # Compute forward response for each period
        rho_app = np.zeros(len(log_periods))
        phase = np.zeros(len(log_periods))
        
        # Ensure resistivity_model has the correct length
        if len(resistivity_model) > len(layer_thicknesses) + 1:
            # Truncate extra values if model is longer than needed
            resistivity_model = resistivity_model[:len(layer_thicknesses) + 1]
        elif len(resistivity_model) < len(layer_thicknesses) + 1:
            # Pad with last value if model is shorter than needed
            padding = np.ones(len(layer_thicknesses) + 1 - len(resistivity_model)) * resistivity_model[-1]
            resistivity_model = np.concatenate([resistivity_model, padding])
        
        for i, period in enumerate(log_periods):
            # Estimate skin depth for this period
            skin_depth = 500 * np.sqrt(rho_ref * period)
            
            # Calculate weight for each layer based on skin depth
            weights = np.exp(-depths / skin_depth)
            weights = np.append(1.0, weights)  # Add weight for first layer
            weights = np.diff(np.append(weights, 0))  # Convert to layer contributions
            
            # Make sure weights and resistivity_model have the same shape
            weights = weights[:len(resistivity_model)]
            
            # Calculate weighted average resistivity
            weighted_rho = np.sum(resistivity_model * weights) / np.sum(weights)
            
            # Add some random variations (5-10%) to simulate measurement noise
            noise_factor = 1.0 + 0.05 * np.random.randn()
            rho_app[i] = weighted_rho * noise_factor
            
            # Phase is related to derivative of log(rho) vs log(period)
            # For simplicity, we use values between 0 and 90 degrees
            # with higher values for areas of decreasing resistivity with depth
            if i > 0:
                # Calculate approximate derivative
                d_log_rho = np.log10(rho_app[i]) - np.log10(rho_app[i-1])
                d_log_period = np.log10(log_periods[i]) - np.log10(log_periods[i-1])
                slope = d_log_rho / d_log_period
                
                # Convert slope to phase (45° for flat, 0° for rising, 90° for dropping)
                phase_base = 45 - slope * 45
                phase[i] = np.clip(phase_base, 0, 90) + 2 * np.random.randn()
            else:
                phase[i] = 45 + 2 * np.random.randn()
        
        return rho_app, phase
    
    # Generate synthetic dataset
    print(f"Generating {num_samples} synthetic data samples...")
    X_data = []  # Resistivity models
    y_rho = []   # Apparent resistivity curves
    y_phi = []   # Phase curves
    
    for _ in tqdm(range(num_samples)):
        resistivity_model = generate_random_model()
        rho_app, phase = forward_model(resistivity_model)
        
        X_data.append(resistivity_model)
        y_rho.append(rho_app)
        y_phi.append(phase)
    
    X_data = np.array(X_data)
    y_rho = np.array(y_rho)
    y_phi = np.array(y_phi)
    
    # Save synthetic data
    synthetic_data = {
        'X_data': X_data,
        'y_rho': y_rho,
        'y_phi': y_phi,
        'periods': log_periods,
        'layer_thicknesses': layer_thicknesses
    }
    
    with open('synthetic_data.pkl', 'wb') as f:
        pickle.dump(synthetic_data, f)
    
    print("Synthetic data saved to synthetic_data.pkl")
    return synthetic_data

# Train model
def train_model(synthetic_data, model_type='lstm', epochs=100, batch_size=32, learning_rate=0.001, seed=SEED):
    # Set seed for this function
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Extract data
    X_data = synthetic_data['X_data']
    y_rho = synthetic_data['y_rho']
    y_phi = synthetic_data['y_phi']
    layer_thicknesses = synthetic_data['layer_thicknesses']
    
    # Standardize inputs and outputs
    scaler_X_model = StandardScaler()
    scaler_y_rho = StandardScaler()
    scaler_y_phi = StandardScaler()
    
    X_data_scaled = scaler_X_model.fit_transform(X_data)
    y_rho_scaled = scaler_y_rho.fit_transform(y_rho)
    y_phi_scaled = scaler_y_phi.fit_transform(y_phi)
    
    # Combine rho and phi for input to model
    y_combined = np.hstack((y_rho_scaled, y_phi_scaled))
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_data_scaled, y_combined, test_size=0.2, random_state=seed
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(y_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(y_val_tensor, X_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = y_combined.shape[1]
    output_size = X_data_scaled.shape[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model based on selected architecture
    model = create_model(
        model_type=model_type,
        input_size=input_size,
        output_size=output_size,
        hidden_size1=128,
        hidden_size2=64,
        dropout=0.2
    ).to(device)
    
    print(f"Created {model_type.upper()} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save the model and scalers
    torch.save(model.state_dict(), f"{model_type}_model.pt")
    
    scalers = {
        'scaler_X_model': scaler_X_model,
        'scaler_y_rho': scaler_y_rho,
        'scaler_y_phi': scaler_y_phi
    }
    
    with open(f"{model_type}_scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    
    print(f"Model and scalers saved successfully as {model_type}_model.pt and {model_type}_scalers.pkl")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.upper()} Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{model_type}_training_curve.png')
    
    # Calculate final validation metrics 
    print("\nCalculating validation metrics...")
    metrics = calculate_and_save_validation_metrics(
        model=model,
        X_val=y_val_tensor,  # Input MT data
        y_val=X_val_tensor,  # Target resistivity profiles
        model_type=model_type,
        scaler_X_model=scaler_X_model,
        device=device
    )
    
    # Generate validation plots for a few samples
    print("Generating validation sample plots...")
    model.eval()
    
    # Use fixed sample indices for validation plots (for consistent comparison between models)
    # Make sure indices are within the range of validation samples
    global VALIDATION_SAMPLE_INDICES
    valid_indices = [idx for idx in VALIDATION_SAMPLE_INDICES if idx < len(X_val)]
    
    if len(valid_indices) < len(VALIDATION_SAMPLE_INDICES):
        print(f"Warning: Some validation indices are out of range. Using {len(valid_indices)} samples.")
    
    if len(valid_indices) == 0:
        # Fallback: select a few random samples if none of the fixed indices are valid
        num_samples = min(5, len(X_val))
        valid_indices = np.random.choice(len(X_val), num_samples, replace=False)
        print(f"Using random indices for validation plots: {valid_indices}")
    
    for i, idx in enumerate(valid_indices):
        with torch.no_grad():
            inputs = y_val_tensor[idx:idx+1].to(device)  # Get input (MT data)
            true_profile = X_val[idx]  # Get true resistivity profile
            pred_profile_scaled = model(inputs).cpu().numpy()[0]  # Get predicted profile (scaled)
            pred_profile = scaler_X_model.inverse_transform(pred_profile_scaled.reshape(1, -1))[0]  # Unscale
            
        # Calculate depths
        depths = np.cumsum(layer_thicknesses)
        
        # Create stepped visualization
        plt.figure(figsize=(10, 8), dpi=300)
        
        # True profile
        x_true = np.insert(depths, 0, 0)
        y_true = np.insert(scaler_X_model.inverse_transform([true_profile])[0], 0, 
                           scaler_X_model.inverse_transform([true_profile])[0][0])
        
        # Ensure x and y have the same length
        min_len = min(len(x_true), len(y_true))
        x_true = x_true[:min_len]
        y_true = y_true[:min_len]
        
        # Predicted profile
        x_pred = np.insert(depths, 0, 0)
        y_pred = np.insert(pred_profile, 0, pred_profile[0])
        
        # Ensure x and y have the same length
        min_len = min(len(x_pred), len(y_pred))
        x_pred = x_pred[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate sample-specific metrics for display
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Plot
        plt.step(x_true, y_true, where='post', label='True Resistivity', color='red', linewidth=2)
        plt.step(x_pred, y_pred, where='post', label=f'{model_type.upper()} Predicted', linestyle='--', color='blue', linewidth=2)
        
        plt.xlabel('Depth (m)', fontsize=14)
        plt.ylabel('Resistivity (Ohm-m)', fontsize=14)
        plt.title(f'Validation Sample #{idx}: True vs {model_type.upper()} Predicted\nMSE: {mse:.2f}, R²: {r2:.2f}', fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save validation plot
        plt.savefig(f'{model_type}_validation_sample_{idx}.png')
        plt.close()
    
    print(f"Saved {len(valid_indices)} validation plots")
    
    return model, scalers, metrics

# Function to compare metrics across models
def compare_models(model_types=['lstm', 'gru', 'informer']):
    """
    Compare metrics across different models and generate comparison plots
    
    Args:
        model_types: List of model types to compare
    """
    print("\n=== Comparing Models ===")
    
    all_metrics = {}
    valid_models = []
    
    # Load metrics for each model
    for model_type in model_types:
        try:
            with open(f'{model_type}_validation_metrics.pkl', 'rb') as f:
                metrics = pickle.load(f)
                all_metrics[model_type] = metrics
                valid_models.append(model_type)
                print(f"Loaded metrics for {model_type}")
        except FileNotFoundError:
            print(f"No metrics found for {model_type}")
    
    if not valid_models:
        print("No valid models found for comparison")
        return
    
    # Create comparison dataframe
    comparison_data = {
        'Model': [],
        'MSE': [],
        'MAE': [],
        'RMSE': [],
        'R^2': [],
        'Correlation': [],
        'SSIM': [],
        'CI': []
    }
    
    for model_type in valid_models:
        metrics = all_metrics[model_type]
        comparison_data['Model'].append(model_type.upper())
        comparison_data['MSE'].append(metrics['MSE'])
        comparison_data['MAE'].append(metrics['MAE'])
        comparison_data['RMSE'].append(metrics['RMSE'])
        comparison_data['R^2'].append(metrics['R^2'])
        comparison_data['Correlation'].append(metrics['Correlation'])
        comparison_data['SSIM'].append(metrics['SSIM'])
        comparison_data['CI'].append(metrics['CI'])
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    df.to_csv('model_comparison.csv', index=False)
    print("Model comparison saved to model_comparison.csv")
    
    # Create bar charts for visual comparison
    metrics_to_plot = ['MSE', 'MAE', 'RMSE', 'R^2', 'Correlation', 'SSIM', 'CI']
    
    plt.figure(figsize=(15, 12))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(3, 3, i+1)
        
        # Choose appropriate color based on metric (lower is better for error metrics)
        if metric in ['MSE', 'MAE', 'RMSE']:
            # Find the best (lowest) value
            best_idx = df[metric].idxmin()
            colors = ['grey'] * len(df)
            colors[best_idx] = 'green'
        else:
            # Find the best (highest) value for R^2 and correlation
            best_idx = df[metric].idxmax()
            colors = ['grey'] * len(df)
            colors[best_idx] = 'green'
        
        bars = plt.bar(df['Model'], df[metric], color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.title(f'Comparison of {metric}')
        plt.ylabel(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.figtext(0.5, 0.01, "For error metrics (MSE, MAE, RMSE): Lower is better", 
              ha="center", va="bottom", fontsize=12, style='italic')
    plt.figtext(0.5, 0.03, "For other metrics (R^2, Correlation, SSIM, CI): Higher is better", 
              ha="center", va="bottom", fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()
    
    print("Model comparison visualizations saved to model_comparison.png")
    
    return df

# Prediction functionality
def predict_model(edi_file, layered_file, model_type="lstm"):
    """
    Make predictions using the trained model on real data.
    
    Args:
        edi_file: Path to the EDI file
        layered_file: Path to the layered model file
        model_type: Type of model to use (lstm, gru, cnn, etc.)
    """
    # Set random seed for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Extract site name from the file path
    site_name = os.path.basename(edi_file).split('.')[0]
    
    scalers_file = f"{model_type}_scalers.pkl"
    model_path = f"{model_type}_model.pt"
    
    try:
        # Load scalers
        with open(scalers_file, "rb") as file:
            scalers = pickle.load(file)
            scaler_X_model = scalers['scaler_X_model']
            scaler_y_rho = scalers['scaler_y_rho']
            scaler_y_phi = scalers['scaler_y_phi']
    except FileNotFoundError:
        print(f"Error: Scalers file '{scalers_file}' not found. Train the model first.")
        return
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Get the number of output features from scalers
        output_size = scaler_X_model.scale_.shape[0]
        
        # Calculate input size (combined rho and phase)
        input_size = 50  # Combined input size (25 periods for rho + 25 for phase)
        
        # Create model with same architecture
        model = create_model(
            model_type=model_type,
            input_size=input_size,
            output_size=output_size
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Train the model first.")
        return
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    try:
        # Read EDI data
        periods, app_resistivity, phase = read_edi_file(edi_file)
        print(f"EDI data loaded: {len(periods)} periods")
        
        # Read true layered model
        depths, thicknesses, true_resistivity = read_layered_model(layered_file)
        print(f"Layered model loaded: {len(depths)} layers")
        
        # Preprocess the input data
        # Interpolate to match the expected periods from training
        from scipy.interpolate import interp1d
        
        # Get the periods used during training (from the synthetic data)
        training_periods = np.logspace(-3, 3, 25)  # From synthetic data generation
        
        # Create interpolation functions
        f_rho = interp1d(np.log10(periods), np.log10(app_resistivity), 
                        kind='cubic', bounds_error=False, fill_value="extrapolate")
        f_phase = interp1d(np.log10(periods), phase, 
                          kind='cubic', bounds_error=False, fill_value="extrapolate")
        
        # Interpolate to get values at training periods
        interp_rho = 10 ** f_rho(np.log10(training_periods))
        interp_phase = f_phase(np.log10(training_periods))
        
        # Scale the input data
        scaled_rho = scaler_y_rho.transform(interp_rho.reshape(1, -1))
        scaled_phase = scaler_y_phi.transform(interp_phase.reshape(1, -1))
        
        # Combine and convert to tensor
        X_combined = np.hstack((scaled_rho, scaled_phase))
        X_tensor = torch.FloatTensor(X_combined).to(device)
        
        # Make prediction
        with torch.no_grad():
            y_pred_scaled = model(X_tensor)
        
        # Convert to numpy and inverse transform
        pred_resistivity_profile = scaler_X_model.inverse_transform(
            y_pred_scaled.cpu().numpy()
        )[0]
        
        # Load synthetic data to get the layer thicknesses
        try:
            with open('synthetic_data.pkl', 'rb') as f:
                synthetic_data = pickle.load(f)
                # Use the layer thicknesses from the training data
                layer_thicknesses = synthetic_data['layer_thicknesses']
        except FileNotFoundError:
            # If synthetic data file not found, estimate layer thicknesses
            print("Warning: Could not load synthetic_data.pkl, estimating layer thicknesses")
            mu_0 = 4 * np.pi * 1e-7
            rho_ref = 100
            min_period = training_periods.min()
            max_period = training_periods.max()
            min_skin_depth = 500 * np.sqrt(rho_ref * min_period) / 4
            max_skin_depth = 500 * np.sqrt(rho_ref * max_period)
            
            layer_thicknesses = [min_skin_depth]
            while layer_thicknesses[-1] * 1.2 < max_skin_depth:
                layer_thicknesses.append(layer_thicknesses[-1] * 1.2)
            layer_thicknesses = np.array(layer_thicknesses)
        
        # Make sure pred_resistivity_profile has the correct length
        if len(pred_resistivity_profile) > len(layer_thicknesses) + 1:
            # Truncate if too long
            pred_resistivity_profile = pred_resistivity_profile[:len(layer_thicknesses) + 1]
        elif len(pred_resistivity_profile) < len(layer_thicknesses) + 1:
            # Pad if too short
            padding = np.ones(len(layer_thicknesses) + 1 - len(pred_resistivity_profile)) * pred_resistivity_profile[-1]
            pred_resistivity_profile = np.concatenate([pred_resistivity_profile, padding])
        
        # Calculate depths for predicted model
        pred_depths = np.cumsum(layer_thicknesses)
        
        # Plot the results
        plt.figure(figsize=(12, 10), dpi=300)
        
        # Print shapes for debugging
        print(f"pred_depths shape: {pred_depths.shape}, pred_resistivity_profile shape: {pred_resistivity_profile.shape}")
        
        # Create arrays for stepped visualization with matching dimensions
        x_pred = np.insert(pred_depths, 0, 0)
        y_pred = np.insert(pred_resistivity_profile, 0, pred_resistivity_profile[0])
        
        # Ensure x and y have the same length
        min_len = min(len(x_pred), len(y_pred))
        x_pred = x_pred[:min_len]
        y_pred = y_pred[:min_len]
        
        # Plot the predicted resistivity profile
        plt.step(
            x_pred,
            y_pred,
            where='post',
            label=f'{model_type.upper()} Predicted Resistivity',
            linestyle='--',
            color='blue',
            linewidth=2
        )
        
        # Plot the true resistivity profile
        plt.step(
            depths,
            true_resistivity,
            where='post',
            label='True Resistivity (1D Inversion)',
            color='red',
            linewidth=2
        )
        
        plt.xlabel('Depth (m)', fontsize=18)
        plt.ylabel('Resistivity (Ohm-m)', fontsize=18)
        plt.title(f'Resistivity vs Depth: {site_name} - {model_type.upper()} Model', fontsize=18)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)  # Increase tick label size
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{site_name}_{model_type}_resistivity_comparison.png')
        plt.show()
        
        print("Prediction and plotting completed successfully.")
        print(f"Results saved to {site_name}_{model_type}_resistivity_comparison.png")
        
        return pred_resistivity_profile, pred_depths, true_resistivity, depths
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Add function to run prediction on a specific site
def run_prediction(site_name, data_dir="real_data", model_type="lstm"):
    """
    Run prediction on a specific site
    
    Args:
        site_name: Name of the site (e.g., '3D01')
        data_dir: Directory containing the EDI and layered model files
        model_type: Type of model to use (lstm, gru, cnn, etc.)
    """
    edi_file = f"{data_dir}/{site_name}.edi"
    layered_file = f"{data_dir}/{site_name}Layered.txt"
    
    if os.path.exists(edi_file) and os.path.exists(layered_file):
        print(f"Running {model_type.upper()} prediction for site {site_name}")
        return predict_model(edi_file, layered_file, model_type)
    else:
        print(f"Error: Files for site {site_name} not found in {data_dir}")
        return None

def run_multiple_predictions(sites=None, data_dir="real_data", model_type="lstm"):
    """
    Run prediction on multiple sites
    
    Args:
        sites: List of site names to process. If None, automatically finds all available sites.
        data_dir: Directory containing the EDI and layered model files
        model_type: Type of model to use (lstm, gru, cnn, etc.)
    """
    if sites is None:
        # Find all available sites automatically
        edi_files = [f for f in os.listdir(data_dir) if f.endswith('.edi')]
        sites = [f.split('.')[0] for f in edi_files]
    
    results = {}
    for site in sites:
        print(f"\n{'='*50}")
        print(f"Processing site: {site}")
        print(f"{'='*50}")
        result = run_prediction(site, data_dir, model_type)
        results[site] = result
        
    # Print summary
    print(f"\n{'='*50}")
    print(f"Prediction completed for {len(results)} sites using {model_type.upper()} model:")
    for site in results:
        status = "Success" if results[site] is not None else "Failed"
        print(f"  - {site}: {status}")
    
    return results

# Add ProbMask class
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device)
        _mask[index] = False
        self.mask = _mask.view(1, 1, L, scores.shape[-1]).expand(B, H, L, scores.shape[-1])

# Add function to calculate and save validation metrics for each model
def calculate_and_save_validation_metrics(model, X_val, y_val, model_type, scaler_X_model, device):
    """
    Calculate and save validation metrics for model comparison
    
    Args:
        model: Trained model
        X_val: Validation inputs (MT data)
        y_val: Validation targets (resistivity profiles)
        model_type: Type of model ('lstm', 'gru', 'informer')
        scaler_X_model: Scaler for resistivity profiles
        device: Computing device (CPU/CUDA)
    """
    model.eval()
    
    # Convert data to tensors if they're not already
    if not isinstance(X_val, torch.Tensor):
        X_val_tensor = torch.FloatTensor(X_val).to(device)
    else:
        X_val_tensor = X_val.to(device)
        
    if not isinstance(y_val, torch.Tensor):
        y_val_tensor = torch.FloatTensor(y_val).to(device)
    else:
        y_val_tensor = y_val.to(device)
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_val_tensor)
    
    # Convert predictions and targets to numpy for metrics calculation
    y_pred_np = y_pred.cpu().numpy()
    y_val_np = y_val_tensor.cpu().numpy()
    
    # Inverse transform if scalers are provided
    if scaler_X_model is not None:
        y_pred_np_inv = scaler_X_model.inverse_transform(y_pred_np)
        y_val_np_inv = scaler_X_model.inverse_transform(y_val_np)
    else:
        y_pred_np_inv = y_pred_np
        y_val_np_inv = y_val_np
        
    # Calculate metrics on the entire dataset
    mse = mean_squared_error(y_val_np_inv, y_pred_np_inv)
    mae = mean_absolute_error(y_val_np_inv, y_pred_np_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val_np_inv, y_pred_np_inv)
    
    # Calculate global correlation
    corr = np.corrcoef(y_val_np_inv.flatten(), y_pred_np_inv.flatten())[0, 1]
    
    # Calculate Shape Similarity Index and Correlation Index
    ssim_values = []
    ci_values = []
    
    # Calculate SSIM and CI for each sample
    for i in range(y_val_np_inv.shape[0]):
        ssim = calculate_ssim(y_val_np_inv[i], y_pred_np_inv[i])
        ci = calculate_ci(y_val_np_inv[i], y_pred_np_inv[i])
        ssim_values.append(ssim)
        ci_values.append(ci)
    
    # Calculate mean values
    mean_ssim = np.mean(ssim_values)
    mean_ci = np.mean(ci_values)
    
    # Save metrics to file
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R^2': r2,
        'Correlation': corr,
        'SSIM': mean_ssim,
        'CI': mean_ci
    }
    
    # Print metrics
    print(f"\n=== {model_type.upper()} Validation Metrics ===")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Correlation: {corr:.4f}")
    print(f"Shape Similarity Index (SSIM): {mean_ssim:.4f}")
    print(f"Correlation Index (CI): {mean_ci:.4f}")
    
    # Save metrics to pickle file
    with open(f'{model_type}_validation_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    # Also save as CSV for easier viewing
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'RMSE', 'R^2', 'Correlation', 'SSIM', 'CI'],
        'Value': [mse, mae, rmse, r2, corr, mean_ssim, mean_ci]
    })
    metrics_df.to_csv(f'{model_type}_validation_metrics.csv', index=False)
    
    print(f"Validation metrics saved for {model_type}")
    
    return metrics

# Function to calculate Shape Similarity Index (SSIM)
def calculate_ssim(y_true, y_pred):
    """
    Calculate Shape Similarity Index between true and predicted profiles
    
    Args:
        y_true: True resistivity profile
        y_pred: Predicted resistivity profile
        
    Returns:
        Shape Similarity Index (0-1 where 1 is perfect match)
    """
    # Apply Gaussian smoothing to reduce noise influence
    y_true_smooth = gaussian_filter1d(y_true, sigma=1.0)
    y_pred_smooth = gaussian_filter1d(y_pred, sigma=1.0)
    
    # Calculate first derivatives (gradients)
    true_grad = np.gradient(y_true_smooth)
    pred_grad = np.gradient(y_pred_smooth)
    
    # Normalize gradients
    true_grad_norm = true_grad / (np.linalg.norm(true_grad) + 1e-8)
    pred_grad_norm = pred_grad / (np.linalg.norm(pred_grad) + 1e-8)
    
    # Calculate the dot product for shape similarity
    dot_product = np.abs(np.sum(true_grad_norm * pred_grad_norm))
    
    # Scale to [0, 1] range
    similarity = dot_product / len(true_grad)
    
    return similarity

# Function to calculate Correlation Index (CI)
def calculate_ci(y_true, y_pred):
    """
    Calculate Correlation Index which measures how well patterns are captured
    
    Args:
        y_true: True resistivity profile
        y_pred: Predicted resistivity profile
        
    Returns:
        Correlation Index (0-1 where 1 is perfect correlation)
    """
    # Calculate Pearson correlation coefficient
    try:
        corr, _ = pearsonr(y_true, y_pred)
        # Convert to [0, 1] range, where 1 is perfect correlation
        ci = (corr + 1) / 2
    except:
        # If correlation cannot be calculated (e.g., constant values)
        ci = 0.0
    
    return ci

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Deep Learning models for MT data - train or predict')
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='both',
                        help='Mode: train, predict, or both (default: both)')
    parser.add_argument('--model', choices=['lstm', 'gru', 'informer', 'all'], 
                        default='lstm', help='Model architecture to use (default: lstm)')
    parser.add_argument('--site', type=str, default='3D01',
                        help='Site name for prediction (default: 3D01)')
    parser.add_argument('--seed', type=int, default=99, 
                        help='Random seed for reproducibility (default: 99)')
    parser.add_argument('--all-sites', action='store_true',
                        help='Run prediction on all available sites')
    parser.add_argument('--sites', type=str, nargs='+',
                        help='List of specific sites to process')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison of all trained models')
    
    args = parser.parse_args()
    
    # Override the global SEED directly without using global statement
    # This works because we're in the module scope
    globals()['SEED'] = args.seed
    
    # Reset random seeds with new SEED value
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True  # To ensure reproducibility
        torch.backends.cudnn.benchmark = False     # To ensure reproducibility
    
    # List of models to train/evaluate
    if args.model == 'all':
        model_types = ['lstm', 'gru', 'informer']
    else:
        model_types = [args.model]
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"PROCESSING MODEL: {model_type.upper()}")
        print(f"{'='*70}")
        
        # Execute based on mode
        if args.mode in ['train', 'both']:
            print(f"=== Training {model_type.upper()} model with seed {SEED} ===")
    # Generate synthetic data based on real data parameters
            synthetic_data = generate_synthetic_data(num_samples=10000, num_periods=25, num_layers=50, seed=SEED)
    
    # Train the model
            model, scalers, metrics = train_model(synthetic_data, model_type=model_type, epochs=50, batch_size=64, learning_rate=0.001, seed=SEED)
        
        if args.mode in ['predict', 'both']:
            print(f"=== {model_type.upper()} Prediction mode with seed {SEED} ===")
            if args.all_sites:
                run_multiple_predictions(model_type=model_type)
            elif args.sites:
                run_multiple_predictions(args.sites, model_type=model_type)
            else:
                run_prediction(args.site, model_type=model_type)
    
    # Compare models if requested or if multiple models were trained
    if args.compare or (args.model == 'all' and args.mode in ['train', 'both']):
        print("\nGenerating model comparison...")
        compare_models(model_types)
