import matplotlib
# Force headless backend to prevent "main thread is not in main loop" crash
matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import re
import random
from tqdm import tqdm
import argparse
import math
from math import sqrt
import pandas as pd
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import cmath
import optuna
from joblib import Parallel, delayed
import multiprocessing

# Set seeds for reproducibility
SEED = 99
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(SEED)

# Determine optimal workers for data loading
# Cap workers at 8 to prevent "Too many open files" error
NUM_WORKERS = min(8, multiprocessing.cpu_count())
print(f"Detected {multiprocessing.cpu_count()} CPU cores. Using {NUM_WORKERS} workers for DataLoading.")

# Fixed validation sample indices for consistent comparisons
VALIDATION_SAMPLE_INDICES = [5, 15, 25, 35, 45]

# -------------------------------------------------------------------------
# MODEL ARCHITECTURES
# -------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.1, output_size=None):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(2)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :] 
        x = self.fc(x)
        return x 

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.1, output_size=None):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size=1, hidden_size=hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(2)
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B, device=device)[:, None, None],
                             torch.arange(H, device=device)[None, :, None],
                             index, :].to(device)
        self.mask = indicator.view(scores.shape).to(device)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device) 
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        Q_reduce = Q[torch.arange(B, device=Q.device)[:, None, None], 
                     torch.arange(H, device=Q.device)[None, :, None], 
                     M_top, :] 
        
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)
        
        context_in[torch.arange(B, device=V.device)[:, None, None],
                   torch.arange(H, device=V.device)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
                   
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B, device=attn.device)[:, None, None], 
                  torch.arange(H, device=attn.device)[None, :, None], 
                  index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, u, U_part)
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
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
        out, attn = self.inner_attention(queries, keys, values)
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
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn

class InformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, n_heads=8, e_layers=3, d_ff=256, dropout=0.0, output_size=None):
        super(InformerModel, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(False, 5, attention_dropout=dropout, output_attention=False),
                    d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation="gelu"
            ) for _ in range(e_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.embedding(x)
        x += self.pos_emb(x)
        for enc_layer in self.encoder_layers:
            x, attn = enc_layer(x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

def create_model(model_type, input_size, output_size, hidden_size1=128, hidden_size2=64, dropout=0.2):
    model_classes = {'lstm': LSTMModel, 'gru': GRUModel, 'informer': InformerModel}
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = model_classes[model_type]
    
    if model_type == 'informer':
        n_heads = 4
        if hidden_size1 % n_heads != 0:
            hidden_size1 = (hidden_size1 // n_heads) * n_heads

        model = model_class(
            input_size=input_size, d_model=hidden_size1, n_heads=n_heads, e_layers=2,
            d_ff=hidden_size2*4, dropout=dropout, output_size=output_size
        )
    else:
        model = model_class(
            input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2,
            dropout=dropout, output_size=output_size
        )
    return model

# -------------------------------------------------------------------------
# DATA PROCESSING
# -------------------------------------------------------------------------
def get_synthetic_depths(synth):
    th = synth['layer_thicknesses']           
    n_layers = synth['X_data'].shape[1]       

    interfaces = np.cumsum(np.insert(th, 0, 0.0))
    mid = 0.5 * (interfaces[:-1] + interfaces[1:])

    if n_layers == len(mid) + 1:
        extra_depth = interfaces[-1] + (th[-1] if len(th) > 0 else interfaces[-1] or 1000.0)
        depths = np.concatenate([mid, [extra_depth]])
    else:
        if len(mid) >= n_layers:
            depths = mid[:n_layers]
        else:
            pad_val = mid[-1] if len(mid) > 0 else 1000.0
            depths = np.concatenate([mid, np.full(n_layers - len(mid), pad_val)])

    return depths.astype(np.float32)

def read_edi_file(edi_file):
    with open(edi_file, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
    
    freq_match = re.search(r'>FREQ.*?\n(.*?)(?=>)', content, re.DOTALL)
    if not freq_match:
        freq_match = re.search(r'>FREQ.*?\n(.*?)(?=$)', content, re.DOTALL)
        
    if freq_match:
        raw_nums = re.sub(r'[^\d\.E\+\-\s]', '', freq_match.group(1).upper())
        frequencies = np.array([float(x) for x in raw_nums.split()])
        periods = 1.0 / frequencies
    else:
        raise ValueError(f"Could not extract frequencies from EDI file: {edi_file}")
    
    def extract_block(name, content):
        match = re.search(f'>{name}.*?\n(.*?)(?=>)', content, re.DOTALL)
        if match:
             raw_nums = re.sub(r'[^\d\.E\+\-\s]', '', match.group(1).upper())
             vals = np.array([float(x) for x in raw_nums.split()])
             return vals
        return None

    zxyr = extract_block('ZXYR', content)
    zxyi = extract_block('ZXYI', content)
    
    if zxyr is None or zxyi is None:
        print(f"Warning: Could not find ZXYR/ZXYI in {edi_file}, looking for impedance tensor...")
        return None, None, None 

    mu_0 = 4 * np.pi * 1e-7
    
    min_len = min(len(frequencies), len(zxyr), len(zxyi))
    frequencies = frequencies[:min_len]
    periods = periods[:min_len]
    zxyr = zxyr[:min_len]
    zxyi = zxyi[:min_len]

    phase = np.arctan2(zxyi, zxyr) * 180 / np.pi
    rho_apparent = (zxyr**2 + zxyi**2) / (2 * np.pi * mu_0 * frequencies)
    
    idx = np.argsort(periods)
    return periods[idx], rho_apparent[idx], phase[idx]

def read_layered_model(layered_file):
    with open(layered_file, 'r') as f: lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and re.match(r'^\s*[\d\.]+', line):
            data_start = i
            break
            
    depths, thicknesses, resistivities = [], [], []
    for line in lines[data_start:]:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 3:
                depths.append(float(parts[0]))
                thicknesses.append(float(parts[1]))
                resistivities.append(float(parts[2]))
    return np.array(depths), np.array(thicknesses), np.array(resistivities)

# -------------------------------------------------------------------------
# PHYSICS
# -------------------------------------------------------------------------
def analytical_forward_model(resistivities, thicknesses, frequencies):
    mu = 4 * np.pi * 1e-7
    n_layers = len(resistivities)
    n_freqs = len(frequencies)
    app_res, phase = np.zeros(n_freqs), np.zeros(n_freqs)
    
    for idx, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq
        k = np.sqrt(-1j * omega * mu * (1.0 / np.array(resistivities, dtype=complex)))
        Z_intrinsic = (1j * omega * mu) / k
        Z_in = Z_intrinsic[-1] 
        
        for n in range(n_layers - 2, -1, -1):
            arg = k[n] * thicknesses[n]
            if arg.real > 40: 
                tanh_kh = 1.0
            else:
                tanh_kh = np.tanh(arg)
                
            numerator = Z_in + Z_intrinsic[n] * tanh_kh
            denominator = Z_intrinsic[n] + Z_in * tanh_kh
            Z_in = Z_intrinsic[n] * (numerator / denominator)
            
        app_res[idx] = (1 / (omega * mu)) * (abs(Z_in) ** 2)
        phase[idx] = np.degrees(cmath.phase(Z_in))
        
    return app_res, phase

def bostick_transform(rho_app, periods):
    mu_0 = 4 * np.pi * 1e-7
    log_rho, log_T = np.log10(rho_app), np.log10(periods)
    m = np.gradient(log_rho, log_T)
    m = np.clip(m, -0.9, 0.9)
    
    rho_bostick = rho_app * (1 + m) / (1 - m)
    depths = np.sqrt((rho_app * periods) / (2 * np.pi * mu_0))
    return depths, rho_bostick

# -------------------------------------------------------------------------
# DATA GENERATION (CONSTRAINED TO REAL DATA STATS)
# -------------------------------------------------------------------------
def _generate_single_sample(layer_thicknesses,
                            frequencies,
                            mean_log_res,
                            std_log_res,
                            min_log_res,
                            max_log_res,
                            seed_offset):
    np.random.seed(SEED + seed_offset)

    n_layers = len(layer_thicknesses) + 1

    # STRATEGY 3: Tighter bounds based on real data stats
    log_res = np.random.normal(mean_log_res, std_log_res * 1.5, n_layers)
    log_res = np.clip(log_res, min_log_res * 0.8, max_log_res * 1.2)

    for i in range(1, len(log_res)):
        log_res[i] = 0.6 * log_res[i - 1] + 0.4 * log_res[i]

    log_res = gaussian_filter1d(log_res, sigma=1.0)
    resistivity_model = 10.0 ** log_res

    rho_app, phase = analytical_forward_model(
        resistivity_model,
        layer_thicknesses,
        frequencies
    )

    return (
        resistivity_model.astype(np.float32),
        rho_app.astype(np.float32),
        phase.astype(np.float32)
    )

def generate_synthetic_data(num_samples=100000, num_periods=25, num_layers=50, seed=SEED):
    np.random.seed(seed)
    print(f"Generating {num_samples} synthetic samples using {NUM_WORKERS} CPU cores...")

    log_periods = np.logspace(-3, 3, num_periods)
    frequencies = 1.0 / log_periods

    # Defaults
    mean_log_res, std_log_res, min_log_res, max_log_res = 2.0, 1.0, 0.0, 4.0

    all_resistivities = []
    if os.path.isdir("real_data"):
        edi_files = [f for f in os.listdir("real_data") if f.endswith(".edi")]
        for edi_file in edi_files:
            site_name = edi_file.split(".")[0]
            layered_file = os.path.join("real_data", f"{site_name}Layered.txt")
            if os.path.exists(layered_file):
                _, _, resistivities = read_layered_model(layered_file)
                if len(resistivities) > 0:
                    all_resistivities.append(resistivities)

    if len(all_resistivities) > 0:
        log_resistivities = [np.log10(res) for res in all_resistivities]
        mean_log_res = np.mean([np.mean(lr) for lr in log_resistivities])
        std_log_res = np.mean([np.std(lr) for lr in log_resistivities])
        min_log_res = min([np.min(lr) for lr in log_resistivities])
        max_log_res = max([np.max(lr) for lr in log_resistivities])
        print(f"Real data stats: Mean={mean_log_res:.2f}, Std={std_log_res:.2f}, Min={min_log_res:.2f}, Max={max_log_res:.2f}")

    rho_ref = 10.0 ** mean_log_res
    min_period = log_periods.min()
    max_period = log_periods.max()

    min_skin_depth = 500.0 * np.sqrt(rho_ref * min_period) / 4.0
    max_skin_depth = 500.0 * np.sqrt(rho_ref * max_period)

    layer_thicknesses = [min_skin_depth]
    while layer_thicknesses[-1] * 1.2 < max_skin_depth and len(layer_thicknesses) < num_layers - 1:
        layer_thicknesses.append(layer_thicknesses[-1] * 1.2)

    layer_thicknesses = np.array(layer_thicknesses, dtype=np.float32)

    results = Parallel(n_jobs=NUM_WORKERS)(
        delayed(_generate_single_sample)(
            layer_thicknesses, frequencies, mean_log_res, std_log_res, min_log_res, max_log_res, i
        )
        for i in tqdm(range(num_samples))
    )

    X_data = np.array([r[0] for r in results], dtype=np.float32)
    y_rho  = np.array([r[1] for r in results], dtype=np.float32)
    y_phi  = np.array([r[2] for r in results], dtype=np.float32)

    synthetic_data = {
        "X_data": X_data,
        "y_rho": y_rho,
        "y_phi": y_phi,
        "periods": log_periods,
        "layer_thicknesses": layer_thicknesses,
    }

    with open("synthetic_data.pkl", "wb") as f: pickle.dump(synthetic_data, f)
    return synthetic_data

class MTAugmentedDataset(Dataset):
    def __init__(self, inputs, targets, augment=False, noise_std=0.05):
        self.inputs = inputs
        self.targets = targets
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx].clone()
        target_sample = self.targets[idx].clone()
        if self.augment:
            noise = torch.randn_like(input_sample) * self.noise_std
            input_sample += noise
        return input_sample, target_sample

# -------------------------------------------------------------------------
# TRAINING & METRICS
# -------------------------------------------------------------------------
def visualize_model_architecture(model, model_type, input_size, device):
    try:
        from torchview import draw_graph
        print(f"\nGenerating architecture diagram for {model_type.upper()}...")
        model_graph = draw_graph(
            model, input_size=(1, input_size), device=device, expand_nested=True,
            graph_name=f"{model_type}_architecture", save_graph=True,
            filename=f"{model_type}_architecture"
        )
        print(f"Diagram saved to {model_type}_architecture.png")
    except Exception as e:
        print(f"Skipping architecture plot: {e}")

def calculate_ci(y_true, y_pred):
    try:
        corr, _ = pearsonr(y_true, y_pred)
        return (corr + 1) / 2
    except: return 0.0

def calculate_ssim(y_true, y_pred):
    data_range = np.max(y_true) - np.min(y_true) + 1e-6
    y_true_norm = (y_true - np.min(y_true)) / data_range
    y_pred_norm = (y_pred - np.min(y_true)) / data_range 
    u_x = np.mean(y_true_norm); u_y = np.mean(y_pred_norm)
    var_x = np.var(y_true_norm); var_y = np.var(y_pred_norm)
    cov_xy = np.mean((y_true_norm - u_x) * (y_pred_norm - u_y))
    c1 = 0.01 ** 2; c2 = 0.03 ** 2
    ssim = (2 * u_x * u_y + c1) * (2 * cov_xy + c2) / ((u_x**2 + u_y**2 + c1) * (var_x + var_y + c2))
    return ssim

# STRATEGY 2: WEIGHTED LOSS FUNCTION
def depth_weighted_smooth_loss(pred, target, depths, lambda_smooth=0.2, lambda_deep=0.1):
    # Base MSE
    mse = F.mse_loss(pred, target)
    # Smoothness (minimize derivative)
    diff_pred = pred[:, 1:] - pred[:, :-1]
    diff_tgt  = target[:, 1:] - target[:, :-1]
    smooth = F.mse_loss(diff_pred, diff_tgt)
    # Depth weighting (prioritize shallow layers)
    d = torch.from_numpy(depths).to(pred.device).float()
    w = 1.0 / torch.sqrt(d + 1.0)
    w = w / w.mean()
    mse_depth = torch.mean(w * (pred - target)**2)
    
    return mse_depth + lambda_smooth * smooth + lambda_deep * mse

# STRATEGY 1: SMOOTHING POST-PROCESSING
def smooth_predictions(y_pred, window_length=5, polyorder=2):
    """Apply Savitzky-Golay filter to smooth predictions."""
    try:
        if len(y_pred) < window_length: return y_pred
        return savgol_filter(y_pred, window_length, polyorder)
    except: return y_pred

def predict_batch(model, data_tensor, batch_size, device):
    model.eval()
    predictions = []
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        for batch in loader:
            batch_data = batch[0].to(device, non_blocking=True)
            pred = model(batch_data)
            predictions.append(pred.cpu())
    return torch.cat(predictions, dim=0)

def calculate_and_save_validation_metrics(model, X_val, y_val, model_type, scaler_X_model, device):
    model.eval()
    if not isinstance(X_val, torch.Tensor): X_val = torch.FloatTensor(X_val)
    y_pred = predict_batch(model, X_val, batch_size=4096, device=device)
    y_pred_np = y_pred.numpy()
    y_target_np = y_val.cpu().numpy() 
    
    if scaler_X_model:
        y_pred_np = scaler_X_model.inverse_transform(y_pred_np)
        y_target_np = scaler_X_model.inverse_transform(y_target_np)
        
    mse = mean_squared_error(y_target_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_target_np, y_pred_np)
    
    subset_size = min(2000, len(y_target_np)) 
    ssim_vals = [calculate_ssim(y_target_np[i], y_pred_np[i]) for i in range(subset_size)]
    ci_vals = [calculate_ci(y_target_np[i], y_pred_np[i]) for i in range(subset_size)]
    
    metrics = {'MSE': mse, 'RMSE': rmse, 'R^2': r2, 'SSIM': np.mean(ssim_vals), 'CI': np.mean(ci_vals)}
    print(f"\n=== {model_type.upper()} Metrics ===\nRMSE: {rmse:.4f}, R^2: {r2:.4f}, SSIM: {metrics['SSIM']:.4f}")
    with open(f'{model_type}_validation_metrics.pkl', 'wb') as f: pickle.dump(metrics, f)
    return metrics

def train_model(synthetic_data, model_type='lstm', epochs=100, batch_size=32, learning_rate=0.001, seed=SEED, hidden_size1=128, hidden_size2=64, dropout=0.2):
    set_seeds(seed)
    X_raw_model = np.log10(synthetic_data['X_data']) 
    y_rho_raw = np.log10(synthetic_data['y_rho'])
    y_phi_raw = synthetic_data['y_phi']
    X_input_raw = np.hstack((y_rho_raw, y_phi_raw))
    
    X_train_in, X_val_in, y_train_tgt, y_val_tgt = train_test_split(X_input_raw, X_raw_model, test_size=0.2, random_state=seed)
    
    scaler_input = StandardScaler()
    scaler_target = StandardScaler()
    
    X_train_scaled = scaler_input.fit_transform(X_train_in)
    y_train_scaled = scaler_target.fit_transform(y_train_tgt)
    X_val_scaled = scaler_input.transform(X_val_in)
    y_val_scaled = scaler_target.transform(y_val_tgt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = MTAugmentedDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled), augment=True, noise_std=0.05)
    val_dataset = MTAugmentedDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled), augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    model = create_model(model_type, X_train_scaled.shape[1], y_train_scaled.shape[1], hidden_size1, hidden_size2, dropout).to(device)
    visualize_model_architecture(model, model_type, X_train_scaled.shape[1], device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    synth_depths = get_synthetic_depths(synthetic_data)
    
    def criterion(pred, target):
        return depth_weighted_smooth_loss(pred, target, synth_depths)

    scaler = torch.amp.GradScaler('cuda') 
    
    train_losses, val_losses = [], []
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(inputs)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out = model(inputs)
                    loss = criterion(out, targets)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1}/{epochs}, Train: {train_losses[-1]:.5f}, Val: {val_losses[-1]:.5f}")

    save_dict = {'model_state': model.state_dict(), 'config': {'input_size': X_train_scaled.shape[1], 'output_size': y_train_scaled.shape[1], 'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2, 'dropout': dropout, 'model_type': model_type}}
    torch.save(save_dict, f"{model_type}_checkpoint.pt")
    with open(f"{model_type}_scalers.pkl", "wb") as f: pickle.dump({'scaler_input': scaler_input, 'scaler_target': scaler_target}, f)
    
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(f'{model_type.upper()} Loss')
    plt.legend()
    plt.savefig(f'{model_type}_training_curve.png')
    plt.close()
    
    metrics = calculate_and_save_validation_metrics(model, torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled), model_type, scaler_target, device)
    return model, metrics

# -------------------------------------------------------------------------
# CROSS VALIDATION
# -------------------------------------------------------------------------
def run_kfold_cross_validation(synthetic_data, model_type, k_folds, epochs, batch_size, learning_rate):
    print(f"\n=== Running {k_folds}-Fold CV for {model_type} ===")
    X_raw_model = np.log10(synthetic_data['X_data']) 
    y_rho_raw = np.log10(synthetic_data['y_rho'])
    y_phi_raw = synthetic_data['y_phi']
    X_input_raw = np.hstack((y_rho_raw, y_phi_raw))
    layer_depths = get_synthetic_depths(synthetic_data)
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    metrics_log = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold, (t_idx, v_idx) in enumerate(kf.split(X_input_raw)):
        print(f"Fold {fold+1}/{k_folds}")
        scaler_in, scaler_out = StandardScaler(), StandardScaler()
        X_t, X_v = X_input_raw[t_idx], X_input_raw[v_idx]
        y_t, y_v = X_raw_model[t_idx], X_raw_model[v_idx]
        X_t_s = scaler_in.fit_transform(X_t); y_t_s = scaler_out.fit_transform(y_t)
        X_v_s = scaler_in.transform(X_v); y_v_s = scaler_out.transform(y_v)
        
        train_ds = MTAugmentedDataset(torch.FloatTensor(X_t_s), torch.FloatTensor(y_t_s), augment=True)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
        model = create_model(model_type, X_t_s.shape[1], y_t_s.shape[1]).to(device)
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler('cuda')

        for _ in range(epochs):
            model.train()
            for x, y in loader:
                opt.zero_grad()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    output = model(x)
                    loss = depth_weighted_smooth_loss(output, y, layer_depths)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        metrics = calculate_and_save_validation_metrics(model, torch.FloatTensor(X_v_s), torch.FloatTensor(y_v_s), f"{model_type}_fold{fold}", scaler_out, device)
        metrics_log.append(metrics)
    
    df = pd.DataFrame(metrics_log)
    df.to_csv(f'{model_type}_kfold_results.csv', index=False)
    print("K-Fold Results:\n", df.mean())

# -------------------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------------------
def predict_model(edi_file, layered_file, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Load scalers
    # -------------------------------
    scaler_path = f"{model_type}_scalers.pkl"
    if not os.path.exists(scaler_path):
        print(f"[ERROR] Scaler file not found: {scaler_path}")
        return

    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    scaler_input = scalers["scaler_input"]
    scaler_target = scalers["scaler_target"]

    # -------------------------------
    # Load fine-tuned OR base checkpoint
    # -------------------------------
    ckpt_ft   = f"{model_type}_checkpoint_finetuned.pt"
    ckpt_base = f"{model_type}_checkpoint.pt"

    if os.path.exists(ckpt_ft):
        ckpt_path = ckpt_ft
        print(f"[INFO] Using FINE-TUNED checkpoint: {ckpt_path}")
    elif os.path.exists(ckpt_base):
        ckpt_path = ckpt_base
        print(f"[INFO] Using BASE checkpoint: {ckpt_path}")
    else:
        print(f"[ERROR] No checkpoint found for model: {model_type}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # -------------------------------
    # Build & load model
    # -------------------------------
    model = create_model(
        config["model_type"],
        config["input_size"],
        config["output_size"],
        config["hidden_size1"],
        config["hidden_size2"],
        config["dropout"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    # -------------------------------
    # Read EDI file
    # -------------------------------
    periods, rho, phase = read_edi_file(edi_file)
    if periods is None:
        print(f"[ERROR] Could not read EDI file: {edi_file}")
        return

    # -------------------------------
    # Load synthetic metadata (period grid, layer depths)
    # -------------------------------
    with open("synthetic_data.pkl", "rb") as f:
        synth = pickle.load(f)

    train_periods = synth["periods"]

    # -------------------------------
    # Interpolate real MT curves onto synthetic frequency grid
    # -------------------------------
    from scipy.interpolate import interp1d
    rho_func = interp1d(np.log10(periods), np.log10(rho), fill_value="extrapolate")
    phi_func = interp1d(np.log10(periods), phase, fill_value="extrapolate")

    rho_in = 10 ** rho_func(np.log10(train_periods))
    phase_in = phi_func(np.log10(train_periods))

    # Input vector in log space
    input_vector = np.hstack((np.log10(rho_in), phase_in)).reshape(1, -1)
    input_scaled = scaler_input.transform(input_vector)
    x_tensor = torch.FloatTensor(input_scaled).to(device)

    # -------------------------------
    # Predict resistivity profile
    # -------------------------------
    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()

    pred_log_res = scaler_target.inverse_transform(pred_scaled)[0]

    # Apply your smoothing post-processing
    pred_log_res = smooth_predictions(pred_log_res)

    pred_res = 10 ** pred_log_res
    pred_res = np.clip(pred_res, 0.1, 1e4)

    # -------------------------------
    # Load TRUE model if available
    # -------------------------------
    if os.path.exists(layered_file):
        true_depths, _, true_res = read_layered_model(layered_file)
    else:
        true_depths, true_res = None, None

    # -------------------------------
    # Build predicted depth axis
    # -------------------------------
    pred_thk = synth["layer_thicknesses"]
    x_p = np.cumsum(np.insert(pred_thk, 0, 0))
    y_p = pred_res

    # Fix length mismatches
    if len(y_p) > len(x_p):      
        x_p = np.append(x_p, x_p[-1] * 1.5 if x_p[-1] > 0 else 10000)

    min_len = min(len(x_p), len(y_p))
    x_p = x_p[:min_len]
    y_p = y_p[:min_len]

    # -------------------------------
    # Plot prediction vs true model
    # -------------------------------
    plt.figure(figsize=(10, 8))
    plt.step(x_p, y_p, where="post", label=f"{model_type.upper()} Prediction",
             color="blue", linewidth=2)

    if true_depths is not None:
        plt.step(true_depths, true_res, where="post",
                 label="True Model", color="black", alpha=0.6, linestyle="--")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Depth (m)")
    plt.ylabel("Resistivity (Ohm-m)")
    plt.title(f"Inversion Result: {os.path.basename(edi_file)}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    out_name = f"{os.path.basename(edi_file)}_{model_type}_result.png"
    plt.savefig(out_name)
    print(f"[INFO] Result saved -> {out_name}")


def fine_tune_on_real_data(model_type='lstm', epochs=50, lr=1e-4, lambda_smooth=0.1, lambda_deep=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{model_type}_scalers.pkl", "rb") as f: scalers = pickle.load(f)
    scaler_input = scalers['scaler_input']; scaler_target = scalers['scaler_target']
    checkpoint = torch.load(f"{model_type}_checkpoint.pt", map_location=device, weights_only=False)
    config = checkpoint['config']
    model = create_model(config['model_type'], config['input_size'], config['output_size'], config['hidden_size1'], config['hidden_size2'], config['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.train()
    
    with open("synthetic_data.pkl", "rb") as f: synth = pickle.load(f)
    layer_depths = get_synthetic_depths(synth)
    train_periods = synth['periods']
    
    X_list, Y_list = [], []
    edi_files = [f for f in os.listdir("real_data") if f.endswith(".edi")]
    for edi_file in edi_files:
        site = edi_file.split(".")[0]
        layered_file = os.path.join("real_data", f"{site}Layered.txt")
        if not os.path.exists(layered_file): continue
        periods, rho, phase = read_edi_file(os.path.join("real_data", edi_file))
        if periods is None: continue
        
        from scipy.interpolate import interp1d
        rho_func = interp1d(np.log10(periods), np.log10(rho), fill_value="extrapolate")
        phi_func = interp1d(np.log10(periods), phase, fill_value="extrapolate")
        rho_in = 10 ** rho_func(np.log10(train_periods))
        phase_in = phi_func(np.log10(train_periods))
        input_vec = np.hstack((np.log10(rho_in), phase_in))
        
        true_depths, _, true_res = read_layered_model(layered_file)
        f_true = interp1d(np.log10(true_depths), np.log10(true_res), fill_value="extrapolate")
        y_log = f_true(np.log10(layer_depths))
        X_list.append(input_vec); Y_list.append(y_log)
        
    if len(X_list) == 0: return
    X_real_scaled = scaler_input.transform(np.vstack(X_list))
    Y_real_scaled = scaler_target.transform(np.vstack(Y_list))
    dataset = TensorDataset(torch.FloatTensor(X_real_scaled), torch.FloatTensor(Y_real_scaled))
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=0, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n=== Fine-tuning {model_type.upper()} on real data ===")
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss = depth_weighted_smooth_loss(pred, y, layer_depths, lambda_smooth=lambda_smooth, lambda_deep=lambda_deep)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if (ep + 1) % 10 == 0: print(f"Fine-tune epoch {ep+1}/{epochs}, loss={loss.item():.5f}")
    torch.save({'model_state': model.state_dict(), 'config': config}, f"{model_type}_checkpoint_finetuned.pt")
    print(f"Fine-tuned model saved to {model_type}_checkpoint_finetuned.pt")

def run_physics_benchmark(synthetic_data, scaler, device):
    print("\n=== Running Bostick Benchmark ===")
    y_rho = synthetic_data['y_rho']; X_target = synthetic_data['X_data'] 
    periods = synthetic_data['periods']; true_depths = get_synthetic_depths(synthetic_data)
    metrics_list = []
    indices = np.random.choice(len(y_rho), size=min(1000, len(y_rho)), replace=False)
    
    for i in tqdm(indices):
        b_depths, b_rho = bostick_transform(y_rho[i], periods)
        interp_log_rho = np.interp(np.log10(true_depths), np.log10(b_depths), np.log10(b_rho))
        t_log = np.log10(X_target[i]); p_log = interp_log_rho
        min_len = min(len(t_log), len(p_log))
        t_log = t_log[:min_len]; p_log = p_log[:min_len]
        mse = mean_squared_error(t_log, p_log)
        metrics_list.append({'MSE': mse, 'RMSE': np.sqrt(mse), 'R^2': r2_score(t_log, p_log), 'SSIM': calculate_ssim(t_log, p_log)})
        
    avg = pd.DataFrame(metrics_list).mean()
    print(f"Bostick Results (Log Space): RMSE={avg['RMSE']:.4f}")
    with open('bostick_validation_metrics.pkl', 'wb') as f: pickle.dump(avg.to_dict(), f)

def compare_models(model_types):
    data = []
    model_types_plus = model_types + ['bostick']
    for m in model_types_plus:
        try:
            with open(f'{m}_validation_metrics.pkl', 'rb') as f: 
                d = pickle.load(f); d['Model'] = m.upper(); data.append(d)
        except: pass
    
    if not data: return
    df = pd.DataFrame(data).set_index('Model')
    df.to_csv('model_comparison.csv')
    metrics = ['RMSE', 'R^2', 'SSIM']
    plt.figure(figsize=(15, 5))
    for i, met in enumerate(metrics):
        if met in df.columns:
            plt.subplot(1, 3, i+1)
            colors = ['green' if x == df[met].min() else 'gray' for x in df[met]] if met == 'RMSE' else ['green' if x == df[met].max() else 'gray' for x in df[met]]
            plt.bar(df.index, df[met], color=colors)
            plt.title(met); plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Comparison saved to model_comparison.png")

def run_optuna_tuning(synthetic_data, model_type='lstm', n_trials=50):
    print(f"\n=== Tuning {model_type.upper()} with Optuna ===")
    X_raw_model = np.log10(synthetic_data['X_data'])
    y_rho_raw = np.log10(synthetic_data['y_rho'])
    y_phi_raw = synthetic_data['y_phi']
    X_input_raw = np.hstack((y_rho_raw, y_phi_raw))
    X_train, X_val, y_train, y_val = train_test_split(X_input_raw, X_raw_model, test_size=0.2, random_state=SEED)
    scaler_in, scaler_out = StandardScaler(), StandardScaler()
    X_train = scaler_in.fit_transform(X_train); y_train = scaler_out.fit_transform(y_train)
    X_val = scaler_in.transform(X_val); y_val = scaler_out.transform(y_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_x_gpu = torch.FloatTensor(X_val).to(device)
    val_y_gpu = torch.FloatTensor(y_val).to(device)
    
    def objective(trial):
        torch.cuda.empty_cache()
        h1 = trial.suggest_categorical("h1", [128, 256, 512, 1024])
        h2 = trial.suggest_categorical("h2", [64, 128, 256, 512])
        do = trial.suggest_float("do", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        bs = trial.suggest_categorical("bs", [1024, 2048, 4096])
        model = create_model(model_type, X_train.shape[1], y_train.shape[1], h1, h2, do).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2)
        scaler = torch.amp.GradScaler('cuda')
        val_batch_size = 4096
        
        for epoch in range(5): 
            model.train()
            for x, y in loader:
                optimizer.zero_grad()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = nn.MSELoss()(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            model.eval()
            total_val_loss = 0
            num_val_samples = val_x_gpu.size(0)
            with torch.no_grad():
                for i in range(0, num_val_samples, val_batch_size):
                    x_batch = val_x_gpu[i : i + val_batch_size]
                    y_batch = val_y_gpu[i : i + val_batch_size]
                    with torch.amp.autocast('cuda'):
                        pred = model(x_batch)
                        loss = nn.MSELoss()(pred, y_batch)
                    total_val_loss += loss.item() * x_batch.size(0)
            avg_val_loss = total_val_loss / num_val_samples
            trial.report(avg_val_loss, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return avg_val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best params: {study.best_params}")
    return study.best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='both')
    parser.add_argument('--model', choices=['lstm', 'gru', 'informer', 'all'], default='lstm')
    parser.add_argument('--site', type=str, default='3D01')
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--num-samples', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--force-regen', action='store_true', help="Force regeneration of synthetic data")
    args = parser.parse_args()
    SEED = args.seed; set_seeds(SEED)
    
    if args.model == 'all': model_types = ['lstm', 'gru', 'informer']
    else: model_types = [args.model]
    
    for model_type in model_types:
        print(f"\n{'='*40}\nPROCESSING: {model_type.upper()}\n{'='*40}")
        h1, h2, do, lr, bs = 256, 128, 0.1, 0.001, args.batch_size
        if args.mode in ['train', 'both']:
            if os.path.exists('synthetic_data.pkl') and not args.force_regen:
                print("Loading existing synthetic_data.pkl...")
                with open('synthetic_data.pkl', 'rb') as f: synth = pickle.load(f)
            else: synth = generate_synthetic_data(num_samples=args.num_samples, seed=SEED)
            
            if args.tune:
                best = run_optuna_tuning(synth, model_type, n_trials=args.n_trials)
                h1, h2, do, lr, bs = best['h1'], best['h2'], best['do'], best['lr'], best['bs']
                print("--> Using Tuned Parameters")
            
            if args.kfold: run_kfold_cross_validation(synth, model_type, 5, args.epochs, bs, lr)
            print(f"Training Final {model_type.upper()} Model...")
            train_model(synth, model_type, epochs=args.epochs, batch_size=bs, learning_rate=lr, seed=SEED, hidden_size1=h1, hidden_size2=h2, dropout=do)
            if os.path.exists('real_data'):
                print("Fine-tuning on real data...")
                fine_tune_on_real_data(model_type, epochs=50, lr=1e-4)
            else: print("Skipping fine-tuning: no 'real_data' folder found.")

        if args.mode in ['predict', 'both']:
            if os.path.exists('real_data'):
                files = [f for f in os.listdir('real_data') if f.endswith('.edi')]
                for f in files:
                    base = f.split('.')[0]
                    predict_model(f"real_data/{f}", f"real_data/{base}Layered.txt", model_type)
            else: print("No 'real_data' folder found for prediction.")
    
    if args.compare or (args.model == 'all' and args.mode in ['train', 'both']):
         if os.path.exists('synthetic_data.pkl'):
             with open('synthetic_data.pkl', 'rb') as f: synth = pickle.load(f)
             run_physics_benchmark(synth, None, None)
         compare_models(model_types)
