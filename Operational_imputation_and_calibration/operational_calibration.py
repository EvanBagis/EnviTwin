import warnings
warnings.filterwarnings("ignore")

# ================================
# Core Libraries
# ================================
import os
import gc
import time
import copy
import random
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import schedule
import traceback
from math import radians, sin, cos, sqrt, atan2

# ================================
# Data Handling & Analysis
# ================================
import pandas as pd
import numpy as np

# ================================
# Spatial interpolation
# ================================
from pykrige.ok import OrdinaryKriging

pd.set_option('display.precision', 2)
# pd.options.mode.chained_assignment = None

# ================================
# Visualization
# ================================
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Progress Bar
# ================================
from tqdm import tqdm

# ================================
# Scikit-Learn
# ================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.random import sample_without_replacement
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from scipy.stats import spearmanr, pearsonr, kurtosis, skew

# ================================
# PyTorch & PyTorch Geometric
# ================================
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, GRU

from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
from torch_geometric.nn import (
    GCNConv, GraphConv, ChebConv, NNConv, PNAConv, PDNConv,
    TransformerConv, GATv2Conv, TAGConv, CGConv, GENConv,
    GINEConv, LEConv
)
from torch_geometric.nn.norm import BatchNorm

torch.serialization.add_safe_globals([DataEdgeAttr])

def load(path):
    d = pd.read_csv(path, index_col=0)
    d.index = pd.to_datetime(d.index)
    d.index.name = ''
    return d

def scale_array(arr, method="standard"):
    """
    Scales a 2D NumPy array using Min-Max scaling or Standardization (Z-score normalization).
    
    Parameters:
        arr (np.ndarray): Input 2D NumPy array.
        method (str): Scaling method. Either 'minmax' (default) or 'standard'.
    
    Returns:
        np.ndarray: Scaled 2D array.
    """
    
    if method == "minmax":
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)  # Avoid division by zero
        return (arr - min_val) / (max_val - min_val)
    
    elif method == "standard":
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return np.zeros_like(arr)  # Avoid division by zero
        return (arr - mean) / std

def synchronize_many_dfs(dfs):
	# dfs should be a dictionary {"name": df, ...}
	min_max_values = []
	for name, df in dfs.items():
		min_max_values.append(df.index[0])
		min_max_values.append(df.index[-1])
	start = min(min_max_values)
	stop = max(min_max_values)
	#print(start, stop)
	idx = pd.date_range(start, stop, freq='h')
	new_dfs = []
	for name, df in dfs.items():
		dff = df
		dff = dff[~dff.index.duplicated(keep='first')]
		dff = dff.reindex(idx, fill_value=np.nan)
		dff.columns = dff.columns + "_" + name
		new_dfs.append(dff)
	return pd.concat(new_dfs, axis=1)


def display_metrics(true, pred, returns=False, epsilon=1e-5):
    """
    Computes and displays or returns regression metrics, using MRAE-style normalization
    for all relative error metrics to ensure consistency and numerical stability.

    Parameters:
    ----------
    true : array-like
        Ground truth values (list, np.array, or pd.Series).
    pred : array-like
        Predicted values (list, np.array, or pd.Series).
    returns : bool, optional (default=False)
        If True, returns metrics as a tuple. If False, prints them.
    epsilon : float, optional
        Small constant added to denominator for stability in relative metrics.

    Returns:
    -------
    tuple (only if returns=True):
        (rmse, mae, r2, pearson_r, spearman_r, mbe, index_agreement, 
         rel_rmse, rel_mae, rel_mbe, mrae)
    """
    
    def index_agreement(s, o):
        """Index of agreement between prediction s and observation o."""
        ia = 1 - (np.sum((o - s) ** 2)) / (
            np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o))) ** 2))
        return ia

    # Ensure inputs are numpy arrays
    true = np.asarray(true)
    pred = np.asarray(pred)

    # Core metrics
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    r = pearsonr(true, pred)[0]
    rs = spearmanr(true, pred)[0]
    mbe = np.mean(true - pred)
    ia = index_agreement(pred, true)

    # MRAE-style relative metrics
    denominator = np.abs(true) + epsilon
    rel_abs_errors = np.abs(true - pred) / denominator
    rel_signed_errors = (true - pred) / denominator

    mrae = np.mean(rel_abs_errors)
    rel_mae = np.mean(rel_abs_errors) * 100
    rel_rmse = np.sqrt(np.mean(rel_abs_errors ** 2)) * 100
    rel_mbe = np.mean(rel_signed_errors) * 100

    if returns:
        return rmse, mae, r2, r, rs, mbe, ia, rel_rmse, rel_mae, rel_mbe, mrae
    else:
        print(f"RMSE       = {rmse:.2f}")
        print(f"MAE        = {mae:.2f}")
        print(f"MBE        = {mbe:.2f}")
        print(f"R²         = {r2:.2f}")
        print(f"Pearson r  = {r:.2f}")
        print(f"Spearman r = {rs:.2f}")
        print(f"Index Agr  = {ia:.2f}")
        print(f"Rel RMSE   = {rel_rmse:.2f}%")
        print(f"Rel MAE    = {rel_mae:.2f}%")
        print(f"Rel MBE    = {rel_mbe:.2f}%")
        print(f"MRAE       = {mrae:.4f}")

def make_edges_optimized(df, threshold=0.9, add_self_loops=True):
    """
    Build an undirected edge list from a station-by-feature matrix (rows=stations, cols=features)
    using Pearson correlation as edge weights. Returns (edge_attr, edge_index) where
    edge_index is 2 x E with BOTH directions for every edge and (optionally) self-loops.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows are stations (index), columns are features. Correlations are computed between rows.
    threshold : float
        Keep edges with |corr| >= threshold.
    add_self_loops : bool
        If True, add self-loop per node with weight 1.0.

    Returns
    -------
    edges : torch.FloatTensor [E, 1]
        Edge weights (corr), duplicated for both directions, and (optionally) self-loop weights.
    edge_index : torch.LongTensor [2, E]
        Directed edge list (src, dst) with both directions for undirected pairs.
    """
    # Correlation of rows (stations). rowvar=True by default => rows are variables (what we want).
    # Ensure we pass a numpy array, not a DataFrame, to avoid surprises.
    x_np = df.values.astype(np.float64, copy=False)
    corrs = np.corrcoef(x_np)

    # Upper triangle indices (i < j) to avoid duplicates
    iu, ju = np.triu_indices_from(corrs, k=1)
    edge_vals = corrs[iu, ju]

    # Remove NaNs
    valid = ~np.isnan(edge_vals)
    iu, ju, edge_vals = iu[valid], ju[valid], edge_vals[valid]

    # Threshold by absolute correlation
    keep = np.abs(edge_vals) >= float(threshold)
    iu, ju, edge_vals = iu[keep], ju[keep], edge_vals[keep]

    # If no edges survive, still build tensors cleanly
    if iu.size == 0:
        if add_self_loops:
            n = df.shape[0]
            loop_src = np.arange(n, dtype=np.int64)
            loop_dst = loop_src.copy()
            loop_w = np.ones(n, dtype=np.float32)
            edge_index = torch.from_numpy(np.stack([loop_src, loop_dst], axis=0))
            edges = torch.from_numpy(loop_w.reshape(-1, 1))
            return edges, edge_index
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edges = torch.empty((0, 1), dtype=torch.float32)
            return edges, edge_index

    # Build UNDIRECTED graph: duplicate both directions
    src = np.concatenate([iu, ju], axis=0).astype(np.int64, copy=False)
    dst = np.concatenate([ju, iu], axis=0).astype(np.int64, copy=False)
    w   = np.concatenate([edge_vals, edge_vals], axis=0).astype(np.float32, copy=False)

    # Optionally add self-loops (weight = 1.0). This guarantees every node can at least
    # use its own features; especially important when message passing is directional.
    if add_self_loops:
        n = df.shape[0]
        loop_src = np.arange(n, dtype=np.int64)
        loop_dst = loop_src.copy()
        loop_w   = np.ones(n, dtype=np.float32)

        src = np.concatenate([src, loop_src], axis=0)
        dst = np.concatenate([dst, loop_dst], axis=0)
        w   = np.concatenate([w,   loop_w  ], axis=0)

    edge_index = torch.from_numpy(np.stack([src, dst], axis=0))
    edges = torch.from_numpy(w.reshape(-1, 1))
    return edges, edge_index

# print once per process
_graph_debug_printed = False

def generate_mask(collocated_nodes, pol_df, search_list):
    cols = list(pol_df.columns)
    mask = np.full(len(cols), False)
    indices = {}
    for gr, node in collocated_nodes.items():
        if node in cols:
            j = cols.index(node)
            mask[j] = True
            indices[node] = j
        else:
            print(f"⚠️ generate_mask: node '{node}' (from {gr}) not found in columns")
    return mask, indices

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def generate_graph_x(target_ref, target_time, dfs, corr_threshold):
    # === Step 1: Extract multi-variate features ===
    end = target_time
    stations = dfs[list(dfs.keys())[0]].columns  # original station names
    features = {s: [] for s in stations}
    

    for col in dfs.keys():  # loop over variables
        df = dfs[col]
        if end not in df.index:
            return None  # missing data

        row = df.loc[end]
        for s in stations:
            val = row[s]
            if np.isnan(val):
                return None
            features[s].append(val)

    x = pd.DataFrame(features).T  # shape: [stations, num_vars]

    # === Step 2: Build graph ===
    if target_ref=='PM2.5':
        edges, edge_index = make_edges_optimized(x, threshold=corr_threshold, add_self_loops=True)
    else:
        edges, edge_index = make_edges_from_coords(list(x.index), coordinates)
    
    make_edges_from_coords
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_tensor = torch.tensor(x_scaled.astype(np.float32))

    # === Step 3: Construct graph object ===
    data_obj = Data(x=x_tensor, edge_index=edge_index, edge_attr=edges)
    data_obj.station_names = list(x.index)  # <--- Add station names here
    return data_obj


def prepare_model_input(model, graphs, features_mask):
    x_seq = torch.stack([g.x[:, features_mask] for g in graphs])  # [T, N, F]

    if isinstance(model, GCN):
        g = graphs[-1]
        return g.x[:, features_mask], g.edge_index, g.edge_attr

    elif isinstance(model, TemporalGCN3) or isinstance(model, TemporalGCN4):
        edge_index_seq = [g.edge_index for g in graphs]  # list of [2, E] per time
        edge_attr_seq = [g.edge_attr for g in graphs]    # list of [E, d] per time
        return x_seq, edge_index_seq, edge_attr_seq

    else:
        edge_index_seq = [g.edge_index for g in graphs]
        edge_attr_seq = [g.edge_attr for g in graphs]
        return x_seq, edge_index_seq, edge_attr_seq

# ===========================
# TRAIN ONE STEP
# ===========================
def train_one(target_ref, model, train_seq, val_seq, features_mask, criterion,
              optimizer=None, max_epochs=200, patience=5):
    """
    Train the model on a single (train, val) sequence pair.
    Returns the trained model and best_val_loss.
    (No saving/loading inside.)
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_tr = train_seq[-1].train_mask_y
    mask_val = val_seq[-1].train_mask_y

    inputs = prepare_model_input(model, train_seq, features_mask)
    val_inputs = prepare_model_input(model, val_seq, features_mask)

    if target_ref!="O3":
        y_tr = torch.log1p(train_seq[-1].y).to(device)
        y_val = torch.log1p(val_seq[-1].y).to(device)
    else:
        y_tr = train_seq[-1].y.to(device)
        y_val = val_seq[-1].y.to(device)
    
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(*inputs)
        loss = criterion(out[mask_tr].squeeze(), y_tr[mask_tr])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(*val_inputs)
            val_loss = criterion(out_val[mask_val].squeeze(), y_val[mask_val])

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    torch.cuda.empty_cache()
    gc.collect()
    return model


# ===========================
# Predict one sequence
# ===========================
def predict_one(target_ref, model, test_seq, features_mask):
    """
    Predict on a single test sequence with a provided model.
    Returns:
        df_all_preds (DataFrame): All predictions for each station with original names.
    """
    device = next(model.parameters()).device
    test_inputs = prepare_model_input(model, test_seq, features_mask)

    model.eval()
    with torch.no_grad():
        out_te = model(*test_inputs)

    # Get station names from graph
    last_graph = test_seq[-1]
    station_names = getattr(last_graph, "station_names", [f"station{i}" for i in range(out_te.size(0))])

    # Ensure correct shape
    if target_ref=="O3" or target_ref=="CO":
        all_preds = out_te.detach().cpu().numpy()
    else:
        all_preds = torch.expm1(out_te).detach().cpu().numpy()
    if all_preds.ndim == 2 and all_preds.shape[1] == 1:
        all_preds = all_preds[:, 0]

    all_preds_dict = {f"{station_names[i]}_p": all_preds[i] for i in range(len(station_names))}
    df_all_preds = pd.DataFrame([all_preds_dict])
    df_all_preds[df_all_preds < 0]=0
    #global_median = df_all_preds.median().median()
    #df_all_preds = df_all_preds.mask(df_all_preds > 5 * global_median, global_median)

    torch.cuda.empty_cache()
    gc.collect()
    return df_all_preds
    
def save_model(model, path: str, config: dict = None):
    """
    Save model weights + architecture config.
    Pass `config` dict with the model init parameters.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "config": config,  # store init params like input_dim, hidden_dim, etc.
    }
    torch.save(checkpoint, path)
    print(f"✅ Model saved to {path}")


def load_model(model_class, path: str, device="cpu"):
    """
    Load model from checkpoint.
    Rebuilds using saved config automatically.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = model_class(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Model loaded from {path}")
    return model, config


# === Helper to convert predictions/observations ===
def to_station_dict(preds, trues, mask, prefix="station"):
    """
    Convert preds/trues tensors into dicts with keys station{i}_p and station{i}_y.
    """
    idxs = torch.nonzero(mask, as_tuple=False).flatten().cpu().numpy()
    preds_dict = {f"{prefix}{i}_p": preds[j] for i, j in enumerate(idxs)}
    trues_dict = {f"{prefix}{i}_y": trues[j] for i, j in enumerate(idxs)}
    return preds_dict, trues_dict


# ===========================
# MODEL DEFINITIONS
# ===========================
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(input_dim, hidden_dim, edge_dim=1, heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(TransformerConv(hidden_dim * num_heads, hidden_dim, edge_dim=1, heads=num_heads, dropout=dropout))
        self.out = TransformerConv(hidden_dim * num_heads, 1, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv in self.layers:
            h = F.leaky_relu(conv(h, edge_index, edge_attr))
        return self.out(h, edge_index, edge_attr)

class TemporalGCN1(nn.Module):  # GNN before GRU
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            TransformerConv(input_dim, hidden_dim, edge_dim=1, heads=num_heads, dropout=dropout)
        ] + [
            TransformerConv(hidden_dim * num_heads, hidden_dim, edge_dim=1, heads=num_heads, dropout=dropout)
            for _ in range(num_layers - 2)
        ])
        self.gru = nn.GRU(hidden_dim * num_heads, hidden_dim * num_heads, batch_first=True)
        self.out = TransformerConv(hidden_dim * num_heads, 1, edge_dim=1)

    def forward(self, x_seq, edge_index_seq, edge_attr_seq):
        gnn_outs = []
        for t in range(x_seq.size(0)):
            h = x_seq[t]
            for conv in self.gnn_layers:
                h = F.leaky_relu(conv(h, edge_index_seq[t], edge_attr_seq[t]))
            gnn_outs.append(h)
        gnn_stack = torch.stack(gnn_outs, dim=0).permute(1, 0, 2)  # [N, T, H]
        _, h_final = self.gru(gnn_stack)
        return self.out(h_final.squeeze(0), edge_index_seq[-1], edge_attr_seq[-1])

class TemporalGCN2(nn.Module):  # GRU before GNN
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim * num_heads, batch_first=False)
        self.gnn_layers = nn.ModuleList([
            TransformerConv(hidden_dim * num_heads, hidden_dim, edge_dim=1, heads=num_heads, dropout=dropout)
            for _ in range(num_layers - 1)
        ])
        self.out = TransformerConv(hidden_dim * num_heads, 1, edge_dim=1)

    def forward(self, x_seq, edge_index_seq, edge_attr_seq):
        gru_out, _ = self.gru(x_seq)  # [T, N, H]
        h = gru_out.mean(dim=0)      # [N, H]
        for conv in self.gnn_layers:
            h = F.leaky_relu(conv(h, edge_index_seq[-1], edge_attr_seq[-1]))
        return self.out(h, edge_index_seq[-1], edge_attr_seq[-1])


class TemporalGCN3(nn.Module):  # GRU + GNN per timestep
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.expects_temporal_graphs = True
        self.gru = nn.GRU(input_dim, hidden_dim * num_heads, batch_first=False)
        self.gnn_layers = nn.ModuleList([
            TransformerConv(hidden_dim * num_heads, hidden_dim * num_heads, edge_dim=1, heads=1, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out = TransformerConv(hidden_dim * num_heads, 1, edge_dim=1)

    def forward(self, x_seq, edge_index_seq, edge_attr_seq):
        gru_out, _ = self.gru(x_seq)  # [T, N, H]
        h_list = []

        for t in range(gru_out.size(0)):
            h_t = gru_out[t]  # [N, H]
            edge_index = edge_index_seq[t]
            edge_attr = edge_attr_seq[t]
            for conv in self.gnn_layers:
                h_t = F.leaky_relu(conv(h_t, edge_index, edge_attr))
            h_list.append(h_t)

        h = torch.stack(h_list).mean(dim=0)  # [N, H]
        return self.out(h, edge_index_seq[-1], edge_attr_seq[-1])

class TemporalGCN4(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2, k_top=3):
        super().__init__()
        self.k_top = k_top
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.gru = nn.GRU(input_dim, hidden_dim * num_heads, batch_first=False)

        self.gnn_layers = nn.ModuleList([
            TransformerConv(hidden_dim * num_heads, hidden_dim * num_heads, edge_dim=1, heads=1, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.out = TransformerConv(hidden_dim * num_heads, 1, edge_dim=1)

        self.attn_proj = nn.Linear(hidden_dim * num_heads, 1)

    def forward(self, x_seq, edge_index_seq, edge_attr_seq):
        """
        x_seq: [T, N, F]
        edge_index_seq: list of T [2, E_t]
        edge_attr_seq: list of T [E_t, edge_dim]
        """
        gru_out, _ = self.gru(x_seq)  # [T, N, H]
        T, N, H = gru_out.size()

        h_list = []
        for t in range(T):
            h_t = gru_out[t]  # [N, H]
            for conv in self.gnn_layers:
                h_t = F.leaky_relu(conv(h_t, edge_index_seq[t], edge_attr_seq[t]))
            h_list.append(h_t)  # each h_t: [N, H]

        h_stack = torch.stack(h_list, dim=0)  # [T, N, H]

        # Compute attention scores: [T, N]
        scores = self.attn_proj(h_stack).squeeze(-1)

        # Apply softmax over time per node → [T, N]
        attn_weights = torch.softmax(scores, dim=0)

        # --- Top-k sparse attention ---
        k = min(self.k_top, T)
        topk_vals, topk_idx = torch.topk(attn_weights, k=k, dim=0)  # each column is a node

        sparse_mask = torch.zeros_like(attn_weights)
        sparse_mask.scatter_(0, topk_idx, 1.0)
        attn_weights_sparse = attn_weights * sparse_mask
        attn_weights_sparse = attn_weights_sparse / (attn_weights_sparse.sum(dim=0, keepdim=True) + 1e-8)  # normalize

        # Weighted sum over time using sparse attention
        h_attn = (attn_weights_sparse.unsqueeze(-1) * h_stack).sum(dim=0)  # [N, H]

        out = self.out(h_attn, edge_index_seq[-1], edge_attr_seq[-1])  # [N, 1]
        return out


class TemporalGCN5(nn.Module):  # GRU + GNN per timestep + sparse attention + residual avg
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2, k_top=3, attn_weight=0.8):
        super().__init__()
        self.k_top = k_top
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_weight = attn_weight
        self.expects_temporal_graphs = True

        self.gru = nn.GRU(input_dim, hidden_dim * num_heads, batch_first=False)

        self.gnn_layers = nn.ModuleList([
            TransformerConv(hidden_dim * num_heads, hidden_dim * num_heads, edge_dim=1, heads=1, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.out = TransformerConv(hidden_dim * num_heads, 1, edge_dim=1)
        self.attn_proj = nn.Linear(hidden_dim * num_heads, 1)

    def forward(self, x_seq, edge_index_seq, edge_attr_seq):
        gru_out, _ = self.gru(x_seq)  # [T, N, H]
        T, N, H = gru_out.size()

        h_list = []
        for t in range(T):
            h_t = gru_out[t]
            for conv in self.gnn_layers:
                h_t = F.leaky_relu(conv(h_t, edge_index_seq[t], edge_attr_seq[t]))
            h_list.append(h_t)

        h_stack = torch.stack(h_list, dim=0)  # [T, N, H]

        # Attention scores [T, N]
        scores = self.attn_proj(h_stack).squeeze(-1)
        #attn_weights = torch.softmax(scores, dim=0)  # [T, N]

        attn_weights = torch.sigmoid(scores)  # [T, N]
        attn_weights = attn_weights / (attn_weights.sum(dim=0, keepdim=True) + 1e-8)

        # Sparse attention
        k = min(self.k_top, T)
        topk_vals, topk_idx = torch.topk(attn_weights, k=k, dim=0)
        sparse_mask = torch.zeros_like(attn_weights)
        sparse_mask.scatter_(0, topk_idx, 1.0)
        attn_weights_sparse = attn_weights * sparse_mask
        attn_weights_sparse = attn_weights_sparse / (attn_weights_sparse.sum(dim=0, keepdim=True) + 1e-8)

        h_attn = (attn_weights_sparse.unsqueeze(-1) * h_stack).sum(dim=0)  # [N, H]
        h_avg = h_stack.mean(dim=0)  # [N, H]

        # Weighted fusion of attention and mean
        h_final = self.attn_weight * h_attn + (1 - self.attn_weight) * h_avg

        out = self.out(h_final, edge_index_seq[-1], edge_attr_seq[-1])  # [N, 1]
        return out

coordinates = {
        'nkastom04': {'lat': 40.631897, 'lon': 22.9406710000001},
        'nkastom07': {'lat': 40.645599, 'lon': 22.858091}, # EXCLUDE BECAUSE DATA ARE CLOSE TO ZERO AND VERY FEW DATA
        #'nkastom08': {'lat': 40.57679000000005, 'lon': 22.970080000000053}, EXCLUDE BECAUSE THE DATA ARE SHIT
        'nkastom09': {'lat': 40.6339266, 'lon': 22.9729144},
        'nkastom10': {'lat': 40.686661, 'lon': 22.952488},
        'nkastom11': {'lat': 40.58216, 'lon': 22.95007},
        'nkastom12': {'lat': 40.641326, 'lon': 22.913513},
        'nkastom13': {'lat': 40.641326, 'lon': 22.95575},
        'nkastom14': {'lat': 40.673396000000004, 'lon': 22.928832000000057},
        'nkastom15': {'lat': 40.6375520955108, 'lon': 22.9410431299578},
        'nkastom16': {'lat': 40.63373, 'lon': 22.945150000000012},
        'nkastom17': {'lat': 40.644282, 'lon': 22.958302},
        'nkastom18': {'lat': 40.62678, 'lon': 22.9612500000001},
        'nkastom20': {'lat': 40.67351800000001, 'lon': 22.89320299999997},
        'nkastom22': {'lat': 40.65809499999999, 'lon': 22.80154600000003},
        #'nkastom23': {'lat': 40.60114, 'lon': 22.960505}, ALSO 19, 28 AND 29 ARE EXCLUDED BUT ARE NOT ON THIS LIST ANYWAY
        'nkastom24': {'lat': 40.6009789520081, 'lon': 22.9605397132275},
        'nkastom32': {'lat': 40.652402, 'lon': 22.9415289999999},
        'nkastom34': {'lat': 40.65349, 'lon': 22.922811},
        'nkastom35': {'lat': 40.63547, 'lon': 22.94154},
        'nkastom36': {'lat': 40.631992, 'lon': 22.944305},
        'nkastom37': {'lat': 40.60698, 'lon': 22.9552200000001},
        'nkastom38': {'lat': 40.6305, 'lon': 22.94868},
        'nkastom39': {'lat': 40.68281189704966, 'lon': 22.812980513855564},
        'nkastom40': {'lat': 40.667861, 'lon': 22.910971},
        'nkastom41': {'lat': 40.640383, 'lon': 22.935223},
        'nkastom42': {'lat': 40.6132239, 'lon': 22.9608034},
        'nkastom43': {'lat': 40.658643, 'lon': 22.942914},
        'nkastom44': {'lat': 40.59957, 'lon': 22.9880499999999},
        'nkastom45': {'lat': 40.63759, 'lon': 22.94902}
    }

def haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance (in km) between two coordinates."""
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def make_edges_from_coords(station_names, coordinates, add_self_loops=True, min_distance=1e-3):
    """
    Build edges based on geographic distance between stations.
    
    Parameters
    ----------
    station_names : list[str]
        Ordered list of station IDs (must align with x.index in the graph data).
    coordinates : dict[str, dict]
        Mapping: station -> {lat, lon}.
    add_self_loops : bool
        Whether to add self-loops with weight=1.0.
    min_distance : float
        Small epsilon to avoid division by zero (km).

    Returns
    -------
    edge_attr : torch.FloatTensor [E, 1]
    edge_index : torch.LongTensor [2, E]
    """
    N = len(station_names)
    src, dst, weights = [], [], []

    for i in range(N):
        s1 = station_names[i]
        if s1 not in coordinates:
            continue
        lat1, lon1 = coordinates[s1]["lat"], coordinates[s1]["lon"]

        for j in range(i + 1, N):
            s2 = station_names[j]
            if s2 not in coordinates:
                continue
            lat2, lon2 = coordinates[s2]["lat"], coordinates[s2]["lon"]

            dist = haversine(lat1, lon1, lat2, lon2)
            w = 1.0 / max(dist, min_distance)

            # undirected: add both (i→j) and (j→i)
            src.extend([i, j])
            dst.extend([j, i])
            weights.extend([w, w])

    # optional self-loops
    if add_self_loops:
        for i in range(N):
            src.append(i)
            dst.append(i)
            weights.append(1.0)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.array(weights, dtype=np.float32).reshape(-1, 1))
    return edge_attr, edge_index

def load_sequences_incremental_update(path, files, t_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_seq = [torch.load(f'{path}{files[j]}', weights_only=False).to(device) for j in range(t_dim)]
    val_seq = [torch.load(f'{path}{files[j + 1]}', weights_only=False).to(device) for j in range(t_dim)]
    return train_seq, val_seq

MODEL_CLASSES = {
    "GCN": GCN,
    "TemporalGCN1": TemporalGCN1,
    "TemporalGCN2": TemporalGCN2,
    "TemporalGCN3": TemporalGCN3,
    "TemporalGCN4": TemporalGCN4,
    "TemporalGCN5": TemporalGCN5,
}

CONFIG = {
    # === Data Paths ===
    "load_path": "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/",
    "save_path": "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/aggregates/",
    "save_path_imputed": "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/imputed_aggregates/",
    "graph_root": "/home/envitwin/Desktop/venvs/EnviTwin/Operational_imputation_and_calibration/graph_data/",
    "models_path": "/home/envitwin/Desktop/venvs/EnviTwin/Operational_imputation_and_calibration/calibration_models/",
    "preds_path": "/home/envitwin/Desktop/venvs/EnviTwin/Operational_imputation_and_calibration/predictions/",
    "plot_dir": "./plots",

    # === Data & Target Settings ===
    "variables": ['pm_10', 'pm_1_0', 'pm_2_5', 'humidity', 'pressure', 'temperature'],
    "pollutants": ["PM2.5", "PM10", "NO2", "O3", "CO"],  # possibly redundant with above

    # === Preprocessing / Filtering ===
    "exclude": ['nkastom07', 'nkastom08', 'nkastom19', 'nkastom23', 'nkastom28', 'nkastom29'],
    "corr_threshold": 0,
    "split": 1000,
    "min_value": 3.0,

    # === Time Window Control ===
    "start_timestamp": "2020-01-01 00:00:00+00:00",
    "window_size_multiplier": 2,
    "historical_plot_last_days": 30,
    "latest_plot_last_days": 7,

    # === Runtime Settings ===
    "run_once_on_start": True,

    # === Training Settings ===
    "train": {
        "model": "TemporalGCN2",  # <-- Model selection here
        "criterion": torch.nn.HuberLoss(),
        "max_epochs": 200,
        "patience": 5,
        "num_of_nodes": 32,
        "num_of_layers": 4,
        "num_of_heads": 4,
        "learning_rate": 0.001,
        "dropout_rate": 0.0,
        "features_mask": [True, True, True, True, True, True],
        "t_dim": 24,
    },
}

def update_model(target_ref):
    print(f"\n🔧 Updating model for: {target_ref}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === Load configuration
    graph_path = f"{CONFIG['graph_root']}{target_ref}/"
    models_path = f"{CONFIG['models_path']}{target_ref}/"
    preds_path = CONFIG["preds_path"]
    t_dim = CONFIG["train"]["t_dim"]
    features_mask = CONFIG["train"]["features_mask"]
    corr_threshold = CONFIG["corr_threshold"]

    # === Load reference data
    ref_path = f"/home/envitwin/Desktop/venvs/databases/data/eea_data/data/{target_ref}/thessaloniki/raw/"
    ref_stations = [f for f in os.listdir(ref_path) if f.endswith(".csv")]
    ref_dfs = {f[:-4]: load(ref_path + f) for f in ref_stations}
    ref_df = synchronize_many_dfs(ref_dfs).loc['2020-01-01 00:00:00+00:00':]

    # === Load operational data
    save_path_imputed = "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/imputed_aggregates/"
    paths = sorted(f for f in os.listdir(save_path_imputed) if f.endswith('imputed_operational.csv'))
    dfs = {f[:-4]: load(save_path_imputed + f) for f in paths}

    # === Determine timestamps
    graph_files = sorted(f for f in os.listdir(graph_path) if f.endswith('.pt'))
    if not graph_files:
        print(f"[{target_ref}] ⚠️ No graph files found.")
        return

    previous_time = graph_files[-1].strip('.pt')  # last graph time
    operational_time = dfs[list(dfs.keys())[0]].index[-1]
    input_timestamps = sorted([dfs[list(dfs.keys())[0]].index[-i] for i in range(1, t_dim + 1)])

    # === Build sequences
    train_seq_timestamps = pd.date_range(
        pd.to_datetime(previous_time) - timedelta(hours=t_dim),
        pd.to_datetime(previous_time),
        freq='h'
    )
    files = [f"{t}.pt" for t in train_seq_timestamps]
    train_seq, val_seq = load_sequences_incremental_update(graph_path, files, t_dim=t_dim)
    test_seq = [generate_graph_x(target_ref, operational_time, dfs, corr_threshold).to(device) for timestep in input_timestamps]

    # === Model handling (timestamped only)
    ensure_dir(models_path)
    model_class = MODEL_CLASSES.get(CONFIG["train"]["model"], "GCN")

    # Load latest checkpoint if available
    model_files = sorted(f for f in os.listdir(models_path) if f.endswith(".pt"))
    resume_model, config = None, None
    if model_files:
        last_model_file = model_files[-1]
        resume_model, config = load_model(
            model_class,
            path=os.path.join(models_path, last_model_file),
            device=device
        )
        print(f"📂 Loaded last checkpoint: {last_model_file}")
    else:
        print("🆕 No previous model found, training from scratch.")

    # Train
    model = resume_model if resume_model else model_class(
        input_dim=len(features_mask),
        hidden_dim=CONFIG["train"]["num_of_nodes"],
        num_layers=CONFIG["train"]["num_of_layers"],
        num_heads=CONFIG["train"]["num_of_heads"],
        dropout=CONFIG["train"]["dropout_rate"],
    ).to(device)

    model = train_one(
        target_ref, model,
        train_seq, val_seq,
        features_mask, CONFIG["train"]["criterion"],
        max_epochs=CONFIG["train"]["max_epochs"],
        patience=CONFIG["train"]["patience"]
    )

    # Save new checkpoint with timestamp
    timestamp_str = pd.to_datetime(previous_time)
    save_path = os.path.join(models_path, f"{timestamp_str}.pt")
    save_model(model, path=save_path, config=config)
    print(f"💾 Saved new checkpoint: {save_path}")

    # === Optional: prune old checkpoints (keep only last N)
    max_models_to_keep = 10
    model_files = sorted(f for f in os.listdir(models_path) if f.endswith(".pt"))
    if len(model_files) > max_models_to_keep:
        to_remove = model_files[:-max_models_to_keep]
        for rm in to_remove:
            os.remove(os.path.join(models_path, rm))
            print(f"🗑️ Removed old checkpoint: {rm}")

    # === Predict
    all_preds = predict_one(target_ref, model, test_seq, features_mask)
    print(f"✅ Predictions for {target_ref} at {operational_time}:\n{all_preds}\n")


def update():
    print(f"\n🕙 Update triggered at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    for target_ref in CONFIG["pollutants"]:
        try:
            update_model(target_ref)
        except Exception as e:
            print(f"[{target_ref}] ❌ Error: {e}")
            traceback.print_exc()

def main():
    if CONFIG["run_once_on_start"]:
        update()

    schedule.every().hour.at(":14").do(update)

    print("⏲️ Scheduler started. Running every hour at :00. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting on user interrupt.")
        sys.exit(0)
