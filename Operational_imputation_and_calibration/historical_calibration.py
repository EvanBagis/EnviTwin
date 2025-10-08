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
import tqdm
import sys
import schedule
import traceback

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

def evaluate_metrics_block(predictions, observations,
                           num_features, num_of_nodes,
                           num_of_layers, num_of_heads,
                           dropout_rate, learning_rate,
                           mode=None):
    # merge predictions and observations
    f_p = pd.concat(predictions).reset_index(drop=True)
    f_y = pd.concat(observations).reset_index(drop=True)
    f = pd.concat([f_p, f_y], axis=1).dropna()
    #print(f_p.shape, f_y.shape, f.shape)
    #print(f_p.isna().sum())
    #print(f_y.isna().sum())
    #print(f)

    # Identify all stations automatically
    stations = sorted(set([c.split("_")[0] for c in f.columns if c.endswith("_p")]))

    rmse_list, mae_list, r2_list, ia_list, mrae_list = [], [], [], [], []

    for st in stations:
        y_true = f[f"{st}_y"]
        y_pred = f[f"{st}_p"]

        rmse, mae, r2, r, rs, mbe, ia, rel_rmse, rel_mae, rel_mbe, mrae = display_metrics(
            y_true, y_pred, returns=True
        )

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        ia_list.append(ia)
        mrae_list.append(mrae)

        if mode != "val":
            print(f"Station {st}: R²={r2:.4f} | RMSE={rmse:.4f} | MRAE={mrae:.4f} | IOA={ia:.4f}")

    # Aggregate
    final_rmse = np.mean(rmse_list)
    final_r2 = np.mean(r2_list)
    final_ia = np.mean(ia_list)
    final_mrae = np.mean(mrae_list)

    if mode != "val":
        print(
            f"Features: {num_features} | Nodes: {num_of_nodes} | "
            f"Layers: {num_of_layers} | Heads: {num_of_heads} | "
            f"Dropout: {dropout_rate:.2f} | LR: {learning_rate:.1e}"
        )
        print(
            f"Test Avg: MRAE={final_mrae:.4f} | RMSE={final_rmse:.4f} "
            f"| R²={final_r2:.4f} | IOA={final_ia:.4f}"
        )

    return {
        "mae": final_mrae,
        "rmse": final_rmse,
        "r2": final_r2,
        "ioa": final_ia,
    }


def plot_predictions_with_metrics(predictions, observations):
    # Merge predictions and observations
    f_p = pd.concat(predictions).reset_index(drop=True)
    f_y = pd.concat(observations).reset_index(drop=True)
    f = pd.concat([f_p, f_y], axis=1).dropna()

    # Automatically detect station names
    stations = sorted(set(c.split("_")[0] for c in f.columns if c.endswith("_p")))

    for station in stations:
        y = f[f"{station}_y"]
        y_hat = f[f"{station}_p"]

        rmse, mae, r2, r, rs, mbe, ia, rel_rmse, rel_mae, rel_mbe, mrae = display_metrics(
            y, y_hat, returns=True
        )

        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(y.values, label="True", linewidth=1.5)
        plt.plot(y_hat.values, label="Predicted", linewidth=1.5)
        plt.title(f"{station.upper()} | RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()

        # Metrics box
        textstr = (
            f"R² = {r2:.2f}\n"
            f"Rel RMSE = {rel_rmse:.2f}%\n"
            f"Rel MAE = {rel_mae:.2f}%\n"
            f"Rel MBE = {rel_mbe:.2f}%\n"
            f"MRAE = {mrae:.4f}"
        )
        plt.gca().text(
            0.01,
            0.95,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()


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

def train_model(
    model,
    data_sequences,
    features_mask,
    *,
    criterion,
    lr=1e-3,
    t_dim=24,
    max_epochs=200,
    patience=5,
    weight_decay=1e-5,
    timestamps=None,
    pollutant="O3",   # <--- added parameter
):
    """Train model incrementally on provided data sequences with station-aware outputs.
    If pollutant == "O3", skip log-transformations.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    all_predictions = []
    predictions, observations = [], []
    val_predictions, val_observations = [], []
    kept_timestamps = [] if timestamps is not None else None

    use_log = pollutant.upper() not in ["O3", "CO"]  # <--- log transform only if not O3

    for k, (train_seq, val_seq, test_seq) in enumerate(tqdm(data_sequences, desc="Training windows")):
        if len(train_seq) < t_dim or len(val_seq) < t_dim or len(test_seq) < t_dim:
            print(f"⚠️ skip window {k}: bad lengths")
            continue

        try:
            mask_tr = train_seq[-1].train_mask_y
            mask_val = val_seq[-1].train_mask_y
            mask_te = test_seq[-1].train_mask_y

            if (not mask_tr.any()) or (not mask_val.any()) or (not mask_te.any()):
                print(f"⚠️ skip window {k}: empty masks.")
                continue

            # === Training loop ===
            best_val_loss, epochs_no_improve = float("inf"), 0
            best_state = None

            for epoch in range(max_epochs):
                model.train()
                optimizer.zero_grad()
                out = model(*prepare_model_input(model, train_seq, features_mask))

                y_tr = train_seq[-1].y
                if use_log:
                    y_tr = torch.log1p(y_tr)

                loss = criterion(out[mask_tr].squeeze(), y_tr[mask_tr])
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    out_val = model(*prepare_model_input(model, val_seq, features_mask))
                    y_val = val_seq[-1].y
                    if use_log:
                        y_val = torch.log1p(y_val)

                    val_loss = criterion(out_val[mask_val].squeeze(), y_val[mask_val])

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

            if best_state:
                model.load_state_dict(best_state)

            # === Evaluate ===
            model.eval()
            with torch.no_grad():
                out_te = model(*prepare_model_input(model, test_seq, features_mask))
                out_val = model(*prepare_model_input(model, val_seq, features_mask))

            # station IDs preserved from graph
            station_names = getattr(test_seq[-1], "station_names",
                                    [f"station{i}" for i in range(test_seq[-1].num_nodes)])
            val_station_names = getattr(val_seq[-1], "station_names",
                                        [f"station{i}" for i in range(val_seq[-1].num_nodes)])

            # === Convert predictions back ===
            if use_log:
                preds_te = torch.expm1(out_te[mask_te]).cpu().numpy().reshape(-1)
                preds_val = torch.expm1(out_val[mask_val]).cpu().numpy().reshape(-1)
                all_preds = torch.expm1(out_te).cpu().numpy().reshape(-1)
            else:
                preds_te = out_te[mask_te].cpu().numpy().reshape(-1)
                preds_val = out_val[mask_val].cpu().numpy().reshape(-1)
                all_preds = out_te.cpu().numpy().reshape(-1)

            trues_te = test_seq[-1].y[mask_te].cpu().numpy().reshape(-1)
            trues_val = val_seq[-1].y[mask_val].cpu().numpy().reshape(-1)

            mask_te_idx = np.where(mask_te.cpu().numpy())[0]
            mask_val_idx = np.where(mask_val.cpu().numpy())[0]

            # supervised-only dicts
            preds_te_dict = {f"{station_names[i]}_p": preds_te[j] for j, i in enumerate(mask_te_idx)}
            trues_te_dict = {f"{station_names[i]}_y": trues_te[j] for j, i in enumerate(mask_te_idx)}
            preds_val_dict = {f"{val_station_names[i]}_p": preds_val[j] for j, i in enumerate(mask_val_idx)}
            trues_val_dict = {f"{val_station_names[i]}_y": trues_val[j] for j, i in enumerate(mask_val_idx)}

            predictions.append(pd.DataFrame([preds_te_dict]))
            observations.append(pd.DataFrame([trues_te_dict]))
            val_predictions.append(pd.DataFrame([preds_val_dict]))
            val_observations.append(pd.DataFrame([trues_val_dict]))

            # all nodes dict
            all_preds_dict = dict(zip([f"{s}_p" for s in station_names], all_preds))
            all_predictions.append(pd.DataFrame([all_preds_dict]))

            if kept_timestamps is not None:
                kept_timestamps.append(timestamps[k])

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"⚠️ skip window {k}: {e}")
            continue

    # build df_all_preds once at the end
    df_all_preds = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    df_all_preds[df_all_preds < 0]=0
    global_median = df_all_preds.median().median()
    df_all_preds = df_all_preds.mask(df_all_preds > 5 * global_median, global_median)
    return predictions, observations, val_predictions, val_observations, model, kept_timestamps, df_all_preds

CONFIG = {
    "preds_path": "/home/envitwin/Desktop/venvs/EnviTwin/Operational_imputation_and_calibration/predictions/",
    "models_path": "/home/envitwin/Desktop/venvs/EnviTwin/Operational_imputation_and_calibration/calibration_models/",
    "graph_root": "/home/envitwin/Desktop/venvs/EnviTwin/Operational_imputation_and_calibration/graph_data/",
    "pollutants": ["O3", "PM2.5", "PM10", "NO2", "CO"],
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
        "features_mask": [True, True, True, True, True, True],#['humidity pm_10 pm_1_0 pm_2_5 pressure 'temperature]
        "t_dim": 24,
    },
    "run_once_on_start": True,
}

# ===========================
# UPDATE MODEL
# ===========================
MODEL_CLASSES = {
    "GCN": GCN,
    "TemporalGCN1": TemporalGCN1,
    "TemporalGCN2": TemporalGCN2,
    "TemporalGCN3": TemporalGCN3,
    "TemporalGCN4": TemporalGCN4,
    "TemporalGCN5": TemporalGCN5,
}

# ===========================
# UTILITY FUNCTION
# ===========================
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

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_sequences(path, files, t_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_sequences = []
    timestamps =  []
    for i in tqdm(range(len(files) - t_dim - 2)):
        train_seq = [torch.load(f'{path}{files[i + j]}', weights_only=False).to(device) for j in range(t_dim)]
        val_seq = [torch.load(f'{path}{files[i + j + 1]}', weights_only=False).to(device) for j in range(t_dim)]
        test_seq = [torch.load(f'{path}{files[i + j + 2]}',  weights_only=False).to(device) for j in range(t_dim)]
        data_sequences.append((train_seq, val_seq, test_seq))
        test_timestamp = str(files[i + t_dim + 1]).strip('.pt')
        timestamps.append(test_timestamp)
    return data_sequences, timestamps

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

    Parameters:
    ----------
    model_class : class
        The model class to instantiate.
    path : str
        Path to the checkpoint file.
    device : str or torch.device, optional
        Device on which to load the model (default: "cpu").

    Returns:
    -------
    model : nn.Module
        The loaded model on the specified device.
    config : dict
        The model configuration used for initialization.
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = model_class(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set to eval mode for inference
    print(f"✅ Model loaded from {path} on {device}")
    return model, config


def update_model(target_ref: str):
    """Incrementally train and update the calibration model for one pollutant."""
    train_cfg = CONFIG["train"]
    num_feates = sum(train_cfg["features_mask"])

    preds_path = CONFIG["preds_path"]
    models_path = CONFIG["models_path"]
    graph_path = CONFIG["graph_root"] + f"{target_ref}/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted(f for f in os.listdir(graph_path) if f.endswith(".pt"))
    if not files:
        print(f"[{target_ref}] ⚠️ No graph files found.")
        return

    preds_file = preds_path + f"{target_ref}/preds_collocated.csv"
    model_dir = models_path + f"{target_ref}/"   # ⭐ use as directory, not file
    ensure_dir(model_dir)

    # ⭐ load the most recent checkpoint (if any)
    models = sorted(os.listdir(model_dir))
    if len(models) == 0:
        model_name = "2020-01-01 00:00:00+00:00.pt"
    else:
        model_name = sorted(models)[-1]  # pick the latest timestamped model ⭐
    last_trained = pd.to_datetime(model_name.strip(".pt"))
    model_file = os.path.join(model_dir, model_name)

    last_timestamp, resume_model, config = None, None, None

    # Resolve model class from config string
    model_name_cfg = train_cfg.get("model", "GCN")
    model_class = MODEL_CLASSES.get(model_name_cfg)
    if model_class is None:
        print(f"[{target_ref}] ⚠️ Unknown model '{model_name_cfg}'. Check config.")
        return

    # === Resume logic ===
    if os.path.exists(preds_file) and os.path.exists(model_file):
        prev_preds = pd.read_csv(preds_file, index_col=0, parse_dates=True)
        if not prev_preds.empty:
            last_timestamp = prev_preds.index.max()
            print(f"[{target_ref}] ⚡ Resuming from {last_timestamp}")
            resume_model, config = load_model(model_class, model_file, device=device)
    else:
        print(f"[{target_ref}] ℹ️ No previous run found. Training full history...")

    if last_timestamp is not None:
        file_timestamps = pd.to_datetime([f.replace(".pt", "") for f in files])
        new_files = [f for f, ts in zip(files, file_timestamps) if ts > last_timestamp]
        if not new_files:
            print(f"[{target_ref}] ✅ Already up-to-date.")
            return
        files = new_files

    files = files[:800]
    print("starts at", files[0], "ends at", files[-1])

    data_sequences, timestamps = load_sequences(graph_path, files, t_dim=train_cfg["t_dim"])
    timestamps = pd.to_datetime(timestamps)
    print(data_sequences)

    if config is None:
        config = {
            "input_dim": num_feates,
            "hidden_dim": train_cfg["num_of_nodes"],
            "num_layers": train_cfg["num_of_layers"],
            "num_heads": train_cfg["num_of_heads"],
            "dropout": train_cfg["dropout_rate"],
        }

    model = resume_model if resume_model is not None else model_class(**config).to(device)

    preds, obs, val_preds, val_obs, model, kept_ts, df_all_preds = train_model(
        model,
        data_sequences,
        features_mask=train_cfg["features_mask"],
        criterion=train_cfg["criterion"],
        lr=train_cfg["learning_rate"],
        t_dim=train_cfg["t_dim"],
        max_epochs=train_cfg["max_epochs"],
        patience=train_cfg["patience"],
        timestamps=timestamps,
        pollutant=target_ref
    )

    # ⭐ Save model with last training timestamp as filename
    if kept_ts:
        new_model_name = str(kept_ts[-1]) + ".pt"
    else:
        new_model_name = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S%z") + ".pt"

    new_model_path = os.path.join(model_dir, new_model_name)
    ensure_dir(os.path.dirname(new_model_path))
    save_model(model, new_model_path, config=config)
    print(f"[{target_ref}] 💾 Model saved as {new_model_name}")

    try:
        test_results = evaluate_metrics_block(
            preds, obs,
            num_feates, train_cfg["num_of_nodes"],
            train_cfg["num_of_layers"], train_cfg["num_of_heads"],
            train_cfg["dropout_rate"], train_cfg["learning_rate"]
        )
    
        val_results = evaluate_metrics_block(
            val_preds, val_obs,
            num_feates, train_cfg["num_of_nodes"],
            train_cfg["num_of_layers"], train_cfg["num_of_heads"],
            train_cfg["dropout_rate"], train_cfg["learning_rate"], mode="val"
        )
    except Exception as e:
        print(f"[{target_ref}] ❌ Error: {e}")

    # === Save predictions (CSV OR Mongo) ===
    print(preds)
    p = pd.concat(preds, ignore_index=True); p.index = pd.to_datetime(kept_ts)
    o = pd.concat(obs, ignore_index=True);   o.index = pd.to_datetime(kept_ts)
    df_all_preds.index = pd.to_datetime(kept_ts)
    ensure_dir(os.path.dirname(preds_file))

    def safe_append(df_new, file_path):
        if os.path.exists(file_path):
            old = pd.read_csv(file_path, index_col=0, parse_dates=True)
            all_cols = sorted(set(old.columns).union(set(df_new.columns)))
            df_new = df_new.reindex(columns=all_cols)
            old = old.reindex(columns=all_cols)
            combined = pd.concat([old, df_new])
            combined.to_csv(file_path)
        else:
            df_new = df_new.reindex(columns=sorted(df_new.columns))
            df_new.to_csv(file_path, index=True)

    safe_append(p, preds_file)
    safe_append(o, preds_path + f"{target_ref}/obser_collocated.csv")
    safe_append(df_all_preds, preds_path + f"{target_ref}/all_preds.csv")

    print(f"[{target_ref}] ✅ Training complete. Saved {len(kept_ts)} new rows (CSV).")

def update():
    print(f"\n🕙 Update triggered at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    for target_ref in CONFIG["pollutants"]:
        try:
            update_model(target_ref)
        except Exception as e:
            print(f"[{target_ref}] ❌ Error: {e}")
            traceback.print_exc()  # full stack trace with line numbers

def main():
    if CONFIG["run_once_on_start"]:
        update()

    schedule.every().hour.at(":58").do(update)
    print(f"✅ Finished ETL run at {pd.Timestamp.now()}\n")

    print("⏲️ Scheduler started. Running every hour at :09. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting on user interrupt.")
        sys.exit(0)