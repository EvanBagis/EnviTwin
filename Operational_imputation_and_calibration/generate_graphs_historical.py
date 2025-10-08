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
from math import radians, sin, cos, sqrt, atan2

# ================================
# Data Handling & Analysis
# ================================
import pandas as pd
import numpy as np

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

# ================================
# Utility Functions
# ================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)
    
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

def generate_save_graph4(target_ref, target_time, dfs, ref_df, collocated_nodes, corr_threshold, search_list, graph_path):
    global _graph_debug_printed

    # === Step 1: Extract multi-variate features at target_time ===
    end = target_time
    first_key = list(dfs.keys())[0]
    stations = dfs[first_key].columns  # canonical station axis
    features = {s: [] for s in stations}

    for var_key, df in dfs.items():
        if end not in df.index:
            return None
        row = df.loc[end]
        if not row.index.equals(stations):
            print(f"⚠️ variable '{var_key}' columns mismatch with canonical station list")
            return None
        if row.isna().any():
            return None
        for s in stations:
            features[s].append(row[s])

    # shape: [stations, num_vars]
    x = pd.DataFrame(features).T

    # === Step 2: Target y from reference ===
    y_ref = ref_df.loc[end]
    if y_ref.isnull().all():
        return None

    y_ref = y_ref.copy()
    y_ref[y_ref == 0] = np.nan
    y_ref = y_ref.dropna()
    if y_ref.empty:
        return None

    pt_save_name = y_ref.name

    # normalize to GR codes
    gr_codes = y_ref.index.str.extract(r'(GR\d{4}A)')[0]
    y_ref.index = gr_codes

    # === Step 3: Edges ===
    edges, edge_index = make_edges_optimized(x, corr_threshold)

    # === Step 4: Align y using collocated_nodes mapping ===
    mapped_nodes = y_ref.index.to_series().map(collocated_nodes)
    valid_mask = mapped_nodes.notna()

    # ✅ keep only nodes that actually exist in x.index
    mapped_nodes = mapped_nodes[valid_mask]
    common = mapped_nodes[mapped_nodes.isin(x.index)]

    aligned_y = pd.Series(index=x.index, dtype="float32")  # default NaNs
    aligned_y.loc[common] = y_ref.loc[common.index].values

    # === Mask and tensors ===
    mask_y = ~aligned_y.isna()
    x_scaled = StandardScaler().fit_transform(x.values)

    x_tensor = torch.tensor(x_scaled.astype(np.float32))
    y_tensor = torch.tensor(aligned_y.values.astype(np.float32))  # will contain NaNs
    train_mask_y = torch.tensor(mask_y.values)

    data_obj = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, edge_attr=edges)
    data_obj.train_mask_y = train_mask_y
    data_obj.station_names = list(x.index)  # 🚀 preserve real station IDs

    if not _graph_debug_printed:
        print("=== Graph generation info (first graph only) ===")
        print(f"Time: {pt_save_name}")
        print(f"x.shape: {x_tensor.shape}  (stations × features)")
        print(f"y.shape: {y_tensor.shape}  (stations, aligned to x.index)")
        print(f"edge_index.shape: {edge_index.shape}")
        print(f"edge_attr.shape: {edges.shape}")
        print(f"Train mask sum: {train_mask_y.sum().item()}  /  {len(train_mask_y)}")
        print(f"Stations in x (first 5): {list(x.index)[:5]}")
        print("==============================================")
        print("Stations in dfs:", list(stations)[:5])
        print("Stations in x.index:", list(x.index)[:5])
        print("Stations in aligned_y:", list(aligned_y.index)[:5])
        print("Mask (first 5):", mask_y.values[:5])
        print("y values (first 5):", aligned_y.values[:5])
        _graph_debug_printed = True

    torch.save(data_obj, os.path.join(graph_path, f"{pt_save_name}.pt"))

    del data_obj, x, y_ref, edges, edge_index
    torch.cuda.empty_cache()
    gc.collect()

    return "ok"



# ================================
# Reproducibility
# ================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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


ref_names = {'GR0018A':'Agia Sofia', 
             'GR0019A':'Kalamaria',
             'GR0020A':'Kordelio',
             'GR0044A':'Penepistimio',
             'GR0045A':'Neochorouda',
             'GR0046A':'Sindos',
             'GR0047A':'Panorama',
             'GR0461A':'AUTH',
             'GR0462A':'PKM',
             'GR0463A':'Stavroupoli'
            }
node_names = {
#'nkastom01' :"AERODROMIO",
#'nkastom02' :"PANEPISTIMIO 1",
#'nkastom03' :"PANEPISTIMIO 2",
#'nkastom04':"ARISTOTELOUS PARALIA",
#'nkastom05' :"KOMVOS GIOFYRO",
#'nkastom06' :"ALIKARNASSOS",
'nkastom07':"PEIRAMATIKO SXOLIO",
'nkastom08' :"NOS AGIOS PAVLOS",
'nkastom09' :"PERIFERIAKOS",
'nkastom10' :"EFKARPIA",
'nkastom11' :"DIMARXEIO KALAMARIAS",
'nkastom12' :"EMPORIKO LIMANI",
'nkastom13' :"METEORA",
'nkastom14' :"FYSIKO",
'nkastom15' :"PLATEIA MEG. ALEXANDROU",
'nkastom16' :"AGIA SOFIA NISIDA",
'nkastom17' :"EPTAPYRGIO",
'nkastom18' :"TARATSA MENG AUTH",
#'nkastom19' :"SKG",
'nkastom20' :"KORDELIO",
#'nkastom21' :"MESAMPELIES",
'nkastom22' :"TEI SINDOU",
'nkastom23' :"POL KENTRO TOUMPAS",
'nkastom24' :"25HS MARTIOU",
'nkastom25' :"KEDEK",
#"nkastom26" :"PANEPISTIMIOUPOLI VOUTWN",
#"nkastom27" :"GIOFIROS/10o GYMNASIO",
#"nkastom28" :"PLATEIA ELEUTHERIAS",
#"nkastom29" :"LEOF. DIMOKRATIAS",
#"nkastom30" :"GIOFIROS",
#"nkastom31" :"PYROSVESTIKI IRAKLEIOU",
'nkastom32' :"NEAPOLI",
'nkastom33' :"OMPRELES PARALIA",
'nkastom34' :"AMPELOKIPOI",
'nkastom35' :"ERMOU KAPANI",
'nkastom36' :"TSIMISKI AGIA SOFIAS",
'nkastom37' :"BOTSARI",
'nkastom38' :"NAVARINOU",
'nkastom39' :"PIROSVESTIKI SINDOU",
'nkastom40' :"PLATEIA EVOSMOU",
'nkastom41' :"VARDARIS",
'nkastom42' :"IPPOKRATEIO NOS",
'nkastom43' :"POLOCHNI",
'nkastom44' :"PYLAIA",
'nkastom45' :"AGIOU DIMITRIOU"}

collocated_nodes = {
             'GR0018A':'nkastom16', 
             'GR0019A':'nkastom11',
             'GR0020A':'nkastom20',
             #'GR0044A':'nkastom14', this is integrated into GR0461A where it exists
             #'GR0045A':'Neochorouda', this is out the ROI, no collocation
             'GR0046A':'nkastom22',
             #'GR0047A':'Panorama',m no collocation
             'GR0461A':'nkastom14',
             #'GR0462A':'PKM',
             #'GR0463A':'Stavroupoli'    
}

CONFIG = {
    "load_path": "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/",
    "save_path": "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/aggregates/",
    "save_path_imputed": "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/imputed_aggregates/",
    "plot_dir": "./plots",
    "variables": ['pm_10', 'pm_1_0', 'pm_2_5', 'humidity', 'pressure', 'temperature'],
    "target": "pm_10",
    "corr_threshold": 0,
    "split": 1000,
    "min_value": 3.0,
    "window_size_multiplier": 2,
    "historical_plot_last_days": 30,
    "latest_plot_last_days": 7,
    "exclude": ['nkastom07', 'nkastom08', 'nkastom19', 'nkastom23', 'nkastom28', 'nkastom29'],
    "run_once_on_start": True,
    "reference_pollutants": ["O3", "PM2.5", "PM10", "NO2", "CO"],
    "start_timestamp": "2020-01-01 00:00:00+00:00",
}


def latest_processed_timestamp(graph_path: str) -> pd.Timestamp | None:
    """Find last processed graph timestamp from filenames."""
    if not os.path.isdir(graph_path):
        return None
    files = sorted([f for f in os.listdir(graph_path) if f.endswith(".pt")])
    if not files:
        return None
    last_file = files[-1]
    try:
        return pd.to_datetime(last_file[:-3])  # strip ".pt"
    except Exception:
        return None


def load_reference_df(target_ref: str, start_ts: str) -> pd.DataFrame:
    """Load & synchronize EEA reference data for a given pollutant."""
    ref_path = f"/home/envitwin/Desktop/venvs/databases/data/eea_data/data/{target_ref}/thessaloniki/raw/"
    ref_nodes = sorted(os.listdir(ref_path))
    ref_stations = [x for x in ref_nodes if x.endswith('.csv')]
    ref_dfs = {path[:-4]: load(os.path.join(ref_path, path)) for path in ref_stations}
    ref_df = synchronize_many_dfs(ref_dfs)
    return ref_df.loc[start_ts:]


def load_operational_imputed(save_path_imputed: str) -> dict:
    """Load all imputed operational CSVs."""
    paths = sorted(f for f in os.listdir(save_path_imputed) if f.endswith("imputed_operational.csv"))
    print(paths)
    return {path[:-4]: load(os.path.join(save_path_imputed, path)) for path in paths}

def get_search_list(save_path_imputed: str, target: str) -> list[str]:
    """
    Load the imputed operational CSV for a given target and return its column names.

    Args:
        save_path_imputed (str): Path to the directory containing imputed operational CSVs.
        target (str): Target variable name (e.g., "pm_10").

    Returns:
        list[str]: List of column names in the CSV.
    """
    oper_csv = os.path.join(save_path_imputed, f"{target}_imputed_operational.csv")
    pol_df = load(oper_csv)  # assumes your custom load() returns a DataFrame
    return list(pol_df.columns)

def process_target_ref(target_ref: str, dfs_oper: dict, corr_threshold: float, start_ts: str, search_list):
    """Catch-up graph generation for one pollutant."""
    graph_path = f"/home/envitwin/Desktop/venvs/EnviTwin/Operational_imputation_and_calibration/graph_data/{target_ref}/"
    ensure_dir(graph_path)

    ref_df = load_reference_df(target_ref, start_ts)
    last_done = latest_processed_timestamp(graph_path)
    print(f"[{target_ref}]Last graph found ⚡ at {last_done}")
    

    timestamps = ref_df.index
    if last_done is not None:
        timestamps = timestamps[timestamps > last_done]
        if len(timestamps) == 0:
            print(f"[{target_ref}] ✅ Up-to-date (last: {last_done}).")
            return

    print(f"[{target_ref}] ⚡ Processing {len(timestamps)} new timestamps"
          f"{'' if last_done is None else f' after {last_done}'}...")

    #=============timestamps[:4000]=================
    for t in tqdm(timestamps, desc=f"{target_ref}"):
        r = generate_save_graph4(target_ref, t, dfs_oper, ref_df, collocated_nodes, corr_threshold, search_list, graph_path)
        if r is None:
            continue

import traceback
def update():
    """One scheduled update run over all pollutants."""
    print(f"\n🕙 Update triggered at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    dfs_oper = load_operational_imputed(CONFIG["save_path_imputed"])
    search_list = get_search_list(CONFIG["save_path_imputed"], CONFIG["target"])
    for target_ref in CONFIG["reference_pollutants"]:
        try:
            process_target_ref(target_ref, dfs_oper, CONFIG["corr_threshold"], CONFIG["start_timestamp"], search_list)
        except Exception as e:
            print(f"[{target_ref}] ❌ Error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    # Optional immediate run once at startup
    if CONFIG.get("run_once_on_start", True):
        update()

    # Schedule to run every hour at hh:10
    schedule.every().hour.at(":13").do(update)

    print("⏲️ Scheduler started. Running every hour at :08 (UTC). Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)