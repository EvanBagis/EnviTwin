# kastom_imputation.py

import os
import math
import pickle
import time
from typing import Optional, Tuple, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

import schedule  # <-- scheduler for hourly run at minute 1

# ================================
# STATIC CONFIG — CONTROL EVERYTHING HERE
# ================================
CONFIG = {
    # paths and target
    "load_path": "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/",
    "save_path": "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/aggregates/",
    "save_path_imputed": "/home/envitwin/Desktop/venvs/databases/data/kastom_data/hour/imputed_aggregates/",
    "plot_dir": "./plots",
    "variables": ['pm_10', 'pm_1_0', 'pm_2_5', 'humidity', 'pressure', 'temperature'],
    "target": "pm_10",          # one of variables above

    # core params
    "split": 1000,
    "min_value": 0.1,
    "window_size_multiplier": 2,  # window_size = multiplier * split

    # plotting windows
    "historical_plot_last_days": 30,
    "latest_plot_last_days": 7,

    # station exclusions
    "exclude": ['nkastom07', 'nkastom08', 'nkastom19', 'nkastom23', 'nkastom25', 'nkastom28', 'nkastom29'],

    # scheduler behavior
    "run_once_on_start": True,   # run an immediate latest catch-up at startup
}

# derived
paths = [f"{v}.csv" for v in CONFIG["variables"]]
WINDOW_SIZE = CONFIG["window_size_multiplier"] * CONFIG["split"]

# ================================
# UTILS
# ================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_df(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, index_col=0)
    d.index = pd.to_datetime(d.index)
    d.index.name = ''
    return d

def save_plot(fig: plt.Figure, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)

def save_indices_pickle(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_indices_pickle(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def merge_indices_dict(base: dict, add: dict, allowed_cols: list | None = None) -> dict:
    result = {}
    cols = set(base.keys()) | set(add.keys())
    if allowed_cols is not None:
        cols = [c for c in cols if c in allowed_cols]
    for c in cols:
        a = base.get(c, [])
        b = add.get(c, [])
        combined = pd.Index(pd.to_datetime(a + b)).unique()
        result[c] = list(combined.sort_values())
    return result

def impute_with_index_tracking(df: pd.DataFrame,
                                n: int = 30,
                                min_non_missing_frac: float = 0.10,
                            ) -> tuple[pd.DataFrame, dict]:
    present_frac = df.notna().mean()
    cols_to_impute = present_frac[present_frac >= min_non_missing_frac].index.tolist()
    if not cols_to_impute:
        return pd.DataFrame(index=df.index), {}
    df_to_impute = df[cols_to_impute]
    na_mask = df_to_impute.isna()
    imputer = KNNImputer(n_neighbors=n)
    # imputer = IterativeImputer(estimator=XGBRegressor(n_estimators=n, verbosity=0),
    #                            max_iter=10, random_state=0)
    arr = imputer.fit_transform(df_to_impute)
    df_imputed = pd.DataFrame(arr, columns=df_to_impute.columns, index=df_to_impute.index)
    imputed_indices = {col: df.index[na_mask[col]].tolist() for col in cols_to_impute}
    return df_imputed, imputed_indices

def plot_imputed_columns(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame,
    imputed_indices: dict,
    plot_dir: str,
    last_days: Optional[int],
    title_suffix: str,
    variable: str,  # keep this arg in your calls
) -> None:
    df_imputed = df_imputed[~df_imputed.index.duplicated(keep='last')].sort_index()
    df_original = df_original[~df_original.index.duplicated(keep='last')].sort_index()

    # last_days=None -> no truncation
    if not df_imputed.empty and last_days is not None:
        last_ts = df_imputed.index.max()
        cutoff = last_ts - pd.Timedelta(days=last_days)
        df_imputed = df_imputed.loc[df_imputed.index >= cutoff]
        df_original = df_original.loc[df_original.index >= cutoff]

    if df_imputed.empty:
        print("plot_imputed_columns: nothing to plot (df_imputed empty after filtering).")
        return

    # Visible window start/end for filename
    start_ts = df_imputed.index.min()
    end_ts = df_imputed.index.max()

    n_cols = len(df_imputed.columns)
    n_plot_cols = math.ceil(math.sqrt(max(n_cols, 1)))
    n_plot_rows = math.ceil(max(n_cols, 1) / n_plot_cols)
    fig, axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=(5 * n_plot_cols, 4 * n_plot_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(df_imputed.columns):
        ax = axes[i]
        ax.set_title(f"{col} ({title_suffix})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        x_idx = df_imputed.index
        y_series = df_imputed[col].reindex(x_idx)
        ax.plot(x_idx, y_series, linestyle='--', linewidth=1)

        imp_ts = set(imputed_indices.get(col, []))
        imp_mask = y_series.index.isin(list(imp_ts))
        orig_mask = ~imp_mask
        ax.scatter(y_series.index[orig_mask], y_series[orig_mask], marker='o', label='Original')
        if np.any(imp_mask):
            ax.scatter(y_series.index[imp_mask], y_series[imp_mask], marker='x', label='Imputed')
        ax.legend(); ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Friendlier, filename-safe timestamps
    def fmt_range(ts: pd.Timestamp) -> str:
        # e.g., 2025-Sep-05_01h00
        return ts.strftime("%Y-%b-%d_%Hh%M")

    def fmt_now(ts: datetime) -> str:
        # e.g., 2025-Sep-05_07h01m22s
        return ts.strftime("%Y-%b-%d_%Hh%Mm%Ss")

    now_tag = fmt_now(datetime.now())
    fname = f"{variable}_imputed_from-{fmt_range(start_ts)}_to-{fmt_range(end_ts)}_generated-{now_tag}.png"
    out = os.path.join(plot_dir, fname)

    save_plot(fig, out)
    print(f"Saved plot -> {out}")


# ================================
# CORE FUNCTIONS
# ================================
def run_historical_imputation(df: pd.DataFrame, variable: str,
                              save_path_imputed: str, plot_dir: str,
                              split: int, last_days_plot: int) -> tuple[pd.DataFrame, Dict]:
    print("Running historical imputation...")
    df_hist = df.loc[:df.index[-split]]
    df_hist = df_hist[~df_hist.index.duplicated(keep='last')].sort_index()
    df_hist[df_hist==0] = np.nan
    if variable == 'humidity':
        df_hist[df_hist==100.0] = np.nan

    # n controls either K in KNN or num_trees if MICE-xgboost
    hist_imp_df, hist_imp_idx = impute_with_index_tracking(df_hist, n=30)

    # save CSV + pkl
    hist_csv = os.path.join(save_path_imputed, f"{variable}_imputed_historical.csv")
    hist_pkl = os.path.join(save_path_imputed, f"{variable}_imputed_historical_indices.pkl")
    hist_imp_df.to_csv(hist_csv)
    save_indices_pickle(hist_imp_idx, hist_pkl)
    print(f"Saved historical imputation -> {hist_csv}")
    print(f"Saved historical imputed indices -> {hist_pkl}")

    # plot (last N days)
    plot_imputed_columns(df_hist, hist_imp_df, hist_imp_idx, plot_dir,
                         last_days=last_days_plot, title_suffix="Historical", variable=variable)

    return hist_imp_df, hist_imp_idx

def run_operational_imputation(
    target: str,
    df_raw: pd.DataFrame,
    variable: str,
    load_path: str,
    save_path: str,
    save_path_imputed: str,
    plot_dir: str,
    mode: str,                       # "sample" or "latest"
    split: int,
    min_value: float,
    sample_slice: Optional[Tuple[Optional[str], Optional[str]]] = None,
    sample_last_n: int = 200,
    sample_plot_days: Optional[int] = None,    # None -> full sample
    latest_plot_days: int = 7,                 # used only if >1 rows appended in latest
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, Dict]:
    """
    Operational imputation with two modes:

    - mode == 'sample': run on a chosen interval (or last N rows) for visualization only.
                        Does NOT write CSV/pickle when write_outputs=False. Saves plots.

    - mode == 'latest': catch up fully by appending every NEXT unseen timestamp
                        after the final saved timestamp in operational (or historical if missing).
                        Skip a timestamp ONLY if the entire row has no measurements (all-NaN after min_value).
                        If exactly one row is appended -> NO PLOTS.
                        If more than one row appended -> SAVE PLOTS (last `latest_plot_days` days).
    """
    print(f"Running operational imputation (mode='{mode}', write_outputs={write_outputs})...")

    oper_csv = os.path.join(save_path_imputed, f"{variable}_imputed_operational.csv")
    hist_csv = os.path.join(save_path_imputed, f"{variable}_imputed_historical.csv")
    oper_pkl = os.path.join(save_path_imputed, f"{variable}_imputed_operational_indices.pkl")
    hist_pkl = os.path.join(save_path_imputed, f"{variable}_imputed_historical_indices.pkl")
    
    # Load base CSV/indices (prefer operational)
    if os.path.exists(oper_csv):
        print(f"Loading operational base: {oper_csv}")
        df_base = load_df(oper_csv)
        base_idx = load_indices_pickle(oper_pkl)
    else:
        print(f"Loading historical base: {hist_csv}")
        df_base = load_df(hist_csv)
        base_idx = load_indices_pickle(hist_pkl)

    # Clean & lock schema
    df_base = df_base[~df_base.index.duplicated(keep='last')].sort_index()
    print('from operational csv', df_base.index[-1])
    allowed_cols = list(df_base.columns)
    df_base = df_base.reindex(columns=allowed_cols)

    # Pre-mask RAW once; keep only allowed cols
    df_raw = df_raw.reindex(columns=allowed_cols)
    df_raw_masked = df_raw.where(df_raw >= min_value)
    #print(df_raw, df_raw_masked)

    # ---------- LATEST MODE (catch up fully; plots only if appended > 1) ----------
    last_ts = df_base.index.max() if len(df_base) > 0 else pd.Timestamp.min

    #kastom_nodes = sorted(os.listdir(load_path))
    #kastom_nodes = [x for x in kastom_nodes if 'kastom' in x]
    #print(kastom_nodes)
    #kastom_dfs = {path[:-4]:load_df(save_path + path) for path in paths}
    #stations = kastom_dfs[target].columns # any variable will do
    #stations = [f for f in stations if f not in CONFIG["exclude"]]
    #print(f'the following {len(stations)} stations are available for imputation')
    #print(stations)

    # Future rows strictly after last_ts, already masked; keep rows with ANY measurement
    future_masked = df_raw_masked.loc[df_raw_masked.index > last_ts]
    future_masked = future_masked[~future_masked.index.duplicated(keep='last')].sort_index()
    future_masked = future_masked[future_masked.notna().any(axis=1)]
    #print(future_masked)

    if future_masked.empty:
        print("No new timestamps to append. Operational CSV already up-to-date.")
        if write_outputs and os.path.exists(oper_csv) and not os.path.exists(oper_pkl):
            save_indices_pickle(base_idx, oper_pkl)
            print(f"(No-op) Ensured operational indices -> {oper_pkl}")
        return df_base, base_idx

    incremental_df = df_base.where(df_base >= min_value).copy()
    incremental_df = incremental_df.reindex(columns=allowed_cols)

    new_rows_dict: Dict[pd.Timestamp, pd.Series] = {}
    run_idx: dict = {}

    for ts, row_masked in tqdm(future_masked.iterrows(), total=len(future_masked), desc="Catching up"):
        # Bounded rolling window to avoid growing cost
        if len(incremental_df) >= WINDOW_SIZE:
            incremental_df = incremental_df.iloc[-(WINDOW_SIZE-1):]

        incremental_df.loc[ts] = row_masked.reindex(allowed_cols)
        incremental_df[incremental_df==0] = np.nan
        if variable == 'humidity':
            incremental_df[incremental_df==100.0] = np.nan

        imp_df, imp_idx = impute_with_index_tracking(incremental_df, n=30)
        imp_df = imp_df.reindex(columns=allowed_cols)
        if ts not in imp_df.index:
            continue

        new_rows_dict[ts] = imp_df.loc[ts].reindex(allowed_cols)

        # record only if ts was imputed
        for col, idxs in imp_idx.items():
            if len(idxs) == 0:
                continue
            idxs_ts = pd.to_datetime(idxs)
            if ts in idxs_ts:
                run_idx.setdefault(col, []).append(ts)

    if not new_rows_dict:
        print("No future timestamps with any measurements — nothing appended.")
        if write_outputs and os.path.exists(oper_csv) and not os.path.exists(oper_pkl):
            save_indices_pickle(base_idx, oper_pkl)
            print(f"(No-op) Ensured operational indices -> {oper_pkl}")
        return df_base, base_idx

    block = pd.DataFrame.from_dict(new_rows_dict, orient='index').reindex(columns=allowed_cols).sort_index()
    appended_count = len(block)

    final_df = pd.concat([df_base, block]).sort_index()
    final_df = final_df[~final_df.index.duplicated(keep='last')]
    final_idx = merge_indices_dict(base_idx, run_idx, allowed_cols=allowed_cols)

    if write_outputs:
        final_df.to_csv(oper_csv)
        save_indices_pickle(final_idx, oper_pkl)
        print(f"Appended {appended_count} timestamp(s) -> {oper_csv}")
        print(f"Updated operational indices -> {oper_pkl}")

    # Only plot if more than one timestamp appended
    if write_outputs and appended_count > 1:
        try:
            plot_imputed_columns(
                df_original=df_raw,
                df_imputed=final_df,
                imputed_indices=final_idx,
                plot_dir=plot_dir,
                last_days=CONFIG["latest_plot_last_days"],
                title_suffix="Latest Catch-up", variable=variable
            )
            print("Saved latest catch-up plot.")
        except Exception as e:
            print(f"Plotting skipped due to error: {e}")

    return final_df, final_idx

# ================================
# HOURLY UPDATE RUNNER (LATEST MODE AT :01)
# ================================
def update():
    """Run latest-mode catch-up for ALL variables in CONFIG['variables'].
    - Bootstraps historical per variable if no imputed base exists.
    - Appends all missing timestamps in chronological order (skipping rows with all-NaN).
    - Saves plots only if more than one timestamp was appended (handled inside run_operational_imputation).
    """
    target = CONFIG["target"]
    load_path = CONFIG["load_path"]
    save_path = CONFIG["save_path"]
    save_path_imputed = CONFIG["save_path_imputed"]
    plot_dir = CONFIG["plot_dir"]
    variables = CONFIG["variables"]
    split = CONFIG["split"]
    min_value = CONFIG["min_value"]

    ensure_dir(save_path_imputed)
    ensure_dir(plot_dir)

    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] Starting hourly update for all variables...")

    for target in variables:
        try:
            agg_path = os.path.join(save_path, f"{target}.csv")
            if not os.path.exists(agg_path):
                print(f"• {target}: SKIP (missing aggregate file: {agg_path})")
                continue

            # Load raw for this variable
            raw_df = load_df(agg_path)
            stations = [c for c in raw_df.columns if c not in CONFIG["exclude"]]
            if not stations:
                print(f"• {target}: SKIP (no stations left after exclusion)")
                continue

            df = raw_df[stations].copy()
            # Treat very small values as missing (kept same as your code)
            df[df < min_value] = np.nan

            hist_csv = os.path.join(save_path_imputed, f"{target}_imputed_historical.csv")
            oper_csv = os.path.join(save_path_imputed, f"{target}_imputed_operational.csv")

            # Bootstrap historical if neither file exists for this variable
            if not (os.path.exists(hist_csv) or os.path.exists(oper_csv)):
                print(f"• {target}: Bootstrapping historical...")
                run_historical_imputation(
                    df=df,
                    variable=target,
                    save_path_imputed=save_path_imputed,
                    plot_dir=plot_dir,
                    split=split,
                    last_days_plot=CONFIG["historical_plot_last_days"],
                )

            # Run latest catch-up for this variable (writes CSV/PKL; plots only if >1 rows appended)
            print(f"• {target}: Running latest catch-up...")
            run_operational_imputation(
                target=target,
                df_raw=df,
                variable=target,
                load_path=load_path,
                save_path=save_path,
                save_path_imputed=save_path_imputed,
                plot_dir=plot_dir,
                mode="latest",
                split=split,
                min_value=min_value,
                sample_slice=None,
                sample_last_n=0,
                sample_plot_days=None,
                latest_plot_days=CONFIG["latest_plot_last_days"],
                write_outputs=True,
            )

        except Exception as e:
            print(f"• {target}: ERROR during update -> {e}")

    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] Update complete.")

# ================================
# MAIN — SCHEDULE EVERY HOUR AT :07
# ================================
if __name__ == "__main__":
    # Optional immediate run once at startup
    if CONFIG.get("run_once_on_start", True):
        update()

    # Schedule to run on the first minute of every hour
    schedule.every().hour.at(":12").do(update)

    print("Scheduler started. Running every hour at :03. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)

# Tested that it runs smoothly
# KASTOM.py runs but the data from node-RED are late (even after 15 mins new data arive)