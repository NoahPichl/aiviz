"""data_loading.py -- load the TiRex2 cubes and build the three tables used by the notebook.

Tables
------
df_windows : one row per forecast window   (error metrics + series features)
df_series  : one row per series/variate/config (median errors, features)
df_config  : one row per dataset|freq|term  (what most plots use)
"""
import numpy as np
import pandas as pd

SERIES_KEYS = ["model", "dataset", "freq", "term", "series_idx", "variate_idx"]
WINDOW_KEYS = SERIES_KEYS + ["window_idx"]
CONFIG_KEYS = ["dataset", "freq", "term"]
TERM_ORDER = ["short", "medium", "long"]

# Scale-free features only (comparable across datasets). feat_std / feat_spike / feat_mean
# depend on the units of the data, so they are NOT used for cross-dataset comparison.
FEATURES = [
    "feat_e_acf1", "feat_e_acf10", "feat_e_diff1_acf1", "feat_e_kurtosis",
    "feat_e_arch_lm", "feat_trend_strength", "feat_trend_stability",
    "feat_seasonal_strength", "feat_x_entropy", "feat_e_entropy",
]
ERRORS = ["MASE", "CRPS", "MAE", "RMSE", "ND"]


def load_cubes(series_path="output/data_cube/series_cube.parquet",
               window_path="output/data_cube/window_cube.parquet"):
    series_cube = pd.read_parquet(series_path)
    window_cube = pd.read_parquet(window_path)
    for df in (series_cube, window_cube):
        df["term"] = pd.Categorical(df["term"], categories=TERM_ORDER, ordered=True)
    assert series_cube.duplicated(SERIES_KEYS).sum() == 0, "duplicate series keys"
    return series_cube, window_cube


def build_df_windows(window_cube):
    df = window_cube.copy()
    df["config"] = df["dataset"].astype(str) + " | " + df["freq"].astype(str) + " | " + df["term"].astype(str)
    return df


def build_df_series(df_windows):
    agg = {**{c: "first" for c in FEATURES},
           **{c: "median" for c in ERRORS},
           "window_idx": "nunique"}
    out = (df_windows.groupby(SERIES_KEYS, observed=True).agg(agg).reset_index()
           .rename(columns={"window_idx": "n_windows"}))
    out["config"] = out["dataset"].astype(str) + " | " + out["freq"].astype(str) + " | " + out["term"].astype(str)
    return out


def build_df_config(df_windows, df_series):
    """One row per dataset|freq|term: error summary from WINDOWS, features from SERIES."""
    w = df_windows.dropna(subset=["MASE"])
    g = w.groupby(CONFIG_KEYS, observed=True)["MASE"]
    err = pd.DataFrame({
        "median_MASE": g.median(),
        "q25_MASE": g.quantile(0.25),
        "q75_MASE": g.quantile(0.75),
        "q95_MASE": g.quantile(0.95),
        "share_worse_than_naive": g.apply(lambda s: (s > 1).mean()),
        "median_CRPS": w.groupby(CONFIG_KEYS, observed=True)["CRPS"].median(),
        "n_windows": g.size(),
    })
    n_series = (df_series.groupby(CONFIG_KEYS, observed=True)
                .agg(n_series=("series_idx", "nunique"), n_variates=("variate_idx", "nunique")))
    feats = df_series.groupby(CONFIG_KEYS, observed=True)[FEATURES].median()
    out = err.join(n_series).join(feats).reset_index()
    out["config"] = out["dataset"].astype(str) + " | " + out["freq"].astype(str) + " | " + out["term"].astype(str)
    return out


def load_all(series_path="output/data_cube/series_cube.parquet",
             window_path="output/data_cube/window_cube.parquet"):
    """Convenience: returns df_windows, df_series, df_config."""
    _, window_cube = load_cubes(series_path, window_path)
    df_windows = build_df_windows(window_cube)
    df_series = build_df_series(df_windows)
    df_config = build_df_config(df_windows, df_series)
    return df_windows, df_series, df_config


def multi_value_datasets(df_config, column):
    """Datasets that appear with more than one value of `column` ('freq' or 'term').
    These are the only ones where a same-dataset comparison is possible."""
    n = df_config.groupby("dataset", observed=True)[column].nunique()
    return n[n > 1].index.tolist()


def best_worst_configs(df_config, k=3, min_windows=30, min_series=1):
    """k lowest / k highest median-MASE configs, ignoring configs with too few windows."""
    ok = df_config[(df_config["n_windows"] >= min_windows) & (df_config["n_series"] >= min_series)]
    return ok.nsmallest(k, "median_MASE"), ok.nlargest(k, "median_MASE")