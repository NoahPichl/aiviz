"""diagnostics.py -- turn predictions.npz files into ONE ROW PER FORECAST.

Layout of predictions_quantiles (verified on your data):
    (series, window, quantile, variate, horizon)      e.g. (989, 3, 9, 3, 288)
Every (series, window, variate) triple is one forecast; it matches one row of window_cube via
    dataset, freq, term, series_idx, window_idx, variate_idx

Only the *outputs* of the model are used here (black-box view); no raw data needed.
"""
import os
import numpy as np
import pandas as pd

FORECAST_KEYS = ["dataset", "freq", "term", "series_idx", "window_idx", "variate_idx"]
N_REL = 20                      # points of the relative-horizon grid used for width profiles
WR_COLS = [f"wr_{i:02d}" for i in range(N_REL)]
MIN_H_SHAPE = 12                # shape metrics (ACF) need at least this many horizon steps
MIN_H_FLAT = 4                  # flat-line test needs at least this many steps
FLAT_TOL = 1e-3                 # median forecast counts as "flat" if its range < 0.1% of its level
EPS = 1e-12


# ------------------------------------------------------------------ loading
def find_prediction_files(results_dir):
    """[(dataset, freq, term, path)] for results_dir/<dataset>/<freq>/<term>/predictions.npz"""
    out = []
    for root, _, files in os.walk(results_dir):
        if "predictions.npz" in files:
            parts = os.path.normpath(root).split(os.sep)
            out.append((parts[-3], parts[-2], parts[-1], os.path.join(root, "predictions.npz")))
    return sorted(out)


def load_quantiles(path):
    """Returns q (series, window, quantile, variate, horizon) as float64, and the quantile levels.
    float16 -> float64 BEFORE any arithmetic (float16 overflows easily)."""
    with np.load(path) as d:
        q = d["predictions_quantiles"].astype(np.float64)
        levels = (d["quantile_levels"].astype(np.float64) if "quantile_levels" in d.files
                  else np.linspace(0.1, 0.9, q.shape[2]))
    if q.ndim != 5 or q.shape[2] != len(levels):
        raise ValueError(f"unexpected layout {q.shape} in {path}")
    return q, levels


def load_config(results_dir, dataset, freq, term):
    return load_quantiles(os.path.join(results_dir, dataset, freq, term, "predictions.npz"))


def _forecast_array(q):
    """(S, W, Q, V, H) -> (N, Q, H) with rows ordered by (series, window, variate)."""
    S, W, Q, V, H = q.shape
    return np.moveaxis(q, 2, 3).reshape(-1, Q, H), (S, W, V)


def _idx(levels, target):
    return int(np.argmin(np.abs(levels - target)))


# per-forecast table
def forecast_table(q, levels, dataset, freq, term):
    A, (S, W, V) = _forecast_array(q)
    N, Q, H = A.shape
    i_lo, i_med, i_hi = _idx(levels, 0.1), _idx(levels, 0.5), _idx(levels, 0.9)

    ok = np.isfinite(A).all(axis=(1, 2))           # rows with NaN/inf get NaN metrics at the end
    A = np.where(ok[:, None, None], A, 0.0)

    lo, med, hi = A[:, i_lo], A[:, i_med], A[:, i_hi]
    width = hi - lo
    wmean = np.abs(width).mean(1)

    # quantile crossing (adjacent quantiles must be non-decreasing)
    d = np.diff(A, axis=1)
    cross_max = np.maximum(-d, 0).max(axis=(1, 2))
    cols = {
        "horizon": np.full(N, H),
        "cross_rate": (d < 0).mean(axis=(1, 2)),
        "cross_any": (d < 0).any(axis=(1, 2)),
        "cross_max": cross_max,
        "cross_rel": cross_max / (wmean + EPS),     # size of the worst violation vs the interval width
        "lo_gt_hi": (lo > hi).any(axis=1),          # 10% quantile above 90% quantile at some step
    }

    # uncertainty
    zero_w = wmean <= EPS
    cols["width_mean"] = wmean
    cols["zero_width"] = zero_w
    q_ = max(1, H // 4)
    first, last = width[:, :q_].mean(1), width[:, -q_:].mean(1)
    good = (first > 1e-3 * wmean) & (H >= 4)
    cols["width_growth"] = np.where(good, last / np.where(good, first, 1.0), np.nan)
    pos = np.linspace(0, H - 1, N_REL)               # width profile on a relative-horizon grid,
    i0 = np.floor(pos).astype(int)                   # divided by the forecast's own mean width
    i1 = np.minimum(i0 + 1, H - 1)
    fr = pos - i0
    prof = width[:, i0] * (1 - fr) + width[:, i1] * fr
    prof = prof / (wmean[:, None] + EPS)
    prof[zero_w] = np.nan
    for j, c in enumerate(WR_COLS):
        cols[c] = prof[:, j]

    # shape of the median trajectory
    t = np.arange(H) - (H - 1) / 2
    slope = (med * t).sum(1) / (t ** 2).sum() if H > 1 else np.zeros(N)
    amp = med.std(1)
    ptp = np.ptp(med, axis=1)
    level = np.abs(med).mean(1)
    is_flat = ptp <= FLAT_TOL * level + EPS
    cols["amp"] = amp
    cols["trend_change"] = slope * (H - 1)          # change of the fitted line over the whole horizon
    cols["is_flat"] = is_flat if H >= MIN_H_FLAT else np.full(N, np.nan)
    if H >= MIN_H_SHAPE:
        c = med - med.mean(1, keepdims=True)
        den = (c ** 2).sum(1)
        acf1 = (c[:, :-1] * c[:, 1:]).sum(1) / np.where(den > 0, den, np.nan)
        acf1[is_flat] = np.nan
    else:
        acf1 = np.full(N, np.nan)
    cols["acf1"] = acf1

    df = pd.DataFrame(cols)
    metric_cols = [c for c in df.columns if c != "horizon"]
    df[metric_cols] = df[metric_cols].astype(float)
    df.loc[~ok, metric_cols] = np.nan
    s_idx, w_idx, v_idx = np.indices((S, W, V)).reshape(3, -1)
    keys = pd.DataFrame({"dataset": dataset, "freq": freq, "term": term,
                         "series_idx": s_idx, "window_idx": w_idx, "variate_idx": v_idx})
    return pd.concat([keys, df], axis=1)


def build_forecast_table(results_dir, datasets=None, cache_path=None, verbose=True):
    """Scan all predictions.npz under results_dir. With cache_path the result is saved/reloaded (parquet)."""
    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    parts = []
    for ds, freq, term, path in find_prediction_files(results_dir):
        if datasets is not None and ds not in datasets:
            continue
        try:
            q, levels = load_quantiles(path)
            parts.append(forecast_table(q, levels, ds, freq, term))
            if verbose:
                print(f"{ds} | {freq} | {term}: {q.shape} -> {len(parts[-1])} forecasts")
        except Exception as e:                          # keep going, but never silently
            print(f"!! skipped {path}: {e}")
    out = pd.concat(parts, ignore_index=True)
    assert not out.duplicated(FORECAST_KEYS).any(), "duplicate forecast keys"
    if cache_path:
        out.to_parquet(cache_path)
    return out


# join with the cube
def merge_with_windows(df_windows, df_forecasts):
    """df_analysis = window cube (errors + features) + forecast diagnostics. One row per window."""
    if "model" in df_windows and df_windows["model"].nunique() > 1:
        print("WARNING: several models in df_windows; diagnostics belong to ONE model only.")
    left = df_windows.copy()
    for c in ("freq", "term"):
        left[c] = left[c].astype(str)
    right = df_forecasts.copy()
    for c in ("freq", "term"):
        right[c] = right[c].astype(str)
    out = left.merge(right, on=FORECAST_KEYS, how="left", validate="many_to_one")
    matched = out["horizon"].notna().mean()
    print(f"windows with forecast diagnostics: {matched:.1%} of {len(out)}")
    from data_loading import TERM_ORDER
    out["term"] = pd.Categorical(out["term"], categories=TERM_ORDER, ordered=True)
    for c in ("cross_any", "lo_gt_hi", "zero_width", "is_flat"):
        out[c] = out[c].astype(float)
    # scale-free versions: divide by the series' own std (feat_std) -- NaN where feat_std is missing/0
    sd = out["feat_std"].where(out["feat_std"] > 0)
    out["width_norm"] = out["width_mean"] / sd
    out["amp_rel"] = out["amp"] / sd
    out["trend_change_std"] = out["trend_change"] / sd
    return out


def pick_windows_by_error(df_analysis, config, quantiles=(0.1, 0.5, 0.9)):
    """Transparent (non-cherry-picked) examples: windows at given MASE quantiles of one config."""
    d = df_analysis[df_analysis["config"] == config].dropna(subset=["MASE"]).sort_values("MASE")
    idx = [int(round(p * (len(d) - 1))) for p in quantiles]
    return d.iloc[idx][FORECAST_KEYS + ["MASE"]].assign(MASE_quantile=quantiles)


# ACF profiles 
def acf_profiles(results_dir, dataset, max_lag=48):
    """Autocorrelation of the MEDIAN forecast at lags 1..max_lag (capped at horizon//2),
    per freq/term of one dataset. Flat and non-finite forecasts are excluded."""
    rows = []
    for ds, freq, term, path in find_prediction_files(results_dir):
        if ds != dataset:
            continue
        q, levels = load_quantiles(path)
        A, _ = _forecast_array(q)
        H = A.shape[2]
        if H < MIN_H_SHAPE:
            continue
        med = A[:, _idx(levels, 0.5)]
        keep = np.isfinite(med).all(1)
        med = med[keep]
        keep = np.ptp(med, axis=1) > FLAT_TOL * np.abs(med).mean(1) + EPS
        med = med[keep]
        if len(med) == 0:
            continue
        c = med - med.mean(1, keepdims=True)
        den = (c ** 2).sum(1)
        for k in range(1, min(max_lag, H // 2) + 1):
            a = (c[:, :-k] * c[:, k:]).sum(1) / den
            rows.append(dict(freq=freq, term=term, horizon=H, lag=k, median=np.median(a),
                             p25=np.percentile(a, 25), p75=np.percentile(a, 75), n=len(a)))
    return pd.DataFrame(rows)