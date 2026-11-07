"""Build a data cube of forecast metrics, run metadata and time-series features."""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "output" / "results"
FEATURES_DIR = ROOT / "output" / "features"
DATA_DIR = ROOT / "data" / "TIME"
DATASETS_YAML = ROOT / "TIME" / "src" / "timebench" / "config" / "datasets.yaml"
CACHE_DIR = ROOT / "output" / "data_cube"

TERMS = ["short", "medium", "long"]
METRICS = ["MSE", "MAE", "sMAPE", "MASE"]
WINDOW_METRICS = [*METRICS, "QR80_rel"]
FEATURE_ID_COLUMNS = ["dataset_id", "series_name", "variate_name", "unique_id"]
FEATURE_PREFIX = "feat_"
SERIES_KEYS = ["model", "dataset", "freq", "term", "series_idx", "variate_idx"]


def iter_runs(results_dir: Path = RESULTS_DIR):
    """Yield (model, dataset, freq, term, run_dir) for every finished run."""
    for config_path in sorted(results_dir.glob("*/*/*/*/config.json")):
        run_dir = config_path.parent
        model, dataset, freq, term = run_dir.relative_to(results_dir).parts
        if (run_dir / "metrics.npz").exists():
            yield model, dataset, freq, term, run_dir


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Metrics of one run as a long frame (one row per series x window x variate)."""
    config = json.loads((run_dir / "config.json").read_text())
    with np.load(run_dir / "metrics.npz") as npz:
        metrics = {name: npz[name] for name in npz.files if name in METRICS}

    shape = next(iter(metrics.values())).shape
    s_idx, w_idx, v_idx = (a.ravel() for a in np.indices(shape))
    df = pd.DataFrame({"series_idx": s_idx, "window_idx": w_idx, "variate_idx": v_idx})
    for name in METRICS:
        df[name] = metrics[name].ravel() if name in metrics else np.nan
    predictions_path = run_dir / "predictions.npz"
    if predictions_path.exists():
        dataset_id = config["dataset_config"].rsplit("/", 1)[0]
        test_length = load_dataset_config()[dataset_id]["test_length"]
        with np.load(predictions_path) as npz:
            history_scale = _history_iqr(
                dataset_id, shape, test_length, config["prediction_length"]
            )
            df["QR80_rel"] = _relative_quantile_range(
                npz["predictions_quantiles"],
                npz["quantile_levels"],
                history_scale,
                config.get("prediction_scale_factor", 1.0),
            ).ravel()
    else:
        df["QR80_rel"] = np.nan
    return df, config


def _history_iqr(
    dataset_id: str, shape: tuple[int, ...], test_length: int, prediction_length: int
) -> np.ndarray:
    """IQR of each series' available history before each forecast window."""
    num_series, num_windows, num_variates = shape
    dataset = datasets.load_from_disk(str(DATA_DIR / dataset_id))
    if len(dataset) < num_series:
        raise ValueError(f"Not enough series in {dataset_id}")
    scales = np.full(shape, np.nan)
    for series_idx in range(num_series):
        target = np.asarray(dataset[series_idx]["target"], dtype=float)
        if target.ndim == 1:
            target = target[np.newaxis, :]
        if target.ndim != 2 or target.shape[0] != num_variates:
            raise ValueError(f"Unexpected target shape {target.shape} in {dataset_id}")
        for window_idx in range(num_windows):
            end = target.shape[-1] - test_length + window_idx * prediction_length
            if end <= 0 or end > target.shape[-1]:
                continue
            for variate_idx in range(num_variates):
                history = target[variate_idx, :end]
                history = history[np.isfinite(history)]
                if history.size:
                    lower, upper = np.percentile(history, [25, 75])
                    scales[series_idx, window_idx, variate_idx] = upper - lower
    return scales


def _relative_quantile_range(
    predictions: np.ndarray,
    levels: np.ndarray,
    history_scale: np.ndarray,
    prediction_scale_factor: float = 1.0,
) -> np.ndarray:
    """80% interval width divided by historical IQR per forecast window."""
    levels = np.asarray(levels)
    indices = [np.flatnonzero(np.isclose(levels, q)) for q in (0.1, 0.9)]
    if any(len(index) != 1 for index in indices):
        return np.full(predictions.shape[:2] + predictions.shape[3:4], np.nan)

    lower_idx, upper_idx = (int(index[0]) for index in indices)
    interval_width = (
        np.nanmean(
            predictions[:, :, upper_idx].astype(float)
            - predictions[:, :, lower_idx].astype(float),
            axis=-1,
        )
        * prediction_scale_factor
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            interval_width,
            history_scale,
            out=np.full_like(interval_width, np.nan, dtype=float),
            where=np.isfinite(history_scale) & (history_scale > 0),
        )


@cache
def load_dataset_config() -> dict:
    return yaml.safe_load(DATASETS_YAML.read_text()).get("datasets", {})


@cache
def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(FEATURES_DIR / "summary.csv")
    summary = summary[summary["dataset"] != "TOTAL"]
    return summary.rename(
        columns={"n_series": "n_series_total", "n_variates": "n_variates_total"}
    ).drop(columns="test_length")


@cache
def load_series_index(dataset: str, freq: str) -> tuple[pd.DataFrame, list[str] | None]:
    """Series order, start and length from the HF dataset (defines the metrics series axis)."""
    hf = datasets.load_from_disk(str(DATA_DIR / dataset / freq))
    table = hf.data.table
    target = table.column("target").combine_chunks()
    is_multivariate = pa.types.is_list(
        target.type.value_type
    ) or pa.types.is_fixed_size_list(target.type.value_type)
    lengths = pc.list_value_length(
        pc.list_element(target, 0) if is_multivariate else target
    )

    series = pd.DataFrame(
        {
            "series_idx": np.arange(len(table)),
            "item_id": table.column("item_id").to_pylist(),
            "series_start": pd.to_datetime(table.column("start").to_pandas()),
            "series_length": lengths.to_numpy(zero_copy_only=False),
        }
    )
    variate_names = None
    if is_multivariate and "variate_names" in table.column_names:
        variate_names = list(table.column("variate_names")[0].as_py())
    return series, variate_names


@cache
def load_features(dataset: str, freq: str) -> pd.DataFrame | None:
    """Features per (series, variate); prefers the test split, falls back to the full series."""
    for split in ("test", "full"):
        path = FEATURES_DIR / dataset / freq / f"{split}.csv"
        if path.exists():
            feats = pd.read_csv(path)
            feats["series_name"] = feats["series_name"].astype(str)
            feats["variate_name"] = feats["variate_name"].astype(str)
            feats = feats.rename(
                columns={
                    c: FEATURE_PREFIX + c
                    for c in feats.columns
                    if c not in FEATURE_ID_COLUMNS
                }
            )
            feats["feature_split"] = split
            return feats.drop(columns="dataset_id")
    return None


def _name_lookup(
    dataset: str, freq: str, num_series: int, num_variates: int
) -> pd.DataFrame:
    """Map (series_idx, variate_idx) to series/variate names and the matching feature row."""
    series, variate_names = load_series_index(dataset, freq)
    feats = load_features(dataset, freq)
    series = series.iloc[:num_series]

    rows = []
    for s in series.itertuples(index=False):
        for v in range(num_variates):
            if variate_names is not None:
                rows.append((s.series_idx, v, s.item_id, variate_names[v]))
            else:
                rows.append((s.series_idx, v, s.item_id, None))
    lookup = pd.DataFrame(
        rows, columns=["series_idx", "variate_idx", "series_name", "variate_name"]
    )
    lookup = lookup.merge(series, on="series_idx", how="left").drop(columns="item_id")
    lookup["unique_id"] = None

    if feats is None:
        return lookup

    if variate_names is not None:
        by_pair = feats.set_index(["series_name", "variate_name"])["unique_id"]
        keys = list(zip(lookup["series_name"], lookup["variate_name"]))
        lookup["unique_id"] = [by_pair.get(k) for k in keys]
        # Single-series datasets: tolerate a renamed item_id by matching on variate only.
        if (
            lookup["unique_id"].isna().any()
            and feats["series_name"].nunique() == 1
            and len(series) == 1
        ):
            by_var = feats.set_index("variate_name")["unique_id"]
            missing = lookup["unique_id"].isna()
            lookup.loc[missing, "unique_id"] = lookup.loc[missing, "variate_name"].map(
                by_var
            )
    else:
        # Univariate HF datasets embed the variate in item_id with varying conventions.
        for col in ("unique_id", "series_name", "variate_name"):
            if not feats[col].is_unique:
                continue
            by_col = pd.Series(
                feats["unique_id"].to_numpy(), index=feats[col].to_numpy()
            )
            missing = lookup["unique_id"].isna()
            lookup.loc[missing, "unique_id"] = lookup.loc[missing, "series_name"].map(
                by_col
            )
        matched = feats.set_index("unique_id")["variate_name"]
        lookup["variate_name"] = (
            lookup["unique_id"].map(matched).fillna(lookup["series_name"])
        )

    feat_cols = [c for c in feats.columns if c not in ("series_name", "variate_name")]
    return lookup.merge(feats[feat_cols], on="unique_id", how="left")


def _modern_alias(m: re.Match) -> str:
    aliases = {
        "H": "h",
        "T": "min",
        "S": "s",
        "M": "ME",
        "Q": "QE",
        "Y": "YE",
        "A": "YE",
    }
    return m.group(1) + aliases[m.group(2)]


def _window_times(
    df: pd.DataFrame, freq: str, test_length: int, prediction_length: int
) -> pd.DataFrame:
    """Add window_start / window_end timestamps of each forecast window."""
    # TIME uses legacy aliases (H, T, S) that pandas deprecates in favour of lowercase.
    offset = pd.tseries.frequencies.to_offset(
        re.sub(r"^(\d*)(H|T|S|M|Q|Y|A)$", _modern_alias, freq)
    )
    starts, ends = [], []
    for (start, length), grp in df.groupby(
        ["series_start", "series_length"], sort=False
    ):
        w = grp["window_idx"].to_numpy()
        first = w * prediction_length
        last = np.minimum(first + prediction_length - 1, test_length - 1)
        stamps = pd.date_range(
            start + (length - test_length) * offset, periods=test_length, freq=offset
        )
        starts.append(pd.Series(stamps[first], index=grp.index))
        ends.append(pd.Series(stamps[last], index=grp.index))
    df["window_start"] = pd.concat(starts)
    df["window_end"] = pd.concat(ends)
    return df


def build_series_cube(
    results_dir: Path = RESULTS_DIR, verbose: bool = True
) -> pd.DataFrame:
    """One row per model x task x series x variate, aggregated over windows."""
    ds_config = load_dataset_config()
    frames = []
    for model, dataset, freq, term, run_dir in iter_runs(results_dir):
        df, config = load_run(run_dir)
        dataset_id = f"{dataset}/{freq}"
        test_length = ds_config.get(dataset_id, {}).get("test_length")
        val_length = ds_config.get(dataset_id, {}).get("val_length")

        lookup = _name_lookup(
            dataset, freq, config["num_series"], config["num_variates"]
        )
        df = df.merge(lookup, on=["series_idx", "variate_idx"], how="left")
        if test_length is not None:
            df = _window_times(df, freq, test_length, config["prediction_length"])

        df.insert(0, "model", model)
        df.insert(1, "model_id", config.get("model_id"))
        df.insert(2, "dataset", dataset)
        df.insert(3, "freq", freq)
        df.insert(4, "term", term)
        df["dataset_id"] = dataset_id
        df["task_id"] = f"{dataset_id}/{term}"
        df["prediction_length"] = config["prediction_length"]
        df["num_windows"] = config["num_windows"]
        df["num_series"] = config["num_series"]
        df["num_variates"] = config["num_variates"]
        df["seasonality"] = config.get("seasonality")
        df["context_length"] = config.get("context_length")
        df["test_length"] = test_length
        df["val_length"] = val_length
        df["is_multivariate"] = config["num_variates"] > 1

        grouped = df.groupby(SERIES_KEYS, observed=True, sort=False)
        agg = grouped[WINDOW_METRICS].agg(["mean", "median", "std"])
        agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
        agg["n_valid_windows"] = grouped["MAE"].count()

        window_cols = {"window_idx", "window_start", "window_end", *WINDOW_METRICS}
        static_cols = [
            col
            for col in df.columns
            if col not in window_cols and col not in SERIES_KEYS
        ]
        static = grouped[static_cols].first()
        static["first_window_start"] = (
            grouped["window_start"].min() if "window_start" in df else pd.NaT
        )
        static["last_window_end"] = (
            grouped["window_end"].max() if "window_end" in df else pd.NaT
        )
        frames.append(static.join(agg).reset_index())
        if verbose:
            matched = df["unique_id"].notna().mean()
            print(
                f"{model} {dataset_id}/{term}: {len(df):>8} window rows, feature match {matched:.0%}"
            )

    cube = pd.concat(frames, ignore_index=True)
    cube = cube.merge(load_summary(), on=["dataset", "freq"], how="left")
    cube["has_features"] = cube["unique_id"].notna()

    for col in (
        "model",
        "model_id",
        "dataset",
        "freq",
        "dataset_id",
        "task_id",
        "feature_split",
    ):
        cube[col] = cube[col].astype("category")
    cube["term"] = pd.Categorical(cube["term"], categories=TERMS, ordered=True)
    return cube


def load_or_build(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Return the series cube, using the parquet cache unless ``force``."""
    series_path = CACHE_DIR / "series_cube_iqr.parquet"
    if not force and series_path.exists():
        return pd.read_parquet(series_path)

    series_cube = build_series_cube(verbose=verbose)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series_cube.to_parquet(series_path, index=False)
    return series_cube


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


if __name__ == "__main__":
    cube = load_or_build(force=True)
    print(cube.shape)
