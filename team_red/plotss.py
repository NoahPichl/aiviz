"""plots.py -- every function takes DataFrames and returns a matplotlib Figure (no globals)."""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loading import FEATURES, TERM_ORDER, multi_value_datasets
from diagnostics import WR_COLS, N_REL

sns.set_theme(style="whitegrid", context="notebook")
def _freq_sort_key(f):
    """Finest -> coarsest: '5T' < '15T' < 'H' < 'D' < 'W' < 'M' < 'Q' (seconds, rough for M/Q/Y)."""
    unit_seconds = {"S": 1, "T": 60, "MIN": 60, "H": 3600, "D": 86400, "B": 86400,
                    "W": 7 * 86400, "M": 30 * 86400, "Q": 91 * 86400, "A": 365 * 86400, "Y": 365 * 86400}
    m = re.match(r"^(\d*)([A-Za-z]+)", str(f))
    if not m:
        return float("inf")
    mult = int(m.group(1) or 1)
    return mult * unit_seconds.get(m.group(2).upper(), 1e12)


TERM_COLORS = {"short": "#2a9d8f", "medium": "#e9c46a", "long": "#e76f51"}


#  all datasets
def plot_config_overview(df_config):
    """All configs sorted by median MASE; bar = IQR of window MASE, dot = median, colour = term."""
    d = df_config.sort_values("median_MASE").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * len(d))))
    for i, r in d.iterrows():
        ax.plot([r.q25_MASE, r.q75_MASE], [i, i], color=TERM_COLORS[str(r.term)], alpha=0.5, lw=3)
    ax.scatter(d.median_MASE, d.index, c=[TERM_COLORS[str(t)] for t in d.term], s=18, zorder=3)
    ax.set_yticks(d.index)
    ax.set_yticklabels(d.config, fontsize=6)
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("window MASE (log) -- dot = median, bar = IQR")
    ax.set_title("All dataset | freq | term configurations")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in TERM_COLORS.items()]
    ax.legend(handles=handles, title="term", loc="lower right")
    fig.tight_layout()
    plt.close(fig)  
    return fig


def plot_mase_by_dataset(df_windows):
    """Distribution of window-level MASE per dataset (all freqs/terms pooled), sorted by median."""
    d = df_windows.dropna(subset=["MASE"])
    d = d[d["MASE"] > 0]
    order = d.groupby("dataset", observed=True)["MASE"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(order))))
    sns.boxplot(data=d, y="dataset", x="MASE", order=order, showfliers=False,
                color="#8ecae6", ax=ax, linewidth=0.8)
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_title("Window-level MASE per dataset (pooled over freq/term, outliers hidden)")
    fig.tight_layout()
    plt.close(fig)  
    return fig


def plot_feature_profile(df_config, features=FEATURES):
    """Dataset x feature heatmap of percentile ranks -> 'what does each dataset look like?'
    (ranks instead of raw values because features have very different scales / outliers)."""
    per_ds = df_config.groupby("dataset", observed=True)[features].median()
    missing = per_ds.index[per_ds.isna().all(axis=1)].tolist()
    if missing:
        print("No features available, left out of the profile:", missing)
    per_ds = per_ds.dropna(how="all")
    ranks = per_ds.rank(pct=True)
    ranks = ranks.loc[per_ds.index]
    g = sns.clustermap(ranks.fillna(0.5), cmap="viridis", figsize=(9, max(6, 0.25 * len(ranks))),
                       yticklabels=True, cbar_kws={"label": "percentile rank across datasets"})
    g.ax_heatmap.set_title("Dataset feature profiles (clustered)", pad=40)
    plt.close(g.fig)
    return g.fig


#  same dataset, other freq / term
def plot_term_comparison(df_config):
    """Slope plot: for every dataset with >1 term, median MASE as a function of term."""
    ds = multi_value_datasets(df_config, "term")
    d = df_config[df_config.dataset.isin(ds)]
    # one line per dataset|freq
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, f), g in d.groupby(["dataset", "freq"], observed=True):
        g = g.sort_values("term")
        if g.term.nunique() < 2:
            continue
        ax.plot(g.term.astype(str), g.median_MASE, marker="o", alpha=0.6, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("term")
    ax.set_ylabel("median window MASE (log)")
    ax.set_title("Effect of forecast term (each line = one dataset|freq)")
    fig.tight_layout()
    plt.close(fig) 
    return fig


def plot_freq_comparison(df_config, dataset):
    """For one dataset: median MASE (with IQR) per frequency, one panel/line per term."""
    d = df_config[df_config.dataset == dataset].copy()
    d["freq"] = d["freq"].astype(str)
    freq_order = sorted(d["freq"].unique(), key=_freq_sort_key)
    d["freq"] = pd.Categorical(d["freq"], categories=freq_order, ordered=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in TERM_ORDER:
        g = d[d.term == t].sort_values("freq")
        if g.empty:
            continue
        ax.errorbar(g.freq.astype(str), g.median_MASE,
                    yerr=[g.median_MASE - g.q25_MASE, g.q75_MASE - g.median_MASE],
                    marker="o", capsize=3, label=t, color=TERM_COLORS[t])
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(freq_order)))
    ax.set_xticklabels(freq_order)
    ax.set_xlabel("frequency (fine -> coarse)")
    ax.set_ylabel("window MASE (median, IQR)")
    ax.set_title(f"{dataset}: frequency / term comparison")
    ax.legend(title="term")
    fig.tight_layout()
    plt.close(fig)  
    return fig


#  best vs worst
def plot_best_worst(df_windows, best, worst):
    """Left: ECDF of window MASE for best (blue) and worst (red) configs.
    Right: feature percentile profile of those configs (relative to all configs in `best`+`worst`)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})
    blues = sns.color_palette("Blues_d", len(best))
    reds = sns.color_palette("Reds_d", len(worst))
    for group, pal in ((best, blues), (worst, reds)):
        for (_, r), c in zip(group.iterrows(), pal):
            x = df_windows.loc[df_windows.config == r.config, "MASE"].dropna()
            x = np.sort(x[x > 0])
            ax1.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, label=f"{r.config} (n={len(x)})")
    ax1.axvline(1, color="k", ls="--", lw=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("window MASE (log)")
    ax1.set_ylabel("cumulative share of windows")
    ax1.set_title("MASE distribution: best (blue) vs worst (red)")
    ax1.legend(fontsize=6, loc="lower right")

    both = pd.concat([best.assign(group="best"), worst.assign(group="worst")])
    prof = both.set_index("config")[FEATURES]
    prof = (prof - prof.mean()) / (prof.std() + 1e-9)
    sns.heatmap(prof, cmap="coolwarm", center=0, annot=True, fmt=".1f", annot_kws={"size": 7},
                cbar_kws={"label": "z-score within these configs"}, ax=ax2)
    ax2.set_title("Feature profile (best first, worst last)")
    fig.tight_layout()
    plt.close(fig) 
    return fig


#   behaviour plots.  They take df_analysis = merge_with_windows(df_windows, df_forecasts)
#  (one row per window: errors + series features + forecast diagnostics), except where noted.
def _done(fig):
    fig.tight_layout()
    plt.close(fig)
    return fig


def _term_color(t):
    return TERM_COLORS.get(str(t), "grey")


def _binned(x, y, n_bins=10):
    """Median and IQR of y in n_bins equal-count bins of x (x should already be a percentile rank)."""
    b = np.minimum((x * n_bins).astype(int), n_bins - 1)
    g = pd.Series(y.values).groupby(b.values)
    return pd.DataFrame({"x": (g.median().index + 0.5) / n_bins, "med": g.median().values,
                         "q25": g.quantile(0.25).values, "q75": g.quantile(0.75).values,
                         "n": g.size().values})


#  uncertainty
def plot_uncertainty_evolution(df_analysis, dataset):
    """How does the 10-90% interval widen along the horizon?  Width is divided by the forecast's own
    mean width and the horizon is rescaled to 0..1, so configurations with different horizon lengths
    and different units are comparable. 1.0 = average width; >1 late in the horizon = widening."""
    d = df_analysis[df_analysis["dataset"] == dataset]
    grid = np.linspace(0, 1, N_REL)
    fig, ax = plt.subplots(figsize=(11, 5))
    for (f, t), g in d.groupby(["freq", "term"], observed=True):
        g = g.dropna(subset=[WR_COLS[0]])
        if g.empty:
            continue
        med, lo, hi = g[WR_COLS].median().values, g[WR_COLS].quantile(.25).values, g[WR_COLS].quantile(.75).values
        line, = ax.plot(grid, med, label=f"{f} | {t} (H={int(g.horizon.median())}, n={len(g)})")
        ax.fill_between(grid, lo, hi, alpha=0.12, color=line.get_color())
    ax.axhline(1, color="k", ls=":", lw=1)
    ax.set_xlabel("relative forecast horizon (0 = first step, 1 = last step)")
    ax.set_ylabel("10-90% width / mean width of the forecast")
    ax.set_title(f"Uncertainty evolution: {dataset} (median, IQR over windows)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _done(fig)


def plot_width_growth_overview(df_analysis, label_top=6):
    """One dot per configuration: median (width in last quarter / width in first quarter) vs horizon length.
    Shows whether strong widening is a property of the data or just of long horizons."""
    d = df_analysis.dropna(subset=["width_growth"])
    c = d.groupby("config").agg(growth=("width_growth", "median"), horizon=("horizon", "first"),
                                term=("term", "first"), n=("width_growth", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(c.horizon, c.growth, s=15 + 60 * np.sqrt(c.n / c.n.max()), alpha=0.7,
               c=[_term_color(t) for t in c.term], edgecolor="k", lw=0.3)
    for _, r in c.nlargest(label_top, "growth").iterrows():
        ax.annotate(r.config, (r.horizon, r.growth), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("horizon length (steps, log)")
    ax.set_ylabel("median width growth: last quarter / first quarter (log)")
    ax.set_title("Uncertainty growth vs horizon length (dot size = number of windows)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=col, label=t) for t, col in TERM_COLORS.items()],
              title="term")
    return _done(fig)


def plot_width_vs_error(df_analysis, min_windows=30, n_bins=10):
    """Does the model's uncertainty know when it is wrong?
    Left : within each configuration, windows are ranked by interval width / series std; the median MASE
           of each width-decile is shown (pooled over all configs). Rising = wide intervals go with big errors.
    Right: Spearman correlation (width rank vs MASE rank) per configuration."""
    d = df_analysis.dropna(subset=["width_norm", "MASE"])
    d = d[d["MASE"] > 0].copy()
    d["w_rank"] = d.groupby("config")["width_norm"].rank(pct=True)
    b = _binned(d["w_rank"], d["MASE"], n_bins)
    rhos = []
    for cfg, x in d.groupby("config", observed=True):
        if len(x) >= min_windows and x["width_norm"].nunique() > 1:
            rhos.append((cfg, x["width_norm"].rank().corr(x["MASE"].rank()), len(x)))
    r = pd.DataFrame(rhos, columns=["config", "rho", "n"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.errorbar(b.x, b.med, yerr=[b.med - b.q25, b.q75 - b.med], marker="o", capsize=3)
    a1.set_yscale("log"); a1.axhline(1, color="k", ls="--", lw=1)
    a1.set_xlabel("width percentile within configuration (0 = narrowest)")
    a1.set_ylabel("window MASE (median, IQR)")
    a1.set_title(f"Error by predicted-uncertainty decile (n={len(d):,} windows)")
    a2.hist(r.rho, bins=20, color="#8ecae6", edgecolor="w")
    a2.axvline(0, color="k", lw=1)
    a2.axvline(r.rho.median(), color="r", ls="--", label=f"median rho = {r.rho.median():.2f}")
    a2.set_xlabel("Spearman rho (width vs MASE) per configuration")
    a2.set_ylabel("number of configurations")
    a2.set_title(f"{len(r)} configurations with >= {min_windows} windows")
    a2.legend()
    return _done(fig)


def plot_crossing_overview(df_analysis):
    """Quantile crossing per configuration. x = share of forecasts with >= 1 crossing; y = how severe the
    worst crossing is, relative to the interval width (median over the crossing forecasts).
    Points below the dashed line are tiny violations (likely numerical noise, float16 outputs)."""
    d = df_analysis.dropna(subset=["cross_any"])
    g = d.groupby("config")
    share = g["cross_any"].mean()
    sev = d[d.cross_any == 1].groupby("config")["cross_rel"].median()
    c = pd.DataFrame({"share": share, "severity": sev, "n": g.size()}).reset_index()
    n_zero = int((c.share == 0).sum())
    c = c[c.share > 0].dropna()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(c.share * 100, c.severity, s=15 + 60 * np.sqrt(c.n / c.n.max()), alpha=0.7, edgecolor="k", lw=0.3)
    for _, r in c.nlargest(6, "severity").iterrows():
        ax.annotate(r.config, (r.share * 100, r.severity), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.01, color="r", ls="--", lw=1)
    ax.text(ax.get_xlim()[0], 0.011, " 1% of interval width", color="r", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("% of forecasts with at least one crossing (log)")
    ax.set_ylabel("worst crossing / interval width (median, log)")
    ax.set_title(f"Quantile crossing ({n_zero} of {n_zero + len(c)} configurations have none)")
    return _done(fig)


# shape of the median forecast
def plot_drift_distribution(df_analysis, dataset):
    """Trend of the median forecast (fitted change over the horizon, in units of the series' std)
    per window, for every freq|term of one dataset. 0 = no trend. NOT a bias measure: a trending
    series should have a trending forecast -- see plot_drift_vs_history_trend for that comparison."""
    d = df_analysis[(df_analysis["dataset"] == dataset)].dropna(subset=["trend_change_std"]).copy()
    d["Configuration"] = d["freq"].astype(str) + " | " + d["term"].astype(str)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * d.Configuration.nunique()), 5))
    sns.violinplot(data=d, x="Configuration", y="trend_change_std", inner="quartile", cut=0, ax=ax)
    lo, hi = d.trend_change_std.quantile([0.01, 0.99])
    ax.set_ylim(min(lo, 0) * 1.1, max(hi, 0) * 1.1)
    ax.axhline(0, color="red", ls="--")
    ax.set_ylabel("forecast trend over horizon / series std")
    ax.set_title(f"Direction of the median forecast: {dataset} (1-99% range shown)")
    ax.tick_params(axis="x", rotation=45)
    return _done(fig)


def plot_drift_vs_history_trend(df_analysis, trend_col="feat_linearity", min_trend_strength=0.5, min_windows=20):
    """Per dataset: share of windows whose forecast trend has the same SIGN as the series' historical trend
    (only series with feat_trend_strength >= min_trend_strength, non-flat forecasts).
    ~1 = trend is continued, ~0 = trend is reversed, 0.5 = unrelated.
    CHECK that the sign of `trend_col` really encodes the trend direction in your feature set."""
    d = df_analysis.dropna(subset=["trend_change_std", trend_col, "feat_trend_strength"])
    d = d[(d.feat_trend_strength >= min_trend_strength) & (d.is_flat == 0) & (d[trend_col] != 0)].copy()
    d["agree"] = np.sign(d.trend_change_std) == np.sign(d[trend_col])
    r = d.groupby("dataset", observed=True)["agree"].agg(["mean", "size"])
    r = r[r["size"] >= min_windows].sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(r))))
    ax.barh(r.index, r["mean"], color="#8ecae6")
    for i, (m, n) in enumerate(zip(r["mean"], r["size"])):
        ax.text(min(m + 0.01, 0.97), i, f"n={n}", va="center", fontsize=7)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of windows: forecast trend sign == historical trend sign")
    ax.set_title(f"Does the forecast continue the historical trend? (trend strength >= {min_trend_strength})")
    return _done(fig)


def plot_acf_comparison(acf_df, dataset):
    """Autocorrelation of the median forecast at increasing lags (median and IQR over windows), per freq|term.
    Input: acf_profiles(results_dir, dataset). High slowly-decaying ACF = smooth/persistent forecast;
    a bump at the seasonal lag = the forecast reproduces seasonality; ~0 beyond lag 1-2 = jagged forecast."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for (f, t), g in acf_df.groupby(["freq", "term"], observed=True):
        g = g.sort_values("lag")
        line, = ax.plot(g.lag, g["median"], marker="o", ms=3,
                        label=f"{f} | {t} (H={int(g.horizon.iloc[0])}, n={int(g.n.iloc[0])})")
        ax.fill_between(g.lag, g.p25, g.p75, alpha=0.1, color=line.get_color())
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("lag (forecast steps)")
    ax.set_ylabel("autocorrelation of median forecast")
    ax.set_title(f"Forecast ACF: {dataset}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _done(fig)


def plot_seasonal_damping(df_analysis, top_flat=15):
    """Left : does the forecast keep the amplitude the history suggests? For forecasts covering at least one
             full season (horizon >= feat_period1): std of the median forecast / series std, by seasonal-strength
             bin. Dashed = sqrt(strength), a ROUGH reference if only the seasonal part were reproduced.
             Far below the reference = damped / flattened seasonality.
    Right: share of forecasts whose median is (almost) constant, per configuration -- flat-lining."""
    d = df_analysis.dropna(subset=["amp_rel", "feat_seasonal_strength", "feat_period1", "horizon"])
    d = d[(d.feat_period1 > 1) & (d.horizon >= d.feat_period1)].copy()
    d["bin"] = pd.cut(d.feat_seasonal_strength, [0, .2, .4, .6, .8, 1.0], include_lowest=True)
    g = d.groupby("bin", observed=True)["amp_rel"]
    s = pd.DataFrame({"med": g.median(), "q25": g.quantile(.25), "q75": g.quantile(.75), "n": g.size()}).reset_index()
    s["x"] = [iv.mid for iv in s["bin"]]
    flat = (df_analysis.dropna(subset=["is_flat"]).groupby("config")["is_flat"]
            .agg(["mean", "size"]).query("`mean` > 0").nlargest(top_flat, "mean"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    a1.errorbar(s.x, s.med, yerr=[s.med - s.q25, s.q75 - s.med], marker="o", capsize=3, label="forecast")
    xs = np.linspace(0, 1, 50)
    a1.plot(xs, np.sqrt(xs), "k--", label="sqrt(seasonal strength)")
    for _, r in s.iterrows():
        a1.text(r.x, r.q25, f"n={int(r.n)}", fontsize=7, ha="center", va="top")
    a1.set_xlabel("historical seasonal strength"); a1.set_ylabel("std(median forecast) / series std")
    a1.set_title("Amplitude of seasonal forecasts"); a1.legend()
    if len(flat):
        a2.barh(flat.index[::-1], flat["mean"][::-1] * 100, color="#e76f51")
    a2.set_xlabel("% of forecasts with a (nearly) constant median")
    a2.set_title(f"Flat-lining (top {len(flat)} configurations)")
    a2.tick_params(axis="y", labelsize=7)
    return _done(fig)


#  inputs vs error, error in time
def plot_feature_error_relationship(df_series, dataset=None, features=FEATURES, n_bins=10):
    """For every scale-free feature: median MASE (and IQR) per decile of that feature, with Spearman rho.
    Pooled over all series (or one dataset). Uses series-level median MASE, so long series do not dominate."""
    d = df_series if dataset is None else df_series[df_series["dataset"] == dataset]
    d = d[d["MASE"] > 0]
    ncol = 5
    nrow = int(np.ceil(len(features) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, features):
        x = d[[f, "MASE"]].dropna()
        if len(x) < n_bins * 2:
            ax.set_title(f"{f}\n(too few series)", fontsize=8); continue
        rk = x[f].rank(pct=True)
        b = _binned(rk, x["MASE"], n_bins)
        ax.errorbar(b.x, b.med, yerr=[b.med - b.q25, b.q75 - b.med], marker="o", ms=3, capsize=2)
        rho = rk.corr(x["MASE"].rank())
        ax.set_yscale("log"); ax.axhline(1, color="k", ls="--", lw=0.8)
        ax.set_title(f"{f.replace('feat_', '')}\nrho={rho:.2f}, n={len(x)}", fontsize=9)
        ax.set_xlabel("feature percentile")
    for ax in axes[len(features):]:
        ax.axis("off")
    axes[0].set_ylabel("series MASE (median, IQR)")
    fig.suptitle("Series features vs error" + (f": {dataset}" if dataset else " (all series)"), y=1.02)
    return _done(fig)


def plot_error_by_window(df_analysis, configs, ncols=3):
    """Median MASE (and IQR across series/variates) per window_idx for chosen configurations: are failures
    spread evenly or concentrated in certain windows? ASSUMES window_idx follows time -- """
    n = len(configs)
    nrow = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrow, ncols, figsize=(5 * ncols, 3.4 * nrow), squeeze=False)
    for ax, cfg in zip(axes.ravel(), configs):
        d = df_analysis[(df_analysis["config"] == cfg)].dropna(subset=["MASE"])
        d = d[d["MASE"] > 0]
        g = d.groupby("window_idx")["MASE"]
        s = pd.DataFrame({"med": g.median(), "q25": g.quantile(.25), "q75": g.quantile(.75)})
        ax.plot(s.index, s.med, marker="o", ms=3)
        ax.fill_between(s.index, s.q25, s.q75, alpha=0.2)
        ax.axhline(1, color="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{cfg}\n{d.groupby(['series_idx', 'variate_idx']).ngroups} series-variates", fontsize=8)
        ax.set_xlabel("window_idx")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].set_ylabel("MASE")
    return _done(fig)


#  qualitative
def plot_forecast_fan(q, levels, series_idx=0, window_idx=0, variate_idx=0, title=""):
    """Quantile fan of ONE forecast. q has the layout (series, window, quantile, variate, horizon);
    get it with diagnostics.load_config(...). Use pick_windows_by_error to choose windows transparently."""
    f = q[series_idx, window_idx, :, variate_idx, :]
    h = np.arange(1, f.shape[-1] + 1)
    i = [int(np.argmin(np.abs(levels - l))) for l in (0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9)]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(h, f[i[0]], f[i[6]], alpha=0.15, label="10-90%")
    ax.fill_between(h, f[i[1]], f[i[5]], alpha=0.25, label="30-70%")
    ax.fill_between(h, f[i[2]], f[i[4]], alpha=0.35, label="40-60%")
    ax.plot(h, f[i[3]], lw=2, label="median")
    ax.set_xlabel("forecast horizon step"); ax.set_ylabel("predicted value")
    ax.set_title(f"{title}  series {series_idx} | window {window_idx} | variate {variate_idx}")
    ax.legend()
    return _done(fig)