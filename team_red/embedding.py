"""embedding.py -- UMAP embedding of the series' input features + cluster analysis.

Idea: place every series (a point) by its scale-free features ONLY (no model output used),
then colour the map with what the model DID (error, flat-lining, uncertainty ...).
Regions where behaviour changes are then explained by what the inputs look like there.

Unit of analysis: one (dataset, freq, series_idx, variate_idx). Features are series-level, so a series is
embedded once; outcomes are taken from ONE term (default 'short', present in every dataset).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors

from data_loading import FEATURES

SERIES_ID = ["dataset", "freq", "series_idx", "variate_idx"]
OUTCOMES = ["median_MASE", "share_worse_than_naive", "flat_share", "width_growth", "width_norm", "amp_rel", "acf1"]


#  series table
def build_series_table(df_analysis, term="short", features=FEATURES):
    """One row per series-variate: features (constant per series) + behaviour summaries over its windows."""
    d = df_analysis[df_analysis["term"].astype(str) == term].copy()
    d["worse"] = (d["MASE"] > 1).astype(float).where(d["MASE"].notna())
    g = d.groupby(SERIES_ID, observed=True)
    out = g[features].first()
    out["median_MASE"] = g["MASE"].median()
    out["share_worse_than_naive"] = g["worse"].mean()
    out["flat_share"] = g["is_flat"].mean()
    out["width_growth"] = g["width_growth"].median()
    out["width_norm"] = g["width_norm"].median()
    out["amp_rel"] = g["amp_rel"].median()
    out["acf1"] = g["acf1"].median()
    out["horizon"] = g["horizon"].first()
    out["n_windows"] = g.size()
    out["term"] = term
    return out.reset_index()


#  embedding
def _prepare(tbl, features):
    ok = tbl[features].notna().all(axis=1)
    dropped = tbl.loc[~ok, "dataset"].value_counts()
    if len(dropped):
        print("Rows dropped (missing features):", dropped.to_dict())
    t = tbl[ok].reset_index(drop=True).copy()
    qt = QuantileTransformer(output_distribution="normal", random_state=0,
                             n_quantiles=min(1000, len(t)))       # tames heavy tails (kurtosis, ARCH)
    X = qt.fit_transform(t[features].values)
    for j, f in enumerate(features):
        t["z_" + f] = X[:, j]
    return t, X


def embed(tbl, features=FEATURES, method="umap", n_neighbors=30, min_dist=0.1,
          max_per_dataset=500, density_k=15, random_state=0):
    """Returns df_emb: tbl (rows with complete features) + emb_x, emb_y, in_fit, density_dist, z_<feature>.

    max_per_dataset : the map is FITTED on at most this many series per dataset so that huge datasets
                      (thousands of series) do not dictate the layout; all other series are projected
                      into the fitted map afterwards (UMAP/PCA only; t-SNE needs max_per_dataset=None).
    density_dist    : mean distance to the density_k nearest neighbours in FEATURE space
                      (high = sparse region of input space).
    """
    t, X = _prepare(tbl, features)
    rng = np.random.default_rng(random_state)
    if max_per_dataset is None:
        fit = np.ones(len(t), bool)
    else:
        fit = np.zeros(len(t), bool)
        for _, idx in t.groupby("dataset").indices.items():
            take = idx if len(idx) <= max_per_dataset else rng.choice(idx, max_per_dataset, replace=False)
            fit[take] = True

    if method == "umap":
        import umap                                                   # pip install umap-learn
        reducer = umap.UMAP(n_neighbors=min(n_neighbors, fit.sum() - 1), min_dist=min_dist,
                            n_components=2, random_state=random_state)
        Z = np.zeros((len(t), 2))
        Z[fit] = reducer.fit_transform(X[fit])
        if (~fit).any():
            Z[~fit] = reducer.transform(X[~fit])
    elif method == "pca":
        from sklearn.decomposition import PCA
        p = PCA(2, random_state=random_state).fit(X[fit])
        Z = p.transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        if not fit.all():
            raise ValueError("t-SNE cannot project new points: use max_per_dataset=None")
        Z = TSNE(2, perplexity=min(30, len(t) - 1), random_state=random_state, init="pca").fit_transform(X)
    else:
        raise ValueError(method)

    k = min(density_k, len(t) - 1)
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    t["emb_x"], t["emb_y"] = Z[:, 0], Z[:, 1]
    t["in_fit"] = fit
    t["density_dist"] = dist[:, 1:].mean(axis=1)
    return t


# clustering
def cluster_embedding(df_emb, method="hdbscan", min_cluster_size=60, min_samples=None, k=8, selection="eom"):
    """Adds `cluster` (-1 = noise for HDBSCAN). selection="leaf" gives many small clusters instead of a few big ones. Clusters are found in the 2-D map, so they are a
    description of what you SEE; always check them against the original features (cluster_profile)."""
    Z = df_emb[["emb_x", "emb_y"]].values
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True,
                      cluster_selection_method=selection).fit_predict(Z)
    elif method == "kmeans":
        from sklearn.cluster import KMeans
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    else:
        raise ValueError(method)
    out = df_emb.copy()
    out["cluster"] = lab
    return out


#  what is a cluster?
def _composition(s):
    p = s.value_counts(normalize=True)
    top = ", ".join(f"{k} {v:.0%}" for k, v in p.head(3).items())
    return pd.Series({"n_datasets": len(p), "top_dataset_share": p.iloc[0], "top_datasets": top,
                      "dataset_entropy": float(-(p * np.log(p)).sum())})


def cluster_profile(df_emb, features=FEATURES, outcomes=OUTCOMES):
    """One row per cluster: size, dataset composition (is it just ONE dataset?), mean feature z-scores
    (what the inputs look like), median outcomes (what the model does), and sparsity/compactness."""
    rows = []
    for c, g in df_emb.groupby("cluster"):
        r = {"cluster": c, "n": len(g)}
        r.update(_composition(g["dataset"]).to_dict())
        cx, cy = g.emb_x.median(), g.emb_y.median()
        r["spread_2d"] = float(np.median(np.hypot(g.emb_x - cx, g.emb_y - cy)))      # compactness on the map
        r["sparsity_feat"] = g["density_dist"].median()                              # sparsity in feature space
        for f in features:
            r["z_" + f.replace("feat_", "")] = g["z_" + f].mean()
        for o in outcomes:
            r[o] = g[o].median()
        rows.append(r)
    return pd.DataFrame(rows).set_index("cluster")


def distinguishing_features(df_emb, cluster, features=FEATURES, outcomes=OUTCOMES):
    """Cluster vs. all other series. AUC = P(random member > random non-member): 0.5 = no difference,
    1 = always higher, 0 = always lower. Rank-based, so robust to heavy tails."""
    from scipy.stats import mannwhitneyu
    inn = df_emb["cluster"] == cluster
    rows = []
    for col, kind in [(f, "feature") for f in features] + [(o, "outcome") for o in outcomes]:
        a, b = df_emb.loc[inn, col].dropna(), df_emb.loc[~inn, col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        u = mannwhitneyu(a, b).statistic
        rows.append({"variable": col.replace("feat_", ""), "kind": kind, "AUC": u / (len(a) * len(b)),
                     "cluster_median": a.median(), "rest_median": b.median()})
    out = pd.DataFrame(rows)
    out["strength"] = (out.AUC - 0.5).abs()
    return out.sort_values("strength", ascending=False).reset_index(drop=True)


def subcluster(df_emb, cluster, features=FEATURES, method="umap", min_cluster_size=25, n_neighbors=15,
               cluster_method="hdbscan", k=3):
    """Re-embed and re-cluster ONLY the members of one cluster, to look for sub-groups where the
    model behaves differently. Returns the member rows with a new `sub_cluster` column."""
    sub = df_emb[df_emb["cluster"] == cluster].copy()
    drop = [c for c in sub.columns if c.startswith("z_")] + ["emb_x", "emb_y", "in_fit", "density_dist"]
    sub = sub.drop(columns=drop).rename(columns={"cluster": "parent_cluster"})
    sub = embed(sub, features=features, method=method, n_neighbors=min(n_neighbors, max(2, len(sub) - 2)),
                max_per_dataset=None)
    sub = cluster_embedding(sub, method=cluster_method, min_cluster_size=min_cluster_size, k=k)
    return sub.rename(columns={"cluster": "sub_cluster"})


def window_behavior_by_cluster(df_analysis, df_emb, term="short", min_windows=30):
    """Window-level behaviour per cluster (uses ALL windows of the cluster's series):
    error, flat-lining, crossing, width growth, amplitude and -- key question -- does interval width
    track error inside this cluster (Spearman rho)?"""
    lab = df_emb[SERIES_ID + ["cluster"]]
    d = df_analysis[df_analysis["term"].astype(str) == term].merge(lab, on=SERIES_ID, how="inner")
    rows = []
    for c, g in d.groupby("cluster"):
        gg = g.dropna(subset=["width_norm", "MASE"])
        rho = (gg["width_norm"].rank().corr(gg["MASE"].rank())
               if len(gg) >= min_windows and gg["width_norm"].nunique() > 1 else np.nan)
        rows.append({"cluster": c, "n_windows": len(g), "n_series": g.groupby(SERIES_ID).ngroups,
                     "median_MASE": g["MASE"].median(), "share_worse": (g["MASE"] > 1).mean(),
                     "flat_share": g["is_flat"].mean(), "cross_share": g["cross_any"].mean(),
                     "width_growth": g["width_growth"].median(), "amp_rel": g["amp_rel"].median(),
                     "rho_width_vs_error": rho})
    return pd.DataFrame(rows).set_index("cluster")


def dataset_purity(df_emb, features=FEATURES, k=15):
    """For every series: share of its k nearest neighbours (in the z-scored FEATURE space, not the 2-D map)
    that belong to the same dataset.
    IMPORTANT: a dataset with n series can contribute at most n-1 neighbours, so small datasets can never
    reach purity 1. Therefore also returned:
      purity_max  = min(1, (n_dataset - 1) / k)       highest value possible for this series
      purity_norm = purity / purity_max               1 = as pure as the dataset size allows
    Use purity_by_dataset() for a per-dataset summary."""
    X = df_emb[["z_" + f for f in features]].values
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    ds = pd.factorize(df_emb["dataset"])[0]
    out = df_emb.copy()
    out["purity"] = (ds[idx[:, 1:]] == ds[:, None]).mean(axis=1)
    n_ds = out.groupby("dataset")["dataset"].transform("size")
    out["purity_max"] = np.minimum(1.0, (n_ds - 1) / k)
    out["purity_norm"] = out["purity"] / out["purity_max"].where(out["purity_max"] > 0)
    return out


def purity_by_dataset(df_emb):
    """Per dataset: number of series, mean purity, the maximum possible, and the normalized value.
    Datasets with fewer than ~16 series have a low ceiling -- read purity_norm, not purity."""
    g = df_emb.groupby("dataset")
    out = pd.DataFrame({"n_series": g.size(), "purity": g["purity"].mean(),
                        "purity_max": g["purity_max"].mean(), "purity_norm": g["purity_norm"].mean()})
    return out.sort_values("n_series", ascending=False)