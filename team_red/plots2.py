import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_failure_rate(df_analysis):
    failure = df_analysis.copy()

    failure["failed"] = failure["MASE"] > 1

    failure_rate = (
        failure.groupby("term", observed=True)["failed"]
        .mean() * 100
    )

    plt.figure(figsize=(8, 5))

    failure_rate.plot(
        kind="bar",
        color=["steelblue", "orange", "tomato"]
    )

    plt.ylabel("Failed forecasts (%)")
    plt.xlabel("Forecast horizon")
    plt.title("How often does TiRex-2 perform worse than naive?")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.show()

    return failure_rate


def plot_failure_heatmap(df_analysis, top_n=15):

    data = df_analysis.copy()

    data["failed"] = (data["MASE"] > 1).astype(float)

    # Select datasets with the most forecast windows
    top_datasets = data["dataset"].value_counts().head(top_n).index

    data = data[data["dataset"].isin(top_datasets)]

    # Calculate failure percentage
    heatmap_data = data.pivot_table(
        index="dataset",
        columns="term",
        values="failed",
        aggfunc="mean"
    ) * 100

    # Plot heatmap
    plt.figure(figsize=(10, 7))

    plt.imshow(
        heatmap_data,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=100
    )

    plt.colorbar(label="Failed forecasts (%)")

    plt.xticks(
        range(len(heatmap_data.columns)),
        heatmap_data.columns
    )

    plt.yticks(
        range(len(heatmap_data.index)),
        heatmap_data.index
    )

    plt.title("Forecast failure rate across datasets and horizons")
    plt.xlabel("Forecast horizon")
    plt.ylabel("Dataset")

    plt.tight_layout()
    plt.show()

    return heatmap_data



def plot_error_concentration(df_analysis, worst_pct=10):

    # Extract positive MASE values
    errors = df_analysis["MASE"].dropna()
    errors = errors[errors > 0].sort_values().values

    if len(errors) == 0:
        print("No valid positive MASE values available.")
        return None

    # Calculate cumulative error distribution
    cumulative_error = np.cumsum(errors) / np.sum(errors)

    cumulative_windows = (
        np.arange(1, len(errors) + 1) / len(errors)
    )

    # Plot Lorenz curve
    plt.figure(figsize=(8, 6))

    plt.plot(
        cumulative_windows,
        cumulative_error,
        linewidth=2,
        label="TiRex-2 error distribution"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="grey",
        label="Equal error contribution"
    )

    plt.xlabel("Cumulative proportion of forecast windows")
    plt.ylabel("Cumulative proportion of summed MASE")

    plt.title("Are forecasting errors concentrated in a few windows?")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Contribution of worst-performing windows
    n = max(1, int(np.ceil(len(errors) * worst_pct / 100)))

    contribution = errors[-n:].sum() / errors.sum() * 100

    print(
        f"Worst {worst_pct}% of windows contribute "
        f"{contribution:.2f}% of summed MASE"
    )

    return contribution



def plot_domain_behaviour(df_analysis, term="short"):

    # Filter forecasting horizon
    data = df_analysis[
        df_analysis["term"].astype(str) == term
    ].copy()

    # Aggregate behaviour by dataset
    domain = data.groupby("dataset", observed=True).agg(
        failure_rate=("MASE", lambda x: (x > 1).mean() * 100),
        median_MASE=("MASE", "median"),
        forecast_acf1=("acf1", "median"),
        n_windows=("MASE", "count")
    ).dropna(subset=["failure_rate"])

    # Sort datasets by failure rate
    domain = domain.sort_values(
        "failure_rate", ascending=False
    )

    x = np.arange(len(domain))

    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # TOP: Failure rate
    colors = [
        "tomato" if v > 50 else "steelblue"
        for v in domain["failure_rate"]
    ]

    axes[0].bar(x, domain["failure_rate"], color=colors)

    global_failure = data["MASE"].gt(1).mean() * 100

    axes[0].axhline(
        global_failure,
        color="green",
        linestyle="--",
        label=f"Overall failure rate ({global_failure:.1f}%)"
    )

    axes[0].set_ylabel("Failed forecasts (%)")
    axes[0].set_title(
        "TiRex-2 Failure Rate Across All Datasets"
    )
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # BOTTOM: Forecast trajectory autocorrelation
    axes[1].bar(
        x,
        domain["forecast_acf1"],
        color="mediumpurple"
    )

    axes[1].axhline(
        domain["forecast_acf1"].median(),
        color="green",
        linestyle="--",
        label="Median across datasets"
    )

    axes[1].axhline(0, color="black", linewidth=0.8)

    axes[1].set_ylabel("Median forecast ACF1")
    axes[1].set_title(
        "Autocorrelation of Predicted Median Trajectories"
    )
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        domain.index,
        rotation=65,
        ha="right",
        fontsize=8
    )

    fig.suptitle(
        f"Cross-Domain Analysis of TiRex-2 Forecast Behaviour ({term})",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

    return domain.round(3)

import embedding as E

def plot_tsne_comparison(tsne_result):

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # 1. Colour by dataset
    datasets = tsne_result["dataset"].astype(str)
    dataset_codes, dataset_names = pd.factorize(datasets)

    axes[0].scatter(
        tsne_result["emb_x"],
        tsne_result["emb_y"],
        c=dataset_codes,
        cmap="turbo",
        s=18,
        alpha=0.75
    )

    axes[0].set_title("t-SNE: Dataset Distribution")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Dataset legend
    handles = []

    for i, name in enumerate(dataset_names):
        handles.append(
            plt.Line2D(
                [], [],
                marker="o",
                linestyle="",
                color=plt.cm.turbo(
                    i / max(1, len(dataset_names) - 1)
                ),
                label=name,
                markersize=6
            )
        )

    axes[0].legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7
    )

    # 2. Colour by median MASE
    sc2 = axes[1].scatter(
        tsne_result["emb_x"],
        tsne_result["emb_y"],
        c=tsne_result["median_MASE"],
        cmap="plasma",
        s=18,
        alpha=0.75
    )

    axes[1].set_title("t-SNE: Median Forecast Error")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    plt.colorbar(
        sc2, ax=axes[1], label="Median MASE"
    )

    # 3. Colour by failure share
    sc3 = axes[2].scatter(
        tsne_result["emb_x"],
        tsne_result["emb_y"],
        c=tsne_result["share_worse_than_naive"],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        s=18,
        alpha=0.75
    )

    axes[2].set_title("t-SNE: Failure Share")
    axes[2].set_xlabel("t-SNE 1")
    axes[2].set_ylabel("t-SNE 2")

    plt.colorbar(
        sc3,
        ax=axes[2],
        label="Share of forecasts with MASE > 1"
    )

    for ax in axes:
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Time-Series Feature Space: Dataset Identity vs TiRex-2 Behaviour",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()



from sklearn.neighbors import NearestNeighbors
import embedding as E


def plot_local_failure_similarity(tsne_result, n_neighbors=15):
    """
    Measures whether similar time series in the original feature space
    have similar TiRex-2 failure rates.

    Adds a local_failure_similarity column to tsne_result
    and plots its distribution.
    """
    import embedding as E

    # Standardized feature representation
    feature_cols = ["z_" + f for f in E.FEATURES]
    X = tsne_result[feature_cols].values

    # Find nearest neighbours in original feature space
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    # Failure share for each series
    failure = tsne_result["share_worse_than_naive"].to_numpy()

    # Compare each series with its neighbours
    neighbour_failure = failure[indices[:, 1:]]
    difference = np.abs(
        failure[:, None] - neighbour_failure
    )

    # Similarity: 1 = identical failure shares
    local_similarity = 1 - difference.mean(axis=1)

    tsne_result = tsne_result.copy()
    tsne_result["local_failure_similarity"] = local_similarity

    print(
        "Mean local failure similarity:",
        round(local_similarity.mean(), 3)
    )

    # Plot distribution
    plt.figure(figsize=(8, 5))
    plt.hist(
        local_similarity,
        bins=25,
        edgecolor="black",
        alpha=0.8
    )

    plt.xlabel("Local failure similarity")
    plt.ylabel("Number of series")
    plt.title(
        "Do Similar Time Series Have Similar TiRex-2 Failure Rates?"
    )

    plt.tight_layout()
    plt.show()

    return tsne_result



def create_interactive_dashboard(df_analysis, tsne_result=None):

    import pandas as pd
    import numpy as np
    import plotly.express as px
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    # Prepare data
    interactive_data = df_analysis.copy()

    for col in ["dataset", "freq", "term"]:
        interactive_data[col] = interactive_data[col].astype(str)

    if tsne_result is not None:
        tsne_data = tsne_result.copy()
        for col in ["dataset", "freq", "term"]:
            if col in tsne_data.columns:
                tsne_data[col] = tsne_data[col].astype(str)
    else:
        tsne_data = None

    ALL = "All"

    # Dropdown options
    datasets = [ALL] + sorted(interactive_data["dataset"].unique())
    frequencies = [ALL] + sorted(interactive_data["freq"].unique())
    terms = [ALL] + ["short", "medium", "long"]

    features = [
        "feat_x_entropy",
        "feat_trend_strength",
        "feat_trend_stability",
        "feat_seasonal_strength",
        "feat_e_entropy",
        "width_norm",
        "trend_change_std",
        "acf1"
    ]
    features = [f for f in features if f in interactive_data.columns]

    metrics = [
        m for m in ["MASE", "MAE", "RMSE", "CRPS"]
        if m in interactive_data.columns
    ]

    if not features or not metrics:
        raise ValueError("No supported feature or metric columns found.")

    # Controls
    dataset_dropdown = widgets.Dropdown(
        options=datasets, value=ALL,
        description="Dataset:",
        layout=widgets.Layout(width="320px")
    )

    freq_dropdown = widgets.Dropdown(
        options=frequencies, value=ALL,
        description="Frequency:",
        layout=widgets.Layout(width="320px")
    )

    term_dropdown = widgets.Dropdown(
        options=terms, value=ALL,
        description="Horizon:",
        layout=widgets.Layout(width="320px")
    )

    feature_dropdown = widgets.Dropdown(
        options=features,
        value="feat_x_entropy" if "feat_x_entropy" in features else features[0],
        description="X-axis:",
        layout=widgets.Layout(width="320px")
    )

    metric_dropdown = widgets.Dropdown(
        options=metrics,
        value="MASE" if "MASE" in metrics else metrics[0],
        description="Y-axis:",
        layout=widgets.Layout(width="320px")
    )

    tsne_colour = widgets.Dropdown(
        options=["Dataset", "Median MASE", "Failure share"],
        value="Median MASE",
        description="t-SNE colour:",
        layout=widgets.Layout(width="320px")
    )

    output = widgets.Output()

    def update_dashboard(dataset, freq, term, feature, metric, colour):

        d = interactive_data.copy()

        # Filter forecast-window data
        if dataset != ALL:
            d = d[d["dataset"] == dataset]

        if freq != ALL:
            d = d[d["freq"] == freq]

        if term != ALL:
            d = d[d["term"] == term]

        d = d.dropna(subset=[feature, metric])

        with output:
            clear_output(wait=True)

            if d.empty:
                print("No forecast data available for these selections.")
                return

            d["Forecast behaviour"] = np.where(
                d["MASE"] > 1,
                "Worse than naive",
                "Better than naive"
            )

            print("Forecast windows:", f"{len(d):,}")
            print("Median MASE:", round(d["MASE"].median(), 3))
            print(
                "Failure rate:",
                round((d["MASE"] > 1).mean() * 100, 2),
                "%"
            )

            # ------------------------------------------------
            # Plot 1: Feature vs forecast error
            # ------------------------------------------------

            scatter_data = d.sample(
                n=min(8000, len(d)),
                random_state=42
            )

            hover_cols = [
                c for c in [
                    "dataset", "freq", "term",
                    "series_idx", "window_idx", "MASE"
                ] if c in scatter_data.columns
            ]

            fig1 = px.scatter(
                scatter_data,
                x=feature,
                y=metric,
                color="Forecast behaviour",
                hover_data=hover_cols,
                color_discrete_map={
                    "Better than naive": "seagreen",
                    "Worse than naive": "tomato"
                },
                opacity=0.65,
                title=f"{feature} vs {metric}"
            )

            fig1.update_layout(
                height=520,
                template="plotly_white"
            )
            fig1.show()

            # ------------------------------------------------
            # Plot 2: Error distribution
            # ------------------------------------------------

            fig2 = px.box(
                d,
                x="term",
                y=metric,
                color="freq",
                points=False,
                title=f"{metric} distribution across forecast horizons"
            )

            fig2.update_layout(
                height=450,
                template="plotly_white"
            )
            fig2.show()

            # ------------------------------------------------
            # Plot 3: t-SNE map
            # ------------------------------------------------

            if tsne_data is not None:

                t = tsne_data.copy()

                if dataset != ALL:
                    t = t[t["dataset"] == dataset]

                if freq != ALL and "freq" in t.columns:
                    t = t[t["freq"] == freq]

                if term != ALL:
                    t = t[t["term"] == term]

                if not t.empty:

                    if colour == "Dataset":
                        color_col = "dataset"
                        palette = None
                    elif colour == "Median MASE":
                        color_col = "median_MASE"
                        palette = "Plasma"
                    else:
                        color_col = "share_worse_than_naive"
                        palette = "RdYlGn_r"

                    hover_cols = [
                        c for c in [
                            "dataset", "freq", "term",
                            "median_MASE", "share_worse_than_naive"
                        ] if c in t.columns
                    ]

                    fig3 = px.scatter(
                        t,
                        x="emb_x",
                        y="emb_y",
                        color=color_col,
                        hover_data=hover_cols,
                        color_continuous_scale=palette,
                        title=f"t-SNE map — coloured by {colour}"
                    )

                    fig3.update_layout(
                        height=650,
                        template="plotly_white"
                    )

                    fig3.show()

                else:
                    print("No t-SNE data available for these selections.")

    # Connect controls
    controls = widgets.VBox([
        widgets.HBox([dataset_dropdown, freq_dropdown]),
        widgets.HBox([term_dropdown, feature_dropdown]),
        widgets.HBox([metric_dropdown, tsne_colour])
    ])

    dashboard = widgets.interactive_output(
        update_dashboard,
        {
            "dataset": dataset_dropdown,
            "freq": freq_dropdown,
            "term": term_dropdown,
            "feature": feature_dropdown,
            "metric": metric_dropdown,
            "colour": tsne_colour
        }
    )

    display(controls, output)

    return dashboard



import plotly.express as px
import ipywidgets as widgets
from IPython.display import display


def create_tsne_dashboard(tsne_result):

    tsne_data = tsne_result.copy()

    tsne_data["dataset"] = tsne_data["dataset"].astype(str)
    tsne_data["term"] = tsne_data["term"].astype(str)

    # Dropdowns
    tsne_dataset = widgets.Dropdown(
        options=["All"] + sorted(tsne_data["dataset"].unique()),
        value="All",
        description="Dataset:"
    )

    available_terms = [
        t for t in ["short", "medium", "long"]
        if t in tsne_data["term"].unique()
    ]

    tsne_term = widgets.Dropdown(
        options=["All"] + available_terms,
        value="short" if "short" in available_terms else "All",
        description="Horizon:"
    )

    tsne_colour = widgets.Dropdown(
        options=[
            "Dataset",
            "Median MASE",
            "Failure share"
        ],
        value="Median MASE",
        description="Colour:"
    )

    # Update plot
    def update_tsne(dataset, term, colour):

        d = tsne_data.copy()

        if dataset != "All":
            d = d[d["dataset"] == dataset]

        if term != "All":
            d = d[d["term"] == term]

        if d.empty:
            print("No data available for these selections.")
            return

        if colour == "Dataset":
            color_col = "dataset"
            palette = None
        elif colour == "Median MASE":
            color_col = "median_MASE"
            palette = "Plasma"
        else:
            color_col = "share_worse_than_naive"
            palette = "RdYlGn_r"

        hover_cols = [
            c for c in [
                "dataset", "term", "median_MASE",
                "share_worse_than_naive"
            ] if c in d.columns
        ]

        fig = px.scatter(
            d,
            x="emb_x",
            y="emb_y",
            color=color_col,
            hover_data=hover_cols,
            color_continuous_scale=palette,
            title=f"t-SNE explorer — coloured by {colour}"
        )

        fig.update_layout(
            height=650,
            template="plotly_white"
        )

        fig.show()

    # Connect controls
    tsne_dashboard = widgets.interactive_output(
        update_tsne,
        {
            "dataset": tsne_dataset,
            "term": tsne_term,
            "colour": tsne_colour
        }
    )

    controls = widgets.HBox([
        tsne_dataset,
        tsne_term,
        tsne_colour
    ])

    display(widgets.VBox([
        controls,
        tsne_dashboard
    ]))

    return tsne_dashboard