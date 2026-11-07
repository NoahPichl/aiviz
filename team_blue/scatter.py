"""Interactive feature scatter plot colored by series-cube performance metrics."""

from __future__ import annotations

import ipywidgets as w
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display

from team_blue.data_cube import feature_columns

COLOR_OPTIONS = [
    ("MAE", "MAE_mean"),
    ("sMAPE", "sMAPE_mean"),
    ("QR80_rel", "QR80_rel_mean"),
]


def aggregate_datasets(
    df: pd.DataFrame, columns: list[str], by: list[str] | None = None
) -> pd.DataFrame:
    """One point per dataset: the median over all its (filtered) series, freqs and terms.

    ``by`` adds extra grouping keys, e.g. ``["term"]`` when points are colored by term.
    """
    keys = ["dataset", *(by or [])]
    grouped = df.groupby(keys, observed=True)
    agg = grouped[columns].median()
    agg["n_series"] = grouped.size()
    agg["freqs"] = grouped["freq"].agg(lambda v: ", ".join(sorted(set(v))))
    if "term" not in keys:
        order = {"short": 0, "medium": 1, "long": 2}
        agg["term"] = grouped["term"].agg(
            lambda v: ", ".join(sorted(set(v), key=order.get))
        )
    return agg.reset_index()


def _multi_select(label: str, options: list[str]) -> tuple[w.SelectMultiple, w.VBox]:
    select = w.SelectMultiple(
        options=options, value=tuple(options), rows=min(len(options), 8)
    )
    select.layout.width = "220px"
    all_btn = w.Button(description="All", layout=w.Layout(width="70px"))
    none_btn = w.Button(description="None", layout=w.Layout(width="70px"))
    all_btn.on_click(lambda _: setattr(select, "value", tuple(options)))
    none_btn.on_click(lambda _: setattr(select, "value", ()))
    return select, w.VBox([w.Label(label), select, w.HBox([all_btn, none_btn])])


def feature_scatter(series_cube: pd.DataFrame) -> w.VBox:
    """Plot two feat_* columns, colored by a metric's mean over forecast windows.

    Dataset aggregation takes the median of both features and the selected metric.
    """
    df = series_cube.copy()
    for col in ("dataset", "term", "freq"):
        df[col] = df[col].astype(str)
    features = feature_columns(df)
    datasets = sorted(df["dataset"].unique())
    terms = [t for t in ("short", "medium", "long") if t in set(df["term"])]

    dataset_select, dataset_box = _multi_select("Datasets", datasets)
    term_select, term_box = _multi_select("Terms", terms)
    color = w.ToggleButtons(options=COLOR_OPTIONS, description="Color")
    x_axis = w.Dropdown(
        options=features,
        value=features[0],
        description="X-axis",
    )
    y_axis = w.Dropdown(
        options=features,
        value=features[0],
        description="Y-axis",
    )
    aggregate = w.ToggleButtons(
        options=[("Per series", False), ("Aggregate per dataset", True)],
        description="Points",
    )
    log_color = w.Checkbox(value=False, description="Log color")
    status = w.HTML()

    def build() -> go.Figure:
        sub = df[
            df["dataset"].isin(dataset_select.value)
            & df["term"].isin(term_select.value)
        ]
        x, y, c = x_axis.value, y_axis.value, color.value
        columns = list(dict.fromkeys([x, y, c]))
        if aggregate.value:
            sub = aggregate_datasets(sub, columns)
            hover = ["freqs", "n_series"]
        else:
            hover = ["freq", "series_name", "variate_name"]
        sub = sub.dropna(subset=columns)
        color_column = c
        excluded = 0
        colorbar = dict(tickmode="auto", tickvals=None, ticktext=None)
        if log_color.value:
            valid = np.isfinite(sub[c]) & (sub[c] > 0)
            excluded = int((~valid).sum())
            sub = sub.loc[valid].copy()
            color_column = "_log_color"
            sub[color_column] = np.log10(sub[c])
            if not sub.empty:
                ticks = np.unique(
                    np.linspace(sub[color_column].min(), sub[color_column].max(), 5)
                )
                colorbar = dict(
                    tickmode="array",
                    tickvals=ticks,
                    ticktext=[f"{10**value:.3g}" for value in ticks],
                )
        status.value = f"<b>{len(sub)}</b> points"
        if excluded:
            status.value += (
                f" ({excluded} excluded: log color requires positive, finite values)"
            )
        metric_label = {column: label for label, column in COLOR_OPTIONS}[c]

        new = px.scatter(
            sub,
            x=x,
            y=y,
            color=color_column,
            hover_data={
                "dataset": True,
                "term": True,
                **dict.fromkeys(hover, True),
                c: True,
                **({color_column: False} if log_color.value else {}),
            },
            color_continuous_scale="Viridis",
            labels={
                **{column: label for label, column in COLOR_OPTIONS},
                color_column: metric_label
                + (" (log scale)" if log_color.value else ""),
            },
            opacity=0.9 if aggregate.value else 0.6,
            render_mode="svg",
        )
        new.update_layout(
            height=600,
            width=700,
            template="plotly_white",
            xaxis_title=x,
            yaxis_title=y,
            yaxis_type="linear",
        )
        new.update_coloraxes(colorbar=colorbar)
        new.update_traces(marker_size=10 if aggregate.value else 6)
        return new

    out = w.Output()

    def update(*_):
        new = build()
        box.figure = new
        with out:
            out.clear_output(wait=True)
            display(new)

    for widget in (
        dataset_select,
        term_select,
        color,
        x_axis,
        y_axis,
        aggregate,
        log_color,
    ):
        widget.observe(update, names="value")

    controls = w.VBox([x_axis, y_axis, color, aggregate, log_color, status])
    box = w.VBox([w.HBox([dataset_box, term_box, controls]), out])
    update()
    return box
