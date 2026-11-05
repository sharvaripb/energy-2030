from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import COLORS


PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["cream"],
    plot_bgcolor=COLORS["cream"],
    font=dict(color=COLORS["ink"], family="Arial, Helvetica, sans-serif"),
    margin=dict(l=20, r=20, t=50, b=20),
)


def actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            marker=dict(size=8, color=COLORS["purple"], opacity=0.78),
            hovertemplate="Actual: %{x:,.1f}<br>Predicted: %{y:,.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line=dict(color=COLORS["red"], width=3),
            hoverinfo="skip",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="Actual vs predicted")
    fig.update_xaxes(title="Actual primary energy consumption", gridcolor="#C9CABD")
    fig.update_yaxes(title="Predicted primary energy consumption", gridcolor="#C9CABD")
    return fig


def feature_importance(fi: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        fi,
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[COLORS["lime"]],
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="Feature importance")
    fig.update_xaxes(title=None, gridcolor="#C9CABD")
    fig.update_yaxes(title=None)
    return fig


def cluster_map(data: pd.DataFrame) -> go.Figure:
    latest = data.sort_values("year").groupby("country", as_index=False).tail(1).copy()
    latest["cluster_label"] = "Cluster " + latest["cluster"].astype(str)
    latest["status"] = np.where(latest["anomaly"], "Anomaly", "Typical")

    palette = [COLORS["purple"], COLORS["lime"], COLORS["red"], COLORS["deep_teal"]]
    fig = px.scatter_geo(
        latest,
        locations="country",
        locationmode="country names",
        color="cluster_label",
        size="primary_energy_consumption",
        hover_name="country",
        hover_data={
            "year": True,
            "gdp": ":,.0f",
            "population": ":,.0f",
            "fossil_share_energy": ":.1f",
            "renewables_consumption": ":.1f",
            "primary_energy_consumption": ":.1f",
            "cluster_label": True,
            "status": True,
        },
        color_discrete_sequence=palette,
        projection="natural earth",
        scope="asia",
        size_max=36,
    )
    fig.update_geos(
        bgcolor=COLORS["cream"],
        landcolor="#D7D8C7",
        oceancolor=COLORS["pale_blue"],
        lakecolor=COLORS["pale_blue"],
        showocean=True,
        showcountries=True,
        countrycolor=COLORS["cream"],
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Latest modeled energy landscape")
    fig.update_layout(legend_title_text="")
    return fig


def projection_mix_chart(projection: dict[str, float]) -> go.Figure:
    fossil = projection["fossil_share_energy"]
    renewable = projection["renewables_share_energy"]
    other = max(0.0, projection["other_share_energy"])
    total = fossil + renewable + other
    if total <= 0:
        total = 100.0
    values = [100 * fossil / total, 100 * renewable / total, 100 * other / total]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=["2030 energy mix"],
            orientation="h",
            marker_color=[COLORS["ink"], COLORS["lime"], COLORS["pale_blue"]],
            customdata=[["Fossil", values[0]], ["Renewable", values[1]], ["Other", values[2]]],
        )
    )
    # Build as three stacked traces to keep labels controlled.
    fig = go.Figure()
    for name, val, color in zip(
        ["Fossil", "Renewable", "Other"],
        values,
        [COLORS["ink"], COLORS["lime"], COLORS["pale_blue"]],
    ):
        fig.add_trace(
            go.Bar(
                x=[val],
                y=["2030"],
                name=name,
                orientation="h",
                marker_color=color,
                hovertemplate=f"{name}: %{{x:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", title="2030 trend projection")
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title=None, gridcolor="#C9CABD")
    fig.update_yaxes(title=None)
    fig.update_layout(legend_orientation="h", legend_y=-0.2, legend_title_text="")
    return fig


# update : fixed charts to show transition pace and backtested errors
GROUP_COLORS = {
    "Transitioning": COLORS["lime"],
    "Flat": COLORS["blue_grey"],
    "Rising": COLORS["red"],
}


def transition_ranking(table: pd.DataFrame, highlight: str | None = None) -> go.Figure:
    t = table.sort_values("pace", ascending=False)
    colors = [GROUP_COLORS.get(g, COLORS["blue_grey"]) for g in t["group"]]
    line = [3 if c == highlight else 0 for c in t["country"]]
    fig = go.Figure(
        go.Bar(
            x=t["pace"],
            y=t["country"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=COLORS["ink"], width=line)),
            customdata=t[["fossil_share", "crossing_70", "group"]],
            hovertemplate=(
                "<b>%{y}</b><br>%{x:+.2f} pts per year<br>"
                "Fossil share now: %{customdata[0]:.1f}%<br>"
                "Below 70%: %{customdata[1]}<br>%{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="Pace of transition, last 10 years",
                      height=max(420, 22 * len(t)))
    fig.update_xaxes(title="Change in fossil share of energy (points per year)", gridcolor="#C9CABD",
                     zeroline=True, zerolinecolor=COLORS["ink"])
    fig.update_yaxes(title=None)
    return fig


def backtest_scatter(bt: pd.DataFrame, method: str = "Damped growth trend") -> go.Figure:
    d = bt[(bt["quantity"] == "energy") & (bt["method"] == method)]
    low = float(min(d["actual"].min(), d["predicted"].min()))
    high = float(max(d["actual"].max(), d["predicted"].max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["actual"],
            y=d["predicted"],
            mode="markers",
            marker=dict(size=7, color=COLORS["purple"], opacity=0.6),
            customdata=d[["country", "origin"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b>, forecast made in %{customdata[1]}<br>"
                "Actual: %{x:,.0f} TWh<br>Forecast: %{y:,.0f} TWh<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(x=[low, high], y=[low, high], mode="lines",
                   line=dict(color=COLORS["red"], width=3), hoverinfo="skip")
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="Backtest: 6-year-ahead demand forecasts")
    fig.update_xaxes(title="Actual primary energy (TWh, log scale)", type="log", gridcolor="#C9CABD")
    fig.update_yaxes(title="Forecast made 6 years earlier (TWh, log scale)", type="log", gridcolor="#C9CABD")
    return fig


def method_comparison(summary: pd.DataFrame) -> go.Figure:
    d = summary[(summary["quantity"] == "energy") & (summary["segment"] == "All countries")]
    d = d.sort_values("median_abs_error", ascending=False)
    best = d["median_abs_error"].min()
    fig = go.Figure(
        go.Bar(
            x=d["median_abs_error"] * 100,
            y=d["method"],
            orientation="h",
            marker_color=[COLORS["lime"] if v == best else COLORS["blue_grey"] for v in d["median_abs_error"]],
            text=[f"{v * 100:.1f}%" for v in d["median_abs_error"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}% median error<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="Which demand forecast holds up? (lower is better)")
    fig.update_xaxes(title="Median error, 6 years ahead (%)", gridcolor="#C9CABD",
                     range=[0, float(d["median_abs_error"].max()) * 100 * 1.3])
    fig.update_yaxes(title=None)
    return fig


def classifier_weights(coefs: pd.DataFrame) -> go.Figure:
    labels = {
        "fossil_share_energy": "Fossil share now",
        "fossil_change_5y": "Fossil share change, past 5 yrs",
        "renewables_share_energy": "Renewable share now",
        "renewables_change_5y": "Renewable share change, past 5 yrs",
    }
    d = coefs.assign(label=coefs["feature"].map(labels))
    fig = go.Figure(
        go.Bar(
            x=d["weight"],
            y=d["label"],
            orientation="h",
            marker_color=[COLORS["lime"] if w > 0 else COLORS["purple"] for w in d["weight"]],
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="What drives the transition classifier")
    fig.update_xaxes(title="Standardised weight (positive = more likely to drop ≥3 pts)", gridcolor="#C9CABD",
                     zeroline=True, zerolinecolor=COLORS["ink"])
    fig.update_yaxes(title=None)
    return fig
