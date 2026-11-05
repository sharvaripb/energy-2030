from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DEMAND_DAMPING,
    DEMAND_TREND_YEARS,
    FOSSIL_DOMINANT_THRESHOLD,
    RISING_PACE,
    SHARE_TREND_YEARS,
    TARGET,
    TRANSITIONING_PACE,
)

MIN_POINTS = 5


def damped_growth(years: np.ndarray, values: np.ndarray, target_year: int,
                  n_years: int = DEMAND_TREND_YEARS, damping: float = DEMAND_DAMPING) -> float | None:
    years, values = years[-n_years:], values[-n_years:]
    if len(years) < MIN_POINTS or (values <= 0).any():
        return None
    slope = np.polyfit(years, np.log(values), 1)[0]
    horizon = int(round(target_year - years[-1]))
    if horizon <= 0:
        return float(values[-1])
    cumulative = slope * sum(damping ** k for k in range(1, horizon + 1))
    return float(values[-1] * np.exp(cumulative))


def linear_trend(years: np.ndarray, values: np.ndarray, target_year: int,
                 n_years: int, lower: float = 0.0, upper: float | None = None) -> float | None:
    years, values = years[-n_years:], values[-n_years:]
    if len(years) < MIN_POINTS:
        return None
    slope, intercept = np.polyfit(years, values, 1)
    return float(np.clip(slope * target_year + intercept, lower, upper if upper is not None else np.inf))


def share_slope(years: np.ndarray, values: np.ndarray, n_years: int = SHARE_TREND_YEARS) -> float | None:
    years, values = years[-n_years:], values[-n_years:]
    if len(years) < MIN_POINTS:
        return None
    return float(np.polyfit(years, values, 1)[0])


def _series(df: pd.DataFrame, country: str, column: str, up_to: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    g = df.loc[df["country"] == country, ["year", column]].dropna()
    if up_to is not None:
        g = g[g["year"] <= up_to]
    g = g.sort_values("year")
    return g["year"].to_numpy(dtype=float), g[column].to_numpy(dtype=float)


def backtest(df: pd.DataFrame, horizon: int, origins: range | None = None) -> pd.DataFrame:
    last = int(df["year"].max())
    origins = origins or range(2005, last - horizon + 1)
    rows = []
    for country in df["country"].unique():
        ey, ev = _series(df, country, TARGET)
        fy, fv = _series(df, country, "fossil_share_energy")
        for origin in origins:
            target = origin + horizon
            if target in ey and origin in ey:
                actual = ev[ey == target][0]
                hist = ey <= origin
                if actual > 0 and hist.sum() >= MIN_POINTS:
                    preds = {
                        "Damped growth trend": damped_growth(ey[hist], ev[hist], target),
                        "Straight-line trend": linear_trend(ey[hist], ev[hist], target, DEMAND_TREND_YEARS),
                        "No change (last value)": float(ev[hist][-1]),
                    }
                    for method, pred in preds.items():
                        if pred is not None:
                            rows.append(dict(country=country, origin=origin, quantity="energy",
                                             method=method, predicted=pred, actual=actual,
                                             error=pred / actual - 1))
            if target in fy and origin in fy:
                actual = fv[fy == target][0]
                hist = fy <= origin
                if hist.sum() >= MIN_POINTS:
                    preds = {
                        "Straight-line trend": linear_trend(fy[hist], fv[hist], target, SHARE_TREND_YEARS, 0, 100),
                        "No change (last value)": float(fv[hist][-1]),
                    }
                    group = pace_group(share_slope(fy[hist], fv[hist]))
                    for method, pred in preds.items():
                        if pred is not None:
                            rows.append(dict(country=country, origin=origin, quantity="fossil_share",
                                             method=method, predicted=pred, actual=actual,
                                             error=pred - actual, pace_group=group))
    return pd.DataFrame(rows)


def error_band(bt: pd.DataFrame, quantity: str, method: str, low: float = 0.1, high: float = 0.9) -> tuple[float, float]:
    e = bt.loc[(bt["quantity"] == quantity) & (bt["method"] == method), "error"]
    return float(e.quantile(low)), float(e.quantile(high))


def method_summary(bt: pd.DataFrame) -> pd.DataFrame:
    bt = bt.assign(abs_error=bt["error"].abs())
    overall = (
        bt.groupby(["quantity", "method"], as_index=False)
        .agg(median_abs_error=("abs_error", "median"), cases=("abs_error", "size"))
        .assign(segment="All countries")
    )
    fossil = bt[bt["quantity"] == "fossil_share"].copy()
    fossil["segment"] = fossil["pace_group"].where(fossil["pace_group"] == "Transitioning", "Flat or rising")
    split = (
        fossil.groupby(["quantity", "method", "segment"], as_index=False)
        .agg(median_abs_error=("abs_error", "median"), cases=("abs_error", "size"))
    )
    return pd.concat([overall, split], ignore_index=True)


def bias(bt: pd.DataFrame, quantity: str, method: str) -> float:
    return float(bt.loc[(bt["quantity"] == quantity) & (bt["method"] == method), "error"].mean())


def crossing_year(latest_year: float, latest_value: float, slope: float | None,
                  threshold: float = FOSSIL_DOMINANT_THRESHOLD, cap: int = 2050) -> str:
    if latest_value <= threshold:
        return "Already below"
    if slope is None or slope >= 0:
        return f"Not on current trend"
    year = latest_year + (threshold - latest_value) / slope
    return f"~{int(np.ceil(year))}" if year <= cap else f"After {cap}"


def pace_group(slope: float | None) -> str:
    if slope is None:
        return "Insufficient data"
    if slope <= TRANSITIONING_PACE:
        return "Transitioning"
    if slope >= RISING_PACE:
        return "Rising"
    return "Flat"


def transition_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country in sorted(df["country"].unique()):
        y, v = _series(df, country, "fossil_share_energy")
        if len(y) < MIN_POINTS or y[-1] < df["year"].max() - 2:
            continue
        slope = share_slope(y, v)
        if slope is None:
            continue
        rows.append(dict(country=country, latest_year=int(y[-1]), fossil_share=float(v[-1]),
                         pace=slope, group=pace_group(slope),
                         crossing_70=crossing_year(y[-1], v[-1], slope)))
    return pd.DataFrame(rows).sort_values("pace").reset_index(drop=True)
