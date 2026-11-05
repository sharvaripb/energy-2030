from __future__ import annotations

import pandas as pd

from src.config import PROJECTION_YEAR, SHARE_TREND_YEARS, TARGET
from src.forecasting import _series, crossing_year, damped_growth, linear_trend, pace_group, share_slope
from src.modeling import ModelBundle, transition_probability


# update : fixed 2030 projection to use backtested trends with ranges instead of the classifier
def build_2030_projection(df: pd.DataFrame, country: str, bundle: ModelBundle) -> dict | None:
    ey, ev = _series(df, country, TARGET)
    fy, fv = _series(df, country, "fossil_share_energy")
    ry, rv = _series(df, country, "renewables_share_energy")
    if len(ey) < 5 or len(fy) < 5:
        return None

    energy = damped_growth(ey, ev, PROJECTION_YEAR)
    fossil = linear_trend(fy, fv, PROJECTION_YEAR, SHARE_TREND_YEARS, 0, 100)
    renew = linear_trend(ry, rv, PROJECTION_YEAR, SHARE_TREND_YEARS, 0, 100) if len(ry) >= 5 else None
    if energy is None or fossil is None:
        return None
    if renew is None:
        renew = 0.0
    if fossil + renew > 100:
        renew = 100 - fossil

    slope = share_slope(fy, fv)
    e_lo, e_hi = bundle.energy_band
    f_lo, f_hi = bundle.fossil_band

    latest = df[(df["country"] == country) & (df["year"] == fy[-1])].iloc[0]
    prob = transition_probability(bundle, latest.to_dict())

    return {
        "energy_latest_year": int(ey[-1]),
        "energy_latest": float(ev[-1]),
        "energy_2030": energy,
        "energy_low": energy / (1 + e_hi),
        "energy_high": energy / (1 + e_lo),
        "fossil_latest_year": int(fy[-1]),
        "fossil_latest": float(fv[-1]),
        "fossil_share_energy": fossil,
        "fossil_low": max(0.0, fossil - f_hi),
        "fossil_high": min(100.0, fossil - f_lo),
        "fossil_pace": slope,
        "pace_group": pace_group(slope),
        "crossing_70": crossing_year(fy[-1], fv[-1], slope),
        "renewables_share_energy": renew,
        "other_share_energy": max(0.0, 100.0 - fossil - renew),
        "transition_probability": prob,
        "data_is_stale": int(fy[-1]) < int(df["year"].max()) - 2,
    }
