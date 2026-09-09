from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES


def _linear_project(group: pd.DataFrame, column: str, target_year: int = 2030) -> float | None:
    clean = group[["year", column]].dropna().tail(6)
    if len(clean) < 3:
        return None

    x = clean["year"].to_numpy(dtype=float)
    y = clean[column].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    value = slope * target_year + intercept

    if column in {"fossil_share_energy", "renewables_share_energy"}:
        value = float(np.clip(value, 0, 100))
    else:
        value = float(max(value, 0))
    return value


def build_2030_projection(df: pd.DataFrame, country: str) -> dict[str, float] | None:
    group = df[df["country"] == country].sort_values("year")
    if group.empty:
        return None

    projected: dict[str, float] = {}
    for feature in MODEL_FEATURES:
        value = _linear_project(group, feature)
        if value is None:
            return None
        projected[feature] = value

    fossil_share = _linear_project(group, "fossil_share_energy")
    renewable_share = _linear_project(group, "renewables_share_energy")
    if fossil_share is None or renewable_share is None:
        return None

    projected["fossil_share_energy"] = fossil_share
    projected["renewables_share_energy"] = renewable_share
    projected["other_share_energy"] = max(0.0, 100.0 - fossil_share - renewable_share)
    return projected
