from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from src.config import ASIAN_COUNTRIES, DATA_URL, FOSSIL_DOMINANT_THRESHOLD, MODEL_FEATURES, TARGET


REQUIRED_COLUMNS = [
    "country",
    "year",
    *MODEL_FEATURES,
    TARGET,
    "renewables_share_energy",
]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_energy_data() -> pd.DataFrame:
    """Load the current OWID Energy dataset and return the project subset."""
    response = requests.get(DATA_URL, timeout=45)
    response.raise_for_status()

    from io import StringIO

    raw = pd.read_csv(StringIO(response.text), usecols=lambda c: c in REQUIRED_COLUMNS)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"OWID dataset is missing required columns: {missing}")

    df = raw.loc[
        raw["country"].isin(ASIAN_COUNTRIES) & raw["year"].between(2012, 2024),
        REQUIRED_COLUMNS,
    ].copy()

    df["fossil_dominant"] = df["fossil_share_energy"] > FOSSIL_DOMINANT_THRESHOLD
    return df.sort_values(["country", "year"]).reset_index(drop=True)


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows complete enough for supervised modeling."""
    cols = ["country", "year", *MODEL_FEATURES, TARGET, "fossil_dominant"]
    return df[cols].dropna().reset_index(drop=True)


def latest_complete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent complete feature row for each country."""
    complete = df.dropna(subset=MODEL_FEATURES).copy()
    if complete.empty:
        return complete
    idx = complete.groupby("country")["year"].idxmax()
    return complete.loc[idx].sort_values("country").reset_index(drop=True)
