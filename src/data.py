from __future__ import annotations

import numpy as np
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
        raw["country"].isin(ASIAN_COUNTRIES) & raw["year"].between(1995, 2024),  # update : fixed data window so trends can be backtested
        REQUIRED_COLUMNS,
    ].copy()

    df["fossil_dominant"] = df["fossil_share_energy"] > FOSSIL_DOMINANT_THRESHOLD
    return add_derived_columns(df)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    pos = lambda s: s.where(s > 0)
    df["log_gdp_pc"] = np.log(pos(df["gdp"]) / pos(df["population"]))
    df["log_population"] = np.log(pos(df["population"]))
    df["log_energy_pc"] = np.log(pos(df[TARGET]) * 1e9 / pos(df["population"]))
    for col, out in [("fossil_share_energy", "fossil_change_5y"), ("renewables_share_energy", "renewables_change_5y")]:
        lagged = df[["country", "year", col]].copy()
        lagged["year"] = lagged["year"] + 5
        df = df.merge(lagged.rename(columns={col: f"_{out}_lag"}), on=["country", "year"], how="left")
        df[out] = df[col] - df[f"_{out}_lag"]
        df = df.drop(columns=f"_{out}_lag")
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


def latest_year(df: pd.DataFrame, column: str) -> int:
    return int(df.dropna(subset=[column])["year"].max())
