from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASSIFIER_FEATURES,
    CLASSIFIER_HORIZON,
    DEMAND_FEATURES,
    TARGET,
    TRANSITION_DROP_PTS,
)
from src.data import model_frame
from src.forecasting import backtest, bias, error_band, method_summary

N_FOLDS = 5


@dataclass
class ModelBundle:
    elasticity: float
    elasticity_ci: tuple[float, float]
    cross_country_elasticity: float
    demand_model: LinearRegression
    demand_heldout_error: float
    demand_rf_heldout_error: float
    classifier: object
    classifier_balanced_accuracy: float
    baseline_balanced_accuracy: float
    classifier_confusion: np.ndarray
    classifier_positive_rate: float
    classifier_rows: int
    classifier_coefficients: pd.DataFrame
    backtest_6y: pd.DataFrame
    method_errors: pd.DataFrame
    energy_band: tuple[float, float]
    fossil_band: tuple[float, float]
    fossil_bias: float
    clusterer: KMeans
    anomaly_model: IsolationForest
    modeled_data: pd.DataFrame


def _demand_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=[*DEMAND_FEATURES, "log_energy_pc"]).reset_index(drop=True)


def classifier_frame(df: pd.DataFrame) -> pd.DataFrame:
    future = df[["country", "year", "fossil_share_energy"]].copy()
    future["year"] = future["year"] - CLASSIFIER_HORIZON
    future = future.rename(columns={"fossil_share_energy": "fossil_share_future"})
    out = df.merge(future, on=["country", "year"], how="inner")
    out["label"] = (out["fossil_share_future"] - out["fossil_share_energy"] <= -TRANSITION_DROP_PTS).astype(int)
    return out.dropna(subset=[*CLASSIFIER_FEATURES, "label"]).reset_index(drop=True)


def within_country_elasticity(demand: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> tuple[float, tuple[float, float]]:
    d = demand[["country", "log_gdp_pc", "log_energy_pc"]].copy()
    for col in ["log_gdp_pc", "log_energy_pc"]:
        d[col] = d[col] - d.groupby("country")[col].transform("mean")
    per_country = d.assign(xy=d["log_gdp_pc"] * d["log_energy_pc"], xx=d["log_gdp_pc"] ** 2).groupby("country")[["xy", "xx"]].sum()
    beta = float(per_country["xy"].sum() / per_country["xx"].sum())
    rng = np.random.default_rng(seed)
    idx = np.arange(len(per_country))
    boots = []
    for _ in range(n_boot):
        pick = per_country.iloc[rng.choice(idx, len(idx))]
        boots.append(pick["xy"].sum() / pick["xx"].sum())
    return beta, (float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95)))


def _make_classifier():
    # update : fixed inflated probabilities by removing class_weight="balanced"
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame) -> ModelBundle:
    demand = _demand_frame(df)
    if demand["country"].nunique() < 10:
        raise ValueError("Not enough complete OWID rows to train the demand model.")
    X, y, groups = demand[DEMAND_FEATURES], demand["log_energy_pc"], demand["country"]

    lin_err, rf_err = [], []
    # update : fixed leakage from the random train/test split by holding out whole countries
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        lin = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X.iloc[tr], y.iloc[tr])
        p_lin, p_rf = lin.predict(X.iloc[te]), rf.predict(X.iloc[te])
        lin_err += list(np.exp(p_lin - y.iloc[te]) - 1)
        rf_err += list(np.exp(p_rf - y.iloc[te]) - 1)
    demand_model = LinearRegression().fit(X, y)
    elasticity, elasticity_ci = within_country_elasticity(demand)
    cross_elasticity = float(LinearRegression().fit(demand[["log_gdp_pc"]], y).coef_[0])

    # update : fixed single-class fossil-dominance label by predicting a future 3-pt drop instead
    cls = classifier_frame(df)
    Xc, yc, gc = cls[CLASSIFIER_FEATURES], cls["label"], cls["country"]
    preds = np.zeros(len(cls), dtype=int)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(Xc, yc, gc):
        model = _make_classifier().fit(Xc.iloc[tr], yc.iloc[tr])
        threshold = yc.iloc[tr].mean()
        preds[te] = (model.predict_proba(Xc.iloc[te])[:, 1] >= threshold).astype(int)
    momentum = (cls["fossil_change_5y"] <= -TRANSITION_DROP_PTS).astype(int)
    classifier = _make_classifier().fit(Xc, yc)
    coefs = pd.DataFrame({
        "feature": CLASSIFIER_FEATURES,
        "weight": classifier[-1].coef_[0],
    }).sort_values("weight")

    # update : fixed 2030 forecasts to be scored against what actually happened 6 years later
    bt = backtest(df, horizon=6)

    data = model_frame(df)

    # Cluster on log-scaled magnitude features and raw percentage share.
    cluster_features = pd.DataFrame(
        {
            "log_gdp": np.log1p(data["gdp"].clip(lower=0)),
            "log_population": np.log1p(data["population"].clip(lower=0)),
            "fossil_share_energy": data["fossil_share_energy"],
            "log_renewables": np.log1p(data["renewables_consumption"].clip(lower=0)),
        }
    )
    cluster_features = (cluster_features - cluster_features.mean()) / cluster_features.std(ddof=0)

    clusterer = KMeans(n_clusters=4, n_init=20, random_state=42)
    data = data.copy()
    data["cluster"] = clusterer.fit_predict(cluster_features)

    anomaly_model = IsolationForest(contamination=0.08, random_state=42)
    data["anomaly"] = anomaly_model.fit_predict(cluster_features) == -1

    return ModelBundle(
        elasticity=elasticity,
        elasticity_ci=elasticity_ci,
        cross_country_elasticity=cross_elasticity,
        demand_model=demand_model,
        demand_heldout_error=float(np.median(np.abs(lin_err))),
        demand_rf_heldout_error=float(np.median(np.abs(rf_err))),
        classifier=classifier,
        classifier_balanced_accuracy=float(balanced_accuracy_score(yc, preds)),
        baseline_balanced_accuracy=float(balanced_accuracy_score(yc, momentum)),
        classifier_confusion=confusion_matrix(yc, preds),
        classifier_positive_rate=float(yc.mean()),
        classifier_rows=int(len(cls)),
        classifier_coefficients=coefs,
        backtest_6y=bt,
        method_errors=method_summary(bt),
        energy_band=error_band(bt, "energy", "Damped growth trend"),
        fossil_band=error_band(bt, "fossil_share", "Straight-line trend"),
        fossil_bias=bias(bt, "fossil_share", "Straight-line trend"),
        clusterer=clusterer,
        anomaly_model=anomaly_model,
        modeled_data=data,
    )


def anchor_row(df: pd.DataFrame, country: str) -> pd.Series | None:
    rows = df[(df["country"] == country)].dropna(subset=["gdp", "population", TARGET])
    if rows.empty:
        return None
    return rows.sort_values("year").iloc[-1]


# update : fixed what-if demand to start from the country's own measured energy use
def predict_demand(bundle: ModelBundle, base: pd.Series, gdp: float, population: float) -> tuple[float, float, float]:
    if gdp <= 0 or population <= 0:
        return 0.0, 0.0, 0.0
    pop_ratio = population / base["population"]
    gdp_pc_ratio = (gdp / population) / (base["gdp"] / base["population"])

    def energy(beta: float) -> float:
        return float(base[TARGET] * pop_ratio * gdp_pc_ratio ** beta)

    lo_b, hi_b = bundle.elasticity_ci
    a, b = energy(lo_b), energy(hi_b)
    return energy(bundle.elasticity), min(a, b), max(a, b)


def transition_probability(bundle: ModelBundle, features: dict[str, float]) -> float | None:
    if any(features.get(f) is None or pd.isna(features.get(f)) for f in CLASSIFIER_FEATURES):
        return None
    frame = pd.DataFrame([[features[f] for f in CLASSIFIER_FEATURES]], columns=CLASSIFIER_FEATURES)
    return float(bundle.classifier.predict_proba(frame)[0, 1])
