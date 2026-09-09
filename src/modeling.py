from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from src.config import MODEL_FEATURES, TARGET
from src.data import model_frame


@dataclass
class ModelBundle:
    regressor: RandomForestRegressor
    classifier: RandomForestClassifier
    clusterer: KMeans
    anomaly_model: IsolationForest
    regression_r2: float
    classification_accuracy: float
    test_actual: np.ndarray
    test_predicted: np.ndarray
    feature_importance: pd.DataFrame
    modeled_data: pd.DataFrame


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame) -> ModelBundle:
    data = model_frame(df)
    if len(data) < 80:
        raise ValueError("Not enough complete OWID rows to train the project models.")

    X = data[MODEL_FEATURES]
    y_reg = data[TARGET]
    y_cls = data["fossil_dominant"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    regressor = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_test)
    reg_r2 = r2_score(y_test, pred)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X,
        y_cls,
        test_size=0.2,
        random_state=42,
        stratify=y_cls if y_cls.nunique() > 1 else None,
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(Xc_train, yc_train)
    cls_pred = classifier.predict(Xc_test)
    cls_acc = accuracy_score(yc_test, cls_pred)

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

    fi = pd.DataFrame(
        {"feature": MODEL_FEATURES, "importance": regressor.feature_importances_}
    ).sort_values("importance", ascending=True)

    return ModelBundle(
        regressor=regressor,
        classifier=classifier,
        clusterer=clusterer,
        anomaly_model=anomaly_model,
        regression_r2=float(reg_r2),
        classification_accuracy=float(cls_acc),
        test_actual=y_test.to_numpy(),
        test_predicted=pred,
        feature_importance=fi,
        modeled_data=data,
    )


def predict_scenario(bundle: ModelBundle, values: dict[str, float]) -> tuple[float, float]:
    frame = pd.DataFrame([[values[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
    energy = float(bundle.regressor.predict(frame)[0])

    if hasattr(bundle.classifier, "predict_proba") and 1 in bundle.classifier.classes_:
        class_index = list(bundle.classifier.classes_).index(1)
        probability = float(bundle.classifier.predict_proba(frame)[0, class_index])
    else:
        probability = float(bundle.classifier.predict(frame)[0])

    return energy, probability
