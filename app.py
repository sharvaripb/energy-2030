from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.charts import actual_vs_predicted, cluster_map, feature_importance, projection_mix_chart
from src.config import APP_SUBTITLE, APP_TITLE, COLORS, DATA_SOURCE_URL, GITHUB_URL, MODEL_FEATURES
from src.data import latest_complete_rows, load_energy_data
from src.modeling import predict_scenario, train_models
from src.projections import build_2030_projection
from src.ui import hero_art, load_css, result_block, section_header, stat_card


st.set_page_config(
    page_title="ENERGY / 2030",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_css()

st.markdown(
    f"""
    <div class="brand-row">
      <div class="brand-name">{APP_TITLE}</div>
      <div class="brand-tag">{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading the energy dataset and preparing the models..."):
        energy_df = load_energy_data()
        bundle = train_models(energy_df)
except Exception as exc:
    st.error(
        "The live OWID dataset could not be loaded. Check your internet connection and try again. "
        f"Technical detail: {exc}"
    )
    st.stop()

nav_col, source_col = st.columns([5, 1])
with nav_col:
    section = st.radio(
        "Section",
        ["Overview", "Predict", "Explore", "2030", "Model"],
        horizontal=True,
        label_visibility="collapsed",
    )
with source_col:
    st.link_button("SOURCE", GITHUB_URL, use_container_width=True)

if section == "Overview":
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="hero-kicker">ASIAN ENERGY TRANSITION</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">ENERGY /<br>2030</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-subtitle">Machine learning for energy demand and transition.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-copy">My updated version of a university ML project that models primary energy consumption, fossil dominance, country clusters and unusual regional energy patterns.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(f'<div class="hero-art">{hero_art()}</div>', unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("Current test R²", f"{bundle.regression_r2:.3f}", "lime")
    with c2:
        stat_card("Region", "ASIA", "pale_blue")
    with c3:
        stat_card("Projection", "2030", "purple")

    st.markdown('<div class="section-band"></div>', unsafe_allow_html=True)
    section_header(
        "PROJECT SCOPE",
        "From economic indicators to energy behaviour.",
        "Investigated GDP, population, energy mix, classification, clustering and anomaly detection.",
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown("### Predict")
        st.write("Estimate primary energy consumption from four model inputs.")
    with b:
        st.markdown("### Explore")
        st.write("Inspect country clusters and unusual energy patterns across Asia.")
    with c:
        st.markdown("### Project")
        st.write("Extrapolate recent trends to 2030 and score fossil dominance with the classifier.")

elif section == "Predict":
    section_header(
        "SCENARIO BUILDER",
        "What if?",
        "Adjust the inputs to create an energy scenario. The prediction updates using the same feature set as the Random Forest models.",
    )

    latest = latest_complete_rows(energy_df)
    default_row = latest.iloc[len(latest) // 2] if not latest.empty else None

    input_col, output_col = st.columns([1.05, 0.95], gap="large")
    with input_col:
        gdp = st.number_input(
            "GDP, current USD",
            min_value=0.0,
            value=float(default_row["gdp"]) if default_row is not None else 1_000_000_000_000.0,
            step=10_000_000_000.0,
            format="%.0f",
        )
        population = st.number_input(
            "Population",
            min_value=0.0,
            value=float(default_row["population"]) if default_row is not None else 100_000_000.0,
            step=1_000_000.0,
            format="%.0f",
        )
        fossil_share = st.slider(
            "Fossil fuel share of energy (%)",
            0.0,
            100.0,
            float(default_row["fossil_share_energy"]) if default_row is not None else 70.0,
            0.5,
        )
        renewables = st.number_input(
            "Renewable energy consumption (TWh)",
            min_value=0.0,
            value=float(default_row["renewables_consumption"]) if default_row is not None else 250.0,
            step=10.0,
        )

    values = {
        "gdp": gdp,
        "population": population,
        "fossil_share_energy": fossil_share,
        "renewables_consumption": renewables,
    }
    energy_pred, fossil_prob = predict_scenario(bundle, values)

    with output_col:
        result_block(
            "Predicted primary energy consumption",
            f"{energy_pred:,.1f} TWh",
            "Random Forest regression estimate",
            "lime",
        )
        result_block(
            "Fossil dominance probability",
            f"{fossil_prob * 100:.0f}%",
            "Probability that fossil share exceeds the 70% project threshold",
            "pale_blue",
        )
        st.markdown(
            '<div class="small-note">Scenario inputs outside the historical training range come from exploratory model behaviour, not policy forecasts.</div>',
            unsafe_allow_html=True,
        )

elif section == "Explore":
    section_header(
        "REGIONAL PATTERNS",
        "Energy landscape.",
        "The latest row per country is shown using the project cluster model. Isolation Forest flags observations that differ from the regional pattern.",
    )
    st.plotly_chart(cluster_map(bundle.modeled_data), use_container_width=True)

    anomalies = (
        bundle.modeled_data[bundle.modeled_data["anomaly"]]
        .sort_values("year")
        .groupby("country", as_index=False)
        .tail(1)
        [["country", "year", "cluster", "primary_energy_consumption", "fossil_share_energy"]]
        .sort_values("primary_energy_consumption", ascending=False)
    )
    with st.expander("Show latest flagged anomalies"):
        st.dataframe(anomalies, use_container_width=True, hide_index=True)

elif section == "2030":
    section_header(
        "TREND PROJECTION",
        "Asia in 2030.",
        "Choose a country to extend its recent feature trends to 2030. The projected inputs are then passed through the fossil dominance classifier to get the trend scenario",
    )

    countries = sorted(energy_df["country"].dropna().unique().tolist())
    country = st.selectbox("Country", countries, index=countries.index("Vietnam") if "Vietnam" in countries else 0)
    projection = build_2030_projection(energy_df, country)

    if projection is None:
        st.warning("This country does not have enough recent complete observations for the 2030 projection.")
    else:
        energy_pred, fossil_prob = predict_scenario(bundle, projection)
        left, right = st.columns([0.9, 1.1], gap="large")
        with left:
            result_block("2030 projected energy demand", f"{energy_pred:,.1f} TWh", country, "lime")
            result_block("Fossil dominance probability", f"{fossil_prob * 100:.0f}%", country, "pale_blue")
        with right:
            st.plotly_chart(projection_mix_chart(projection), use_container_width=True)

        with st.expander("Show projected inputs"):
            table = pd.DataFrame(
                {
                    "Variable": [
                        "GDP",
                        "Population",
                        "Fossil share",
                        "Renewable consumption",
                        "Renewable share",
                    ],
                    "2030 projection": [
                        projection["gdp"],
                        projection["population"],
                        projection["fossil_share_energy"],
                        projection["renewables_consumption"],
                        projection["renewables_share_energy"],
                    ],
                }
            )
            st.dataframe(table, hide_index=True, use_container_width=True)

elif section == "Model":
    section_header(
        "UNDER THE HOOD",
        "Model evidence.",
        "The app retrains on the current public OWID Energy dataset so that the result is reproducible instead of depending on a local path.",
    )

    m1, m2 = st.columns(2)
    with m1:
        result_block("Regression test R²", f"{bundle.regression_r2:.3f}", "Random Forest Regressor", "lime")
    with m2:
        result_block(
            "Classification accuracy",
            f"{bundle.classification_accuracy * 100:.1f}%",
            "Random Forest Classifier",
            "pale_blue",
        )

    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.plotly_chart(actual_vs_predicted(bundle.test_actual, bundle.test_predicted), use_container_width=True)
    with p2:
        st.plotly_chart(feature_importance(bundle.feature_importance), use_container_width=True)

    st.markdown(
        f"Data source: [{DATA_SOURCE_URL}]({DATA_SOURCE_URL}). The original coursework reported a test R² of approximately 0.954. The number shown above is recomputed from the current dataset snapshot each time the cached model is refreshed."
    )
