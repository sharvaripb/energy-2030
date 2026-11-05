from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.charts import (
    backtest_scatter,
    classifier_weights,
    cluster_map,
    method_comparison,
    projection_mix_chart,
    transition_ranking,
)
from src.config import (
    APP_SUBTITLE,
    APP_TITLE,
    CLASSIFIER_HORIZON,
    DATA_SOURCE_URL,
    FOSSIL_DOMINANT_THRESHOLD,
    GITHUB_URL,
    TARGET,
    TRANSITION_DROP_PTS,
)
from src.data import latest_year, load_energy_data
from src.forecasting import transition_table
from src.modeling import anchor_row, predict_demand, train_models
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

# update : fixed missing data-window label
ENERGY_YEAR = latest_year(energy_df, TARGET)
GDP_YEAR = latest_year(energy_df, "gdp")
DATA_NOTE = f"Latest data: energy and energy mix to {ENERGY_YEAR}, GDP to {GDP_YEAR} (Our World in Data)."

errors = bundle.method_errors
def _err(quantity: str, method: str, segment: str = "All countries") -> float:
    row = errors[(errors["quantity"] == quantity) & (errors["method"] == method) & (errors["segment"] == segment)]
    return float(row["median_abs_error"].iloc[0]) if not row.empty else float("nan")

DEMAND_ERR = _err("energy", "Damped growth trend")

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
        # update : fixed misleading test R² by showing backtested forecast error
        stat_card("Backtested 6-yr error", f"±{DEMAND_ERR * 100:.0f}%", "lime")
    with c2:
        stat_card("Region", "ASIA", "pale_blue")
    with c3:
        stat_card("Projection", "2030", "purple")

    st.markdown('<div class="section-band"></div>', unsafe_allow_html=True)
    section_header(
        "PROJECT SCOPE",
        "From economic indicators to energy behaviour.",
        f"Investigated GDP, population, energy mix, classification, clustering and anomaly detection. {DATA_NOTE}",
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
        st.write("Extrapolate recent trends to 2030 and rank countries by how fast they are moving off fossil fuels.")

elif section == "Predict":
    section_header(
        "SCENARIO BUILDER",
        "What if?",
        "Pick a country, then change its GDP or population. The estimate starts from the country's own "
        "measured energy use and applies how energy use has responded to income growth within Asian countries.",
    )

    # update : fixed what-if to start from a chosen country's measured energy use
    anchors = {c: anchor_row(energy_df, c) for c in sorted(energy_df["country"].unique())}
    anchors = {c: r for c, r in anchors.items() if r is not None}
    names = list(anchors)
    country = st.selectbox("Country", names, index=names.index("Vietnam") if "Vietnam" in names else 0)
    base = anchors[country]
    base_year = int(base["year"])

    input_col, output_col = st.columns([1.05, 0.95], gap="large")
    with input_col:
        gdp = st.number_input(
            f"GDP, USD (actual {base_year}: {base['gdp']:,.0f})",
            min_value=0.0,
            value=float(base["gdp"]),
            step=float(base["gdp"]) * 0.05,
            format="%.0f",
            key=f"gdp_{country}",
        )
        population = st.number_input(
            f"Population (actual {base_year}: {base['population']:,.0f})",
            min_value=0.0,
            value=float(base["population"]),
            step=float(base["population"]) * 0.01,
            format="%.0f",
            key=f"pop_{country}",
        )
        base_fossil = float(base["fossil_share_energy"]) if pd.notna(base["fossil_share_energy"]) else 80.0
        fossil_share = st.slider(
            "Fossil fuel share of energy (%)", 0.0, 100.0, base_fossil, 0.5, key=f"fossil_{country}"
        )
        base_ren = float(base["renewables_consumption"]) if pd.notna(base["renewables_consumption"]) else 0.0
        renewables = st.number_input(
            "Renewable energy consumption (TWh)", min_value=0.0, value=base_ren, step=10.0, key=f"ren_{country}"
        )

    energy_pred, energy_low, energy_high = predict_demand(bundle, base, gdp, population)
    fossil_twh = energy_pred * fossil_share / 100
    ren_share = 100 * renewables / energy_pred if energy_pred > 0 else 0.0

    with output_col:
        changed = abs(energy_high - energy_low) > 0.5
        result_block(
            "Predicted primary energy consumption",
            f"{energy_pred:,.0f} TWh",
            (
                f"Likely range {energy_low:,.0f}–{energy_high:,.0f} TWh · actual {base_year}: {base[TARGET]:,.0f} TWh"
                if changed
                else f"Actual {base_year} consumption · change GDP or population to see the effect"
            ),
            "lime",
        )
        result_block(
            "Fossil energy implied",
            f"{fossil_twh:,.0f} TWh",
            f"At {fossil_share:.1f}% fossil share · renewables would cover {ren_share:.0f}% of demand",
            "pale_blue",
        )
        lo_b, hi_b = bundle.elasticity_ci
        st.markdown(
            f'<div class="small-note">Within Asian countries, a 10% rise in GDP per person has come with about '
            f"{bundle.elasticity * 10:.1f}% more energy per person (90% interval {lo_b * 10:.1f}–{hi_b * 10:.1f}%). "
            f"Comparing different countries suggests a much steeper {bundle.cross_country_elasticity * 10:.0f}%, "
            "but that mostly reflects how different the countries are, not how one country changes as it grows. "
            "Population changes scale demand one-for-one.</div>",
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
        "Choose a country to extend its recent trends to 2030. Each range comes from running the same "
        f"method from past years and checking it against what actually happened. {DATA_NOTE}",
    )

    # update : fixed 100% fossil dominance output by showing the 2030 trajectory and pace of transition
    table = transition_table(energy_df)
    countries = sorted(energy_df["country"].dropna().unique().tolist())
    country = st.selectbox("Country", countries, index=countries.index("Vietnam") if "Vietnam" in countries else 0)
    projection = build_2030_projection(energy_df, country, bundle)

    if projection is None:
        st.warning(
            "Our World in Data does not report enough yearly energy-mix data for this country to "
            "build a trend (at least five recent years are needed)."
        )
    else:
        p = projection
        if p["data_is_stale"]:
            st.warning(f"The latest energy-mix data for {country} is from {p['fossil_latest_year']}, so treat this projection with extra caution.")
        left, right = st.columns([0.9, 1.1], gap="large")
        with left:
            result_block(
                "2030 projected energy demand",
                f"{p['energy_2030']:,.0f} TWh",
                f"Likely range {p['energy_low']:,.0f}–{p['energy_high']:,.0f} TWh · "
                f"{p['energy_latest_year']}: {p['energy_latest']:,.0f} TWh",
                "lime",
            )
            result_block(
                "2030 fossil share of energy",
                f"{p['fossil_share_energy']:.1f}%",
                f"From {p['fossil_latest']:.1f}% in {p['fossil_latest_year']} · likely range {p['fossil_low']:.1f}–{p['fossil_high']:.1f}% · "
                f"{p['fossil_pace']:+.2f} pts/yr ({p['pace_group'].lower()}) · "
                f"below {FOSSIL_DOMINANT_THRESHOLD:.0f}%: {p['crossing_70'].lower()}",
                "pale_blue",
            )
            if p["transition_probability"] is not None:
                st.markdown(
                    f'<div class="small-note">Transition classifier: <b>{p["transition_probability"]:.0%}</b> chance '
                    f"that fossil share falls by {TRANSITION_DROP_PTS:.0f}+ points over the next {CLASSIFIER_HORIZON} years "
                    f"(this happened in {bundle.classifier_positive_rate:.0%} of past cases). "
                    f"Indicative only: on countries it had not seen, it scored {bundle.classifier_balanced_accuracy:.2f} "
                    f"balanced accuracy, against {bundle.baseline_balanced_accuracy:.2f} for a simple "
                    "\"it fell before, so it will fall again\" rule and 0.50 for guessing.</div>",
                    unsafe_allow_html=True,
                )
        with right:
            st.plotly_chart(projection_mix_chart(projection), use_container_width=True)

        with st.expander("Show method and inputs"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Variable": [
                            "Primary energy, latest (TWh)",
                            "Primary energy, 2030 (TWh)",
                            "Fossil share, latest (%)",
                            "Fossil share, 2030 (%)",
                            "Renewable share, 2030 (%)",
                            "Fossil share pace (pts/yr)",
                        ],
                        "Value": [
                            p["energy_latest"], p["energy_2030"], p["fossil_latest"],
                            p["fossil_share_energy"], p["renewables_share_energy"], p["fossil_pace"],
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown(
                "Energy demand: the growth rate of the last 10 years, fading by 10% each year (a damped trend). "
                "Fossil and renewable shares: a straight-line trend over the last 10 years. "
                "Ranges cover 80% of past backtest errors."
            )

    if not table.empty:
        st.markdown('<div class="section-band"></div>', unsafe_allow_html=True)
        counts = table["group"].value_counts()
        fastest = table.iloc[0]
        section_header(
            "PACE OF TRANSITION",
            "Who is actually moving?",
            f"Of {len(table)} Asian countries with recent data, {counts.get('Transitioning', 0)} cut their fossil share by "
            f"at least 0.5 points a year over the last decade, {counts.get('Flat', 0)} changed by less than that, and "
            f"{counts.get('Rising', 0) or 'none'} rose by more than 0.2 points a year. {fastest['country']} moved fastest "
            f"({fastest['pace']:+.2f} pts/yr). Almost every country is still above {FOSSIL_DOMINANT_THRESHOLD:.0f}%, "
            "so the useful question is how fast they are moving, not whether they are fossil-dominant.",
        )
        st.plotly_chart(transition_ranking(table, highlight=country), use_container_width=True)
        with st.expander("Show the full table"):
            st.dataframe(
                table.rename(
                    columns={
                        "country": "Country",
                        "latest_year": "Latest year",
                        "fossil_share": "Fossil share (%)",
                        "pace": "Pace (pts/yr)",
                        "group": "Group",
                        "crossing_70": f"Below {FOSSIL_DOMINANT_THRESHOLD:.0f}%",
                    }
                ).round(2),
                hide_index=True,
                use_container_width=True,
            )

elif section == "Model":
    section_header(
        "UNDER THE HOOD",
        "Model evidence.",
        "Each method is scored on data it has not seen: forecasts are made from past years and checked against "
        f"what happened next, and the classifiers are tested on whole countries left out of training. {DATA_NOTE}",
    )

    # update : fixed model evidence to use unseen years and countries instead of a random split
    flat_err = _err("energy", "No change (last value)")
    n_cases = int(errors[(errors["quantity"] == "energy") & (errors["method"] == "Damped growth trend")]["cases"].iloc[0])
    m1, m2 = st.columns(2)
    with m1:
        result_block(
            "2030 demand forecast error",
            f"{DEMAND_ERR * 100:.1f}%",
            f"Median error 6 years ahead across {n_cases} backtests · assuming no change: {flat_err * 100:.1f}%",
            "lime",
        )
    with m2:
        result_block(
            "Transition classifier",
            f"{bundle.classifier_balanced_accuracy:.2f}",
            f"Balanced accuracy on unseen countries · momentum rule {bundle.baseline_balanced_accuracy:.2f} · guessing 0.50",
            "pale_blue",
        )

    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.plotly_chart(backtest_scatter(bundle.backtest_6y), use_container_width=True)
    with p2:
        st.plotly_chart(method_comparison(bundle.method_errors), use_container_width=True)

    st.markdown("### What the evidence says")
    trend_tr = _err("fossil_share", "Straight-line trend", "Transitioning")
    flat_tr = _err("fossil_share", "No change (last value)", "Transitioning")
    trend_fl = _err("fossil_share", "Straight-line trend", "Flat or rising")
    flat_fl = _err("fossil_share", "No change (last value)", "Flat or rising")
    st.markdown(
        f"""
- **Each country's own trend beats a cross-country model for forecasting demand.** Predicting a country's energy use from
  GDP and population alone, without having seen that country, misses by a median **{bundle.demand_heldout_error * 100:.0f}%**
  (a linear model on log scales) and **{bundle.demand_rf_heldout_error * 100:.0f}%** (Random Forest). Countries with similar incomes
  use very different amounts of energy, so the 2030 forecast starts from each country's own history.
- **Fossil-share trends only help where something is changing.** For transitioning countries, the 10-year trend misses by
  **{trend_tr:.1f} pts** over 6 years against **{flat_tr:.1f} pts** for assuming no change. For flat or rising countries the two are
  close ({trend_fl:.1f} vs {flat_fl:.1f} pts). On average, past trends **over-predicted** fossil share by **{bundle.fossil_bias:+.1f} pts**:
  transitions have tended to speed up rather than level off.
- **The classifier is a weak signal.** It predicts whether fossil share will fall by {TRANSITION_DROP_PTS:.0f}+ points over the next
  {CLASSIFIER_HORIZON} years, which happened in {bundle.classifier_positive_rate:.0%} of {bundle.classifier_rows} country-years. It beats
  the simple momentum rule but is wrong often, so the app shows its probability as indicative only.
"""
    )

    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        st.plotly_chart(classifier_weights(bundle.classifier_coefficients), use_container_width=True)
    with c2:
        cm = bundle.classifier_confusion
        st.markdown("#### Classifier on unseen countries")
        st.dataframe(
            pd.DataFrame(
                {
                    "": ["Actually fell 3+ pts", "Did not"],
                    "Predicted fall": [int(cm[1, 1]), int(cm[0, 1])],
                    "Predicted no fall": [int(cm[1, 0]), int(cm[0, 0])],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown(
        f"Data source: [{DATA_SOURCE_URL}]({DATA_SOURCE_URL}). The original coursework reported a test R² of about 0.954 "
        "from a random train/test split. That split puts neighbouring years of the same country in both training and "
        "test data, so the model is graded on near-copies of what it learned. This version scores every method on "
        "future years and unseen countries instead."
    )
