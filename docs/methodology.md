# Methodology

## Academic origin

The university project explored relationships among GDP, population, fossil fuel share, renewable energy consumption and primary energy consumption across Asian countries. It also included fossil-dominance classification, clustering, anomaly detection and a 2030 visualization.

## Cleanup decisions

This public version removes personal student identifiers, team-member identifiers, local Windows file paths and classroom slide assets. It also separates data loading, modeling, charting and interface code into small reusable modules.

## Data window

The application filters the public Our World in Data Energy dataset to Asian countries and years 2012 to 2024. The supervised models use rows that are complete for the four project features and target.

## Regression

Features:

- GDP
- Population
- Fossil share of energy
- Renewable energy consumption

Target:

- Primary energy consumption

Model:

- Random Forest Regressor

Evaluation:

- 80/20 random train/test split with `random_state=42`
- Test R² displayed in the app

## Classification

The target is `fossil_dominant`, defined as fossil share of energy greater than 70%.

Model:

- Random Forest Classifier with balanced class weights

## Clustering and anomaly detection

The project uses log-transformed magnitude features where appropriate, standardizes them and applies:

- KMeans with 4 clusters
- Isolation Forest with 8% contamination

These are exploratory tools rather than ground-truth labels.

## 2030 scenario

For a selected country, the latest six available observations are used to fit a simple linear trend for each required input. Those trends are extended to 2030. The projected features are then passed through the trained models.

This is intentionally labeled a trend projection. It is not a policy, macroeconomic or causal forecast.
