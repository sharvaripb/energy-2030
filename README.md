# ENERGY 2030

Built as part of a university ML project (MS0003 NTU) exploring Asian energy demand and transition patterns.

The app is designed using a Venoz-inspired colour kit (https://dribbble.com/shots/25552722).

## What the project does

- Predicts primary energy consumption with a Random Forest Regressor.
- Classifies whether a country or scenario is fossil dominant using a 70% fossil-share threshold.
- Groups country-year observations into energy-pattern clusters using KMeans.
- Flags unusual regional patterns using Isolation Forest.
- Builds a transparent 2030 trend projection and passes the projected features through the classifier.

The original coursework reported a regression test R² of approximately 0.954. This version downloads the current Our World in Data Energy dataset and recomputes the train/test score in each run.

## Data

Source: Our World in Data Energy dataset

https://github.com/owid/energy-data

(The repository does not redistribute the full OWID dataset, the app downloads the current public CSV at runtime and caches it for 24 hours)
