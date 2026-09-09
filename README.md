# ENERGY / 2030

A cleaned and reproducible version of a university machine learning project exploring Asian energy demand and transition patterns.

The live Streamlit app is designed as a compact portfolio piece rather than a classroom dashboard. It uses a Venoz-inspired color system, a restrained editorial layout and one original hero graphic.

## What the project does

- Predicts primary energy consumption with a Random Forest Regressor.
- Classifies whether a country or scenario is fossil dominant using a 70% fossil-share threshold.
- Groups country-year observations into energy-pattern clusters using KMeans.
- Flags unusual regional patterns using Isolation Forest.
- Builds a transparent 2030 trend projection and passes the projected features through the classifier.

The original coursework reported a regression test R² of approximately 0.954. This repository does not hard-code that result. The app downloads the current Our World in Data Energy dataset and recomputes its own reproducible train/test score.

## Data

Source: Our World in Data Energy dataset

https://github.com/owid/energy-data

The repository does not redistribute the full OWID dataset. The app downloads the current public CSV at runtime and caches it for 24 hours.

## Project structure

```text
energy-2030/
├── app.py
├── requirements.txt
├── README.md
├── DEPLOY.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── hero.svg
│   └── styles.css
├── docs/
│   └── methodology.md
├── notebooks/
│   └── energy_analysis_clean.ipynb
└── src/
    ├── charts.py
    ├── config.py
    ├── data.py
    ├── modeling.py
    ├── projections.py
    └── ui.py
```

## Run locally

Use Python 3.12 to match the recommended Streamlit Community Cloud deployment setup.

```bash
python -m venv .venv
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Before publishing

Open `src/config.py` and replace:

```python
GITHUB_URL = "https://github.com/your-username/energy-2030"
```

with your real repository URL.

## Resume link

Recommended display text:

**Interactive Demo** | **GitHub**

Suggested project title:

**Energy Demand Prediction and 2030 Transition Modeling**

Suggested bullet:

> Built Random Forest models to predict primary energy consumption from GDP, population and energy-mix indicators, then added clustering, anomaly detection and an interactive 2030 trend scenario for Asian economies.
