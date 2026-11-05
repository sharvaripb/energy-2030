**ENERGY 2030**

Built as part of a university ML project (MS0003 NTU) exploring Asian energy demand and transition patterns.

The app is designed using a Venoz-inspired colour kit (https://dribbble.com/shots/25552722).

**What the project does**

- Forecasts each country's 2030 primary energy consumption from its recent growth trend, with a likely range based on how well the same method predicted past years.
- Projects each country's fossil share of energy to 2030 and estimates when it falls below 70%.
- Ranks Asian countries by how fast their fossil share is changing.
- Estimates the chance that a country's fossil share falls by 3 or more points over the next 5 years using logistic regression.
- Runs what-if scenarios by changing a country's GDP or population.
- Groups country-year observations into energy-pattern clusters using KMeans.
- Flags unusual regional patterns using Isolation Forest.

The original coursework reported a regression test R² of approximately 0.954. This version downloads the current Our World in Data Energy dataset and tests every model on years and countries it was not trained on, instead of a random train/test split.

**Data**

Source: Our World in Data Energy dataset (Asian countries, 1995–2024)

https://github.com/owid/energy-data

(The repository does not redistribute the full OWID dataset, the app downloads the current public CSV at runtime and caches it for 24 hours)
