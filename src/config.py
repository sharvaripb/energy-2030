APP_TITLE = "ENERGY / 2030"
APP_SUBTITLE = "Machine Learning for Asian Energy Demand and Transition"

DATA_URL = "https://owid-public.owid.io/data/energy/owid-energy-data.csv"
DATA_SOURCE_LABEL = "Our World in Data Energy dataset"
DATA_SOURCE_URL = "https://github.com/owid/energy-data"
GITHUB_URL = "https://github.com/sharvaripb/energy-2030"

# Venoz-inspired palette from the reference brand kit.
COLORS = {
    "ink": "#060706",
    "cream": "#E7E8D2",
    "purple": "#4C46BF",
    "lime": "#9DDC1E",
    "deep_teal": "#2B484D",
    "red": "#FC1102",
    "blue_grey": "#9FB2B4",
    "pale_blue": "#A8D2D5",
}

ASIAN_COUNTRIES = [
    "Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh",
    "Bhutan", "Brunei", "Cambodia", "China", "Cyprus", "Georgia",
    "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan",
    "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia",
    "Maldives", "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman",
    "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia",
    "Singapore", "South Korea", "Sri Lanka", "Syria", "Tajikistan",
    "Thailand", "Timor", "Turkey", "Turkmenistan", "United Arab Emirates",
    "Uzbekistan", "Vietnam", "Yemen",
]

MODEL_FEATURES = [
    "gdp",
    "population",
    "fossil_share_energy",
    "renewables_consumption",
]
TARGET = "primary_energy_consumption"
FOSSIL_DOMINANT_THRESHOLD = 70.0
