# Weather Forecast Pipeline

A Python pipeline that fetches weather sensor data from the [DataCanvas](https://datacanvas.io) platform, trains [Prophet](https://facebook.github.io/prophet/) time-series models for five atmospheric parameters, and serves next-day forecasts through a self-contained HTML dashboard.

## Overview

```
data-gathering.py        ← Fetches raw data from DataCanvas API → raw_data.csv
pipeline.py              ← Cleans data, trains models, evaluates, exports forecast
explainability.py        ← Generates model explainability artefacts
dashboard.html           ← Static dashboard; reads forecast_output.js
```

## Features

- **Data ingestion** — Paginates through the DataCanvas API and writes a raw CSV.
- **Data cleaning** — Resamples to 15-minute intervals, masks nighttime (21:00–09:00 local), and interpolates short gaps (≤ 4 slots).
- **Per-parameter Prophet models** — Individually tuned changepoint and seasonality priors for each weather variable.
- **Evaluation** — MAE, RMSE, and sMAPE on a chronological 20 % hold-out, plus a normalised error heatmap.
- **24-hour forecasts** — Point predictions with confidence intervals for the next calendar day (Asia/Colombo timezone), exported as `forecast_output.js`.
- **Static dashboard** — No web server required; open `dashboard.html` directly in a browser.
- **Explainability suite** — Component decomposition, feature importance (variance of Prophet components), partial dependence plots (hour-of-day & day-of-week), changepoint analysis, and residual diagnostics.

## Forecast Parameters

| Parameter | Unit | Stat shown |
|---|---|---|
| Air Temperature | °C | Daily average |
| Air Humidity | % | Daily average |
| Air Pressure | hPa | Daily average |
| Wind Speed | km/h | Daily average |
| Light Intensity | klux | Daily peak |

## Project Structure

```
├── data-gathering.py         # Step 0 – pull raw data from DataCanvas
├── pipeline.py               # Step 1 – clean, train, evaluate, forecast
├── explainability.py         # Step 2 – model explainability (optional)
├── dashboard.html            # Interactive forecast viewer
├── raw_data.csv              # Output of data-gathering.py
├── cleaned_weather_data.csv  # 15-min resampled, masked series
├── daytime_weather_data.csv  # Daytime-only rows used for training
├── model_results.json        # Test-set metrics (MAE / RMSE / sMAPE)
├── forecasts_24h.csv         # Raw forecast table
├── forecast_output.js        # Dashboard-ready JS payload
├── explainability/           # PNG charts + summary text
├── requiremements.txt        # Full pinned dependency list
└── requirements-ci.txt       # Lean CI dependency list
```

## Setup

### 1. Install dependencies

```bash
pip install -r requiremements.txt
```

For a leaner install (no Jupyter packages):

```bash
pip install -r requirements-ci.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
DATACANVAS_ACCESS_KEY_ID=your_access_key_id
DATACANVAS_SECRET_KEY=your_secret_key
DATACANVAS_PROJECT_ID=your_project_id
DATACANVAS_BASE_URL=https://your-datacanvas-instance.com
```

## Usage

### Step 0 — Fetch raw data

```bash
python data-gathering.py
```

Downloads all records for the configured device and writes `raw_data.csv`.

### Step 1 — Run the forecasting pipeline

```bash
python pipeline.py
```

Runs data cleaning, model training, evaluation, and 24-hour forecast generation. Outputs:

- `cleaned_weather_data.csv`
- `daytime_weather_data.csv`
- `model_results.json`
- `forecasts_24h.csv`
- `forecast_output.js`
- `error_heatmap.png`

### Step 2 — (Optional) Generate explainability artefacts

```bash
python explainability.py
```

Writes PNG charts and `explainability/explainability_summary.txt`.

### Step 3 — View the dashboard

Open `dashboard.html` in a browser. No web server is needed — it reads `forecast_output.js` from the same directory.

## Explainability Outputs

| File | Description |
|---|---|
| `components_<param>.png` | Trend + seasonality decomposition |
| `feature_importance.png` | Variance contribution of each component |
| `pdp_hour_<param>.png` | Partial dependence on hour of day |
| `pdp_dow_<param>.png` | Partial dependence on day of week |
| `changepoints_<param>.png` | Detected trend changepoints |
| `residuals_<param>.png` | Residuals over time and by hour |
| `explainability_summary.txt` | Plain-text interpretation of all parameters |

## Configuration

Key constants at the top of `pipeline.py`:

| Constant | Default | Description |
|---|---|---|
| `RESAMPLE_FREQ` | `15min` | Resampling interval |
| `SLEEP_HOUR` | `21` | Start of masked nighttime |
| `WAKE_HOUR` | `9` | End of masked nighttime |
| `MAX_GAP_SLOTS` | `4` | Max consecutive NaN slots to interpolate (~1 hour) |
| `TEST_FRACTION` | `0.20` | Hold-out fraction for evaluation |

## Dependencies

Core libraries:

- [Prophet](https://facebook.github.io/prophet/) — time-series forecasting
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data processing
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) — visualisation
- [datacanvas](https://pypi.org/project/datacanvas/) — DataCanvas API SDK
- [python-dotenv](https://pypi.org/project/python-dotenv/) — environment variable management
