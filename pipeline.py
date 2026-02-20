import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import json

warnings.filterwarnings("ignore")


RAW_DATA_PATH = "raw_data.csv"
CLEANED_DATA_PATH = "cleaned_weather_data.csv"
DAYTIME_DATA_PATH = "daytime_weather_data.csv"
RESULTS_PATH = "model_results.json"

COLS_TO_DROP = ["id", "guid", "updated_at", "rainfall", "soilmoisture"]
RESAMPLE_FREQ = "15min"
MAX_GAP_SLOTS = 4
SLEEP_HOUR = 21
WAKE_HOUR = 9
TEST_FRACTION = 0.20

PARAMETERS = [
    "externaltemp",
    "externalhumidity",
    "pressure",
    "windspeed",
    "lightintensity",
]

MODEL_CONFIGS = {
    "externaltemp": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10,
        "daily_seasonality": True,
        "weekly_seasonality": True,
        "yearly_seasonality": False,
    },
    "externalhumidity": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10,
        "daily_seasonality": True,
        "weekly_seasonality": True,
        "yearly_seasonality": False,
    },
    "pressure": {
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5,
        "daily_seasonality": True,
        "weekly_seasonality": False,
        "yearly_seasonality": False,
    },
    "windspeed": {
        "changepoint_prior_scale": 0.1,
        "seasonality_prior_scale": 5,
        "daily_seasonality": True,
        "weekly_seasonality": False,
        "yearly_seasonality": False,
    },
    "lightintensity": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 15,
        "daily_seasonality": True,
        "weekly_seasonality": False,
        "yearly_seasonality": False,
    },
}


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV exported by main.py."""
    df = pd.read_csv(path)
    print(f"[load] {len(df)} rows loaded from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop unnecessary columns
    - Parse & set datetime index
    - Remove duplicates
    - Drop 'device' column
    - Resample to 15-min intervals
    """
    df = df.copy()

    # Drop columns
    df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)

    # Datetime index
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df.sort_values("created_at", inplace=True)
    df.set_index("created_at", inplace=True)

    # Deduplicate
    df = df[~df.index.duplicated(keep="first")]

    # Drop device column
    if "device" in df.columns:
        df.drop(columns=["device"], inplace=True)

    # Resample
    df = df.resample(RESAMPLE_FREQ).mean()

    print(f"[clean] {len(df)} rows after resample")
    print(f"[clean] Missing values:\n{df.isnull().sum().to_string()}")
    return df


def mask_nighttime(df: pd.DataFrame) -> pd.DataFrame:
    """Set nighttime rows (21:00–09:00) to NaN so models focus on daytime."""
    df = df.copy()
    hour = df.index.hour
    nighttime = (hour >= SLEEP_HOUR) | (hour < WAKE_HOUR)
    df.loc[nighttime] = np.nan
    return df


def interpolate_short_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate gaps of ≤ MAX_GAP_SLOTS consecutive NaN slots (~1 hour)."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time",
        limit=MAX_GAP_SLOTS,
        limit_direction="forward",
        limit_area="inside",
    )
    return df


def extract_daytime(df: pd.DataFrame) -> pd.DataFrame:
    """Return only 09:00–21:00 rows with at least some non-NaN values."""
    daytime = df.between_time("09:00", "21:00").dropna(how="all")
    print(f"[daytime] {len(daytime)} usable daytime rows")
    return daytime


# ─── Prepare Prophet Data

def prepare_prophet_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert a single column into the (ds, y) format Prophet expects."""
    prophet_df = df[[column]].dropna().reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    return prophet_df


# ─── Train & Evaluate

def train_model(train_df: pd.DataFrame, parameter: str) -> Prophet:
    """Train a Prophet model using the given config."""
    config = MODEL_CONFIGS[parameter]
    model = Prophet(**config)
    model.fit(train_df)
    return model


def evaluate_model(model: Prophet, test_df: pd.DataFrame, parameter: str) -> dict:
    """
    Predict on the test set and compute:
      - MAE   (Mean Absolute Error)
      - RMSE  (Root Mean Squared Error)
      - MAPE  (Mean Absolute Percentage Error)
    """
    future = test_df[["ds"]].copy()
    forecast = model.predict(future)

    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    nonzero = y_true != 0
    if nonzero.any():
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = float("nan")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2)}


def forecast_24h(model: Prophet, parameter: str) -> pd.DataFrame:
    """Generate a 24-hour (96 × 15-min) forecast from now, daytime only."""
    # Use Colombo local midnight so forecast timestamps align with the
    # naive-Colombo timestamps that Prophet was trained on.
    import pytz
    colombo = pytz.timezone("Asia/Colombo")
    # Start at tomorrow's midnight (Colombo local) so the forecast covers the next day
    last_timestamp = (
        pd.Timestamp.now(tz=colombo).normalize() + pd.Timedelta(days=1)
    ).tz_localize(None)
    future_times = pd.date_range(start=last_timestamp, periods=96, freq="15min")
    future_df = pd.DataFrame({"ds": future_times})

    forecast = model.predict(future_df)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result.columns = ["timestamp", "predicted", "lower_bound", "upper_bound"]

    result["is_daytime"] = result["timestamp"].dt.hour.between(WAKE_HOUR, SLEEP_HOUR - 1)
    result.loc[~result["is_daytime"], ["predicted", "lower_bound", "upper_bound"]] = None

    result[["predicted", "lower_bound", "upper_bound"]] = result[
        ["predicted", "lower_bound", "upper_bound"]
    ].round(2)
    result["parameter"] = parameter
    return result


# Error Heatmap 

def plot_error_heatmap(all_metrics: dict, save_path: str = "error_heatmap.png"):
    """
    Plot a heatmap of MAE / RMSE / MAPE across all parameters.
    Each metric is normalised (0–1) per column so values with very
    different scales can be compared side-by-side using a shared colour map.
    """
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = "Parameter"
    metrics_df.columns = ["MAE", "RMSE", "MAPE (%)"]

    # Normalise each metric column independently for colour mapping
    normed = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(8, max(4, len(metrics_df) * 0.8 + 1)))

    sns.heatmap(
        normed,
        ax=ax,
        annot=metrics_df.round(4),
        fmt="g",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="#cccccc",
        cbar_kws={"label": "Normalised error (per metric)"},
        vmin=0,
        vmax=1,
    )

    ax.set_title("Model Error Heatmap — Prophet (test set)", fontsize=14, pad=14)
    ax.set_xlabel("Error Metric", fontsize=11)
    ax.set_ylabel("Weather Parameter", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[plot] Error heatmap saved → {save_path}")
    plt.show()


#Main Pipeline

def run_pipeline():
    # --- Data Cleaning ---
    print("=" * 60)
    print("STEP 1: LOADING & CLEANING DATA")
    print("=" * 60)

    df = load_raw_data(RAW_DATA_PATH)
    df = clean_data(df)

    df.index = df.index.tz_convert("Asia/Colombo")

    df = mask_nighttime(df)
    df = interpolate_short_gaps(df)

    # Save cleaned outputs
    df.to_csv(CLEANED_DATA_PATH)
    daytime_df = extract_daytime(df)
    daytime_df.to_csv(DAYTIME_DATA_PATH)
    print(f"[save] Cleaned data  → {CLEANED_DATA_PATH}")
    print(f"[save] Daytime data  → {DAYTIME_DATA_PATH}")

    # Model Training & Evaluation
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING MODELS & EVALUATING ACCURACY")
    print("=" * 60)

    all_metrics = {}
    all_forecasts = {}

    for param in PARAMETERS:
        print(f"\n--- {param} ---")
        prophet_df = prepare_prophet_df(daytime_df, param)

        if len(prophet_df) < 10:
            print(f"  [skip] Not enough data ({len(prophet_df)} rows)")
            continue

        # Train / test split (chronological)
        split_idx = int(len(prophet_df) * (1 - TEST_FRACTION))
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]
        print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

        # Train
        model = train_model(train_df, param)
        print(f"  Model trained ✓")

        # Evaluate on test set
        metrics = evaluate_model(model, test_df, param)
        all_metrics[param] = metrics
        print(f"  MAE  = {metrics['MAE']}")
        print(f"  RMSE = {metrics['RMSE']}")
        print(f"  MAPE = {metrics['MAPE_%']}%")

        # Retrain on ALL data for production forecasts
        full_model = train_model(prophet_df, param)
        all_forecasts[param] = forecast_24h(full_model, param)

    # Summary
    print("\n" + "=" * 60)
    print("STEP 3: RESULTS SUMMARY")
    print("=" * 60)

    metrics_table = pd.DataFrame(all_metrics).T
    metrics_table.index.name = "parameter"
    print("\nModel Accuracy (test set):")
    print(metrics_table.to_string())
    # Error heatmap
    if all_metrics:
        plot_error_heatmap(all_metrics)
    # Save metrics
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[save] Metrics → {RESULTS_PATH}")

    # Save forecasts
    if all_forecasts:
        combined = pd.concat(all_forecasts.values(), ignore_index=True)
        combined.to_csv("forecasts_24h.csv", index=False)
        print(f"[save] 24-hour forecasts → forecasts_24h.csv")

        export_forecast_js(all_forecasts)

    print("\n✅ Pipeline complete.")
    return all_metrics, all_forecasts


# Export Forecast JS for Dashboard 

PARAM_META = {
    "externaltemp": {
        "label": "Air Temperature",
        "unit": "°C",
        "color": "rgb(255,107,74)",
        "scale": 1.0,
        "stat_type": "avg",          # headline card value = daily avg
        "decimals": 1,
    },
    "externalhumidity": {
        "label": "Air Humidity",
        "unit": "%",
        "color": "rgb(0,212,255)",
        "scale": 1.0,
        "stat_type": "avg",
        "decimals": 1,
    },
    "pressure": {
        "label": "Air Pressure",
        "unit": "hPa",
        "color": "rgb(167,139,250)",
        "scale": 0.01,               # Pa → hPa
        "stat_type": "avg",
        "decimals": 1,
    },
    "windspeed": {
        "label": "Wind Speed",
        "unit": "km/h",
        "color": "rgb(52,211,153)",
        "scale": 3.6,                # m/s → km/h
        "stat_type": "avg",
        "decimals": 1,
    },
    "lightintensity": {
        "label": "Light Intensity",
        "unit": "klux",
        "color": "rgb(251,191,36)",
        "scale": 0.001,              # lux → klux
        "stat_type": "peak",
        "decimals": 1,
    },
}

# Dashboard window: 09:30 – 20:30
DASH_START = "09:30"
DASH_END   = "20:30"


def export_forecast_js(all_forecasts: dict, path: str = "forecast_output.js"):
    """
    Convert the forecast DataFrames into a JavaScript variable file
    (forecast_output.js) consumed by dashboard.html without a web server.
    Only the 09:30–20:30 window is included.
    """
    output = {}

    for param, df in all_forecasts.items():
        if param not in PARAM_META:
            continue
        meta = PARAM_META[param]
        scale = meta["scale"]

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        full_df = df.copy()

        mask = (
            df["timestamp"].dt.strftime("%H:%M") >= DASH_START
        ) & (
            df["timestamp"].dt.strftime("%H:%M") <= DASH_END
        ) & (
            df["predicted"].notna()
        )
        window = df[mask].copy()

        def _scale(series):
            s = series * scale
            if param == "lightintensity":
                s = s.clip(lower=0)
            return s.round(meta["decimals"])

        slots = []
        for _, row in window.iterrows():
            slots.append({
                "time":      row["timestamp"].strftime("%H:%M"),
                "predicted": round(float(_scale(pd.Series([row["predicted"]]))[0]), meta["decimals"]) if pd.notna(row["predicted"]) else None,
                "lower":     round(float(_scale(pd.Series([row["lower_bound"]]))[0]), meta["decimals"]) if pd.notna(row["lower_bound"]) else None,
                "upper":     round(float(_scale(pd.Series([row["upper_bound"]]))[0]), meta["decimals"]) if pd.notna(row["upper_bound"]) else None,
            })

        # Full 24-hour slots (includes nighttime nulls)
        slots_24h = []
        for _, row in full_df.iterrows():
            slots_24h.append({
                "time":       row["timestamp"].strftime("%H:%M"),
                "predicted":  round(float(_scale(pd.Series([row["predicted"]]))[0]), meta["decimals"]) if pd.notna(row["predicted"]) else None,
                "lower":      round(float(_scale(pd.Series([row["lower_bound"]]))[0]), meta["decimals"]) if pd.notna(row["lower_bound"]) else None,
                "upper":      round(float(_scale(pd.Series([row["upper_bound"]]))[0]), meta["decimals"]) if pd.notna(row["upper_bound"]) else None,
                "is_daytime": bool(row["is_daytime"]) if "is_daytime" in row.index else True,
            })

        # Headline stats from window
        preds = [s["predicted"] for s in slots if s["predicted"] is not None]
        if preds:
            avg_val  = round(sum(preds) / len(preds), meta["decimals"])
            peak_val = round(max(preds), meta["decimals"])
            min_val  = round(min(preds), meta["decimals"])
        else:
            avg_val = peak_val = min_val = None

        stat_val = {"avg": avg_val, "peak": peak_val, "min": min_val}[meta["stat_type"]]

        output[param] = {
            "label":      meta["label"],
            "unit":       meta["unit"],
            "color":      meta["color"],
            "stat_type":  meta["stat_type"],
            "stat_value": stat_val,
            "avg":        avg_val,
            "peak":       peak_val,
            "min":        min_val,
            "slots":      slots,
            "slots_24h":  slots_24h,
        }

    forecast_date = None
    for df in all_forecasts.values():
        dt = pd.to_datetime(df["timestamp"].iloc[0])
        forecast_date = dt.strftime("%Y-%m-%d")
        break

    payload = json.dumps({
        "generated_at":   pd.Timestamp.now().isoformat(timespec="seconds"),
        "forecast_date":  forecast_date,
        "window":         f"{DASH_START} – {DASH_END}",
        "parameters":     output,
    }, indent=2)

    with open(path, "w") as f:
        f.write(f"// Auto-generated by pipeline.py — do not edit manually\n")
        f.write(f"const FORECAST_DATA = {payload};\n")

    print(f"[export] Dashboard data → {path}")


if __name__ == "__main__":
    run_pipeline()
