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
SKIP_EVALUATION = ["lightintensity"]
COLS_TO_DROP = ["id", "guid", "updated_at", "rainfall", "soilmoisture"]
RESAMPLE_FREQ = "15min"

MAX_GAP_SLOTS = 4
SLEEP_HOUR = 21
WAKE_HOUR = 9
TEST_FRACTION = 0.20




TRAIN_SCALE = {
    "pressure": 0.01,    # Pa → hPa
}

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

def drop_gap_boundaries(series: pd.Series, gap_threshold_slots: int = 5, boundary_slots: int = 3) -> pd.Series:
    """
    Remove rows that sit within `boundary_slots` positions immediately AFTER
    a long NaN gap (>= gap_threshold_slots consecutive NaNs).
    These boundary rows often carry unreliable interpolated edge values that
    create large residual spikes and inflate RMSE.
    """
    is_null = series.isna()
    # Count consecutive NaNs ending at each position
    null_run = is_null.groupby((~is_null).cumsum()).cumsum()
    # Identify positions right after a gap that was long enough
    gap_end = (null_run.shift(1).fillna(0) >= gap_threshold_slots) & (~is_null)
    # Mark boundary_slots rows following each gap end
    drop_mask = pd.Series(False, index=series.index)
    gap_positions = series.index[gap_end]
    for pos in gap_positions:
        loc = series.index.get_loc(pos)
        end_loc = min(loc + boundary_slots, len(series))
        drop_mask.iloc[loc:end_loc] = True
    cleaned = series.copy()
    cleaned[drop_mask] = np.nan
    n_dropped = drop_mask.sum()
    if n_dropped:
        print(f"    [gap-boundary] dropped {n_dropped} boundary rows")
    return cleaned


def prepare_prophet_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert a single column into the (ds, y) format Prophet expects.
    Applies gap-boundary cleaning and unit pre-scaling where configured."""
    series = df[column].copy()
    series = drop_gap_boundaries(series)
    prophet_df = series.dropna().reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    if column in TRAIN_SCALE:
        prophet_df["y"] = (prophet_df["y"] * TRAIN_SCALE[column]).round(2)
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
      - sMAPE (Symmetric MAPE) — robust to near-zero actuals.
              Standard MAPE breaks for wind speed (near-zero values produce
              divisions like |0.2 - 1.7| / 0.2 = 750%), so sMAPE is used:
              sMAPE = 2|y - ŷ| / (|y| + |ŷ|), bounded 0-200%.
    """
    future = test_df[["ds"]].copy()
    forecast = model.predict(future)

    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 0
    if valid.any():
        smape = np.mean(2 * np.abs(y_true[valid] - y_pred[valid]) / denom[valid]) * 100
    else:
        smape = float("nan")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "sMAPE_%": round(smape, 2)}


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
    metrics_df.columns = ["MAE", "RMSE", "sMAPE"]

    # Normalise each metric column independently for colour mapping
    normed = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(8, max(4, len(metrics_df) * 0.8 + 1)))

    sns.heatmap(
        normed,
        ax=ax,
        annot=metrics_df.round(4),
        fmt="g",
        cmap="Blues",
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

        if param in SKIP_EVALUATION:
            full_model = train_model(prophet_df, param)
            all_forecasts[param] = forecast_24h(full_model, param)
            continue

        # Train / test split (chronological)
        split_idx = int(len(prophet_df) * (1 - TEST_FRACTION))
        train_df = prophet_df.iloc[:split_idx]
        test_df  = prophet_df.iloc[split_idx:]
        print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

        model = train_model(train_df, param)
        print(f"  Model trained ✓")

        metrics = evaluate_model(model, test_df, param)
        all_metrics[param] = metrics
        print(f"  MAE   = {metrics['MAE']}")
        print(f"  RMSE  = {metrics['RMSE']}")
        print(f"  sMAPE = {metrics['sMAPE_%']}%")

        full_model = train_model(prophet_df, param)
        all_forecasts[param] = forecast_24h(full_model, param)

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
        "color": "#FF6B4A",
        "scale": 1.0,
        "stat_type": "avg",          # headline card value = daily avg
        "decimals": 1,
    },
    "externalhumidity": {
        "label": "Air Humidity",
        "unit": "%",
        "color": "#00D4FF",
        "scale": 1.0,
        "stat_type": "avg",
        "decimals": 1,
    },
    "pressure": {
        "label": "Air Pressure",
        "unit": "hPa",
        "color": "#A78BFA",
        "scale": 1.0,               # Already in hPa — pre-scaled before training
        "stat_type": "avg",
        "decimals": 1,
    },
    "windspeed": {
        "label": "Wind Speed",
        "unit": "km/h",
        "color": "#34D399",
        "scale": 3.6,                # m/s → km/h
        "stat_type": "avg",
        "decimals": 1,
    },
    "lightintensity": {
        "label": "Light Intensity",
        "unit": "klux",
        "color": "#FBBF24",
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
