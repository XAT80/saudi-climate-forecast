"""
Madar — 24-hour temperature forecast (Streamlit app)

Run from the project root (the folder that contains models/ and raw_data/):
    pip install streamlit
    streamlit run app.py

The app does NOT train anything: it loads the saved pipeline from models/
(model + train-only scaler + settings) and forecasts the next 24 hours.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "raw_data" / "saudi-hourly-weather-data.csv"

COLORS = {"actual": "#2b2b2b", "forecast": "#2a78d6", "now": "#8a8a8a"}


# ---------------------------------------------------------------- pipeline
@st.cache_resource
def load_pipeline(model_dir=str(MODEL_DIR)):
    from tensorflow import keras
    with open(f"{model_dir}/pipeline_config.json") as f:
        config = json.load(f)
    model = keras.models.load_model(f"{model_dir}/{config['model_file']}")
    return model, config


@st.cache_data
def load_hourly(path, timezone, station, fake_threshold):
    "Same cleaning as in training: one station, fake values -> NaN, minute-0 readings, gaps not filled."
    df = pd.read_csv(path, sep=";", usecols=["station_name", "observation_date", "air_temperature"])
    df["observation_date"] = pd.to_datetime(df["observation_date"], utc=True).dt.tz_convert(timezone)
    df["temp"] = df["air_temperature"].where(df["air_temperature"] < fake_threshold)
    s = df[(df["station_name"] == station) & (df["observation_date"].dt.minute == 0)]
    hourly = s.groupby("observation_date")["temp"].mean()
    return hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"))


def build_features(index, temp_c, config):
    sc = config["scaler"]
    hour, doy, dow = index.hour.to_numpy(), index.dayofyear.to_numpy(), index.dayofweek.to_numpy()
    cols = {"temp": (np.asarray(temp_c, dtype=float) - sc["mean_C"]) / sc["std_C"],
            "hour_sin": np.sin(2 * np.pi * hour / 24),    "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin":  np.sin(2 * np.pi * doy / 365.25), "doy_cos":  np.cos(2 * np.pi * doy / 365.25),
            "dow_sin":  np.sin(2 * np.pi * dow / 7),      "dow_cos":  np.cos(2 * np.pi * dow / 7)}
    return np.stack([cols[f] for f in config["features"]], axis=-1)       # same order as in training


def complete_origins(series, config):
    "Every hour that closes a complete input window (no missing hours)."
    w = config["window_hours"]
    full = series.notna().astype(int).rolling(w).sum() == w
    return full[full].index


def forecast(series, origin, model, config):
    W, H = config["window_hours"], config["horizon_hours"]
    past = series.loc[origin - pd.Timedelta(hours=W - 1): origin]           # only values up to the origin
    if len(past) != W or past.isna().any():
        raise ValueError(f"need {W} complete hours ending at {origin}")
    z = model.predict(build_features(past.index, past.to_numpy(), config)[None], verbose=0)[0]
    sc = config["scaler"]
    fc = pd.Series(z * sc["std_C"] + sc["mean_C"],                           # back to °C
                   index=origin + pd.to_timedelta(np.arange(1, H + 1), unit="h"), name="forecast_C")
    return past, fc


def split_of(ts, config):
    for name, (a, b) in config["splits"].items():
        if pd.Timestamp(a, tz=config["timezone"]) <= ts <= pd.Timestamp(b, tz=config["timezone"]):
            return name
    return "outside the splits"


def plot_forecast(past, fc, origin, config, actual=None):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axvspan(fc.index[0], fc.index[-1], color=COLORS["forecast"], alpha=0.07, lw=0)
    ax.plot(past.index, past.to_numpy(), color=COLORS["actual"], lw=2,
            label=f"last {config['window_hours']} h (input)")
    ax.plot([origin, *fc.index], [past.iloc[-1], *fc.to_numpy()], color=COLORS["forecast"], lw=2.5,
            marker="o", ms=4, markevery=range(1, len(fc) + 1), label="forecast (next 24 h)")
    if actual is not None and actual.notna().any():
        ax.plot(actual.index, actual.to_numpy(), color=COLORS["actual"], lw=1.5, ls="--", label="actual")
    lo, hi = min(past.min(), fc.min()), max(past.max(), fc.max())
    ax.set_ylim(lo - 4, hi + 3)
    ax.axvline(origin, color=COLORS["now"], lw=1)
    ax.text(origin, hi + 3, "  now", va="top", color="#5a5a5a")
    for t, label in [(fc.idxmax(), "max"), (fc.idxmin(), "min")]:
        ax.annotate(f"{label} {fc[t]:.1f} °C\n{t:%H:%M}", (t, fc[t]), textcoords="offset points",
                    xytext=(0, 10 if label == "max" else -34), ha="center", fontsize=10, color="#1c5cab")
    ax.set(ylabel="°C", title=f"Next 24 h — forecast made at {origin:%Y-%m-%d %H:%M}")
    ax.grid(alpha=0.3); ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Madar — 24 h forecast", page_icon="🌦️", layout="wide")
st.title("🌦️ Madar — 24-hour temperature forecast")
st.caption("King Khaled International Airport, Riyadh · LSTM · no training here, the saved pipeline is loaded from `models/`")

try:
    model, config = load_pipeline()
except Exception as e:
    st.error(f"Could not load the saved pipeline from `{MODEL_DIR}`: {e}")
    st.stop()

data_path = st.sidebar.text_input("Data file (CSV, same columns)", str(DATA_PATH))
try:
    hourly = load_hourly(data_path, config["timezone"], config["station"],
                         config["cleaning"]["fake_value_threshold"])
except Exception as e:
    st.error(f"Could not read the data file: {e}")
    st.stop()

origins = complete_origins(hourly, config)
if len(origins) == 0:
    st.error(f"No complete {config['window_hours']} h window in the data.")
    st.stop()

# --- sidebar: choose when the forecast is made
st.sidebar.header("Forecast time")
mode = st.sidebar.radio("Make the forecast at", ["Latest available hour", "Choose a date and hour"])
if mode == "Latest available hour":
    origin = origins[-1]
else:
    day = st.sidebar.date_input("Date", value=origins[-1].date(),
                                min_value=origins[0].date(), max_value=origins[-1].date())
    hour = st.sidebar.selectbox("Hour", list(range(24)), index=int(origins[-1].hour),
                                format_func=lambda h: f"{h:02d}:00")
    origin = pd.Timestamp(f"{day} {hour:02d}:00", tz=config["timezone"])
    if origin not in origins:
        st.sidebar.warning("Some of the 72 hours before this time are missing.")
        later = origins[origins >= origin]
        origin = later[0] if len(later) else origins[-1]
        st.sidebar.info(f"Using the next complete hour: {origin:%Y-%m-%d %H:%M}")

split = split_of(origin, config)
st.sidebar.header("Actual values")
show_actual = st.sidebar.checkbox("Show the actual temperature (if available)", value=False)
if split == "test":
    st.sidebar.info("This time is in the **2025 test period** (final evaluation already done).")

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
tab_fc, tab_res, tab_exp, tab_err = st.tabs(["🌦️ Forecast", "📊 Results", "🧪 Experiments", "🔍 Error analysis"])

# --- forecast
with tab_fc:
    past, fc = forecast(hourly, origin, model, config)
    actual = hourly.reindex(fc.index) if show_actual else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast made at", f"{origin:%Y-%m-%d %H:%M}")
    c2.metric("Max next 24 h", f"{fc.max():.1f} °C", f"at {fc.idxmax():%H:%M}", delta_color="off")
    c3.metric("Min next 24 h", f"{fc.min():.1f} °C", f"at {fc.idxmin():%H:%M}", delta_color="off")
    c4.metric("Mean next 24 h", f"{fc.mean():.1f} °C")

    st.pyplot(plot_forecast(past, fc, origin, config, actual))

    if actual is not None and actual.notna().all():
        err = (fc - actual)
        st.info(f"For this one forecast: MAE {err.abs().mean():.2f} °C · bias {err.mean():+.2f} °C "
                f"(one example only — the model's score is on 4,321 validation forecasts).")

    with st.expander("Hourly forecast table"):
        table = pd.DataFrame({"time": fc.index.strftime("%Y-%m-%d %H:%M"),
                              "hours ahead": np.arange(1, len(fc) + 1),
                              "forecast (°C)": fc.round(1).to_numpy()})
        if actual is not None:
            table["actual (°C)"] = actual.round(1).to_numpy()
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("Download as CSV", table.to_csv(index=False), "madar_forecast.csv", "text/csv")

    with st.expander("About the model"):
        val = config.get("validation", {})
        st.markdown(
            f"- **Model:** `{config['selected_experiment']}` — LSTM, "
            f"{config['window_hours']} h in → {config['horizon_hours']} h out\n"
            f"- **Inputs (in order):** {', '.join(config['features'])}\n"
            f"- **Scaler (train only):** mean {config['scaler']['mean_C']:.2f} °C, std {config['scaler']['std_C']:.2f} °C\n"
            f"- **Data until:** {hourly.index.max():%Y-%m-%d %H:%M} · this forecast time is in the **{split}** period"
        )
        if val:
            st.dataframe(pd.DataFrame(val).T.rename_axis("validation"), use_container_width=True)


def show_figure(name, caption):
    path = FIG / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.warning(f"`{path.relative_to(ROOT)}` not found — run notebook 03 first.")


# --- results
with tab_res:
    st.subheader("Validation results — same 4,321 forecasts for every model")
    lb_path = REPORTS / "predictions" / "validation_day3_leaderboard.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        st.dataframe(lb, hide_index=True, use_container_width=True)
        st.caption("MAE / RMSE in °C, lower is better. `MAE_vs_SN_%` = difference vs seasonal naive on the same dates. "
                   "`lstm_w168` is scored on fewer dates (only those with a full week of history).")
    else:
        st.warning(f"`{lb_path.relative_to(ROOT)}` not found — run notebook 03 first.")
    show_figure("day3_06_leaderboard.png", "Validation MAE of every model")

    st.subheader("Final test — 2025, never used for any choice (run once)")
    test_path = REPORTS / "predictions" / "test_final_scores.csv"
    if test_path.exists():
        st.dataframe(pd.read_csv(test_path).round(3), hide_index=True, use_container_width=True)
        st.caption("Same 24-hour horizon and metrics as validation, compared with the simple forecasts on the same 2025 dates.")
    else:
        st.info("Final test not run yet.")
    show_figure("day4_test_actual_vs_forecast.png", "Test 2025: day-ahead forecasts vs actual values, and MAE by hours ahead")
    show_figure("day3_01_actual_vs_forecast.png", "Day-ahead forecasts vs actual values (spring and summer 2024)")

# --- experiments
with tab_exp:
    st.subheader("What we tried on Day 3 — one change at a time")
    log_path = REPORTS / "experiment_log_day3.csv"
    if log_path.exists():
        log = pd.read_csv(log_path).drop(columns=["settings"], errors="ignore")
        st.dataframe(log, hide_index=True, use_container_width=True)
        st.caption("`ΔMAE_vs_ref` < 0 = better than the reference (`ref`) on the same dates. "
                   "Training noise (same model, different seed) ≈ 0.006 °C, so gains under ~0.02 °C are not trusted.")
    else:
        st.warning(f"`{log_path.relative_to(ROOT)}` not found — run notebook 03 first.")
    st.markdown(
        "- **Kept:** hour of day and day of year (calendar, known in advance) → MAE 1.456 → 1.359 °C\n"
        "- **Helped alone but not with calendar inputs:** 64 units\n"
        "- **Rejected:** 168 h window (no gain, half the dates), dropout 0.2 (worse: the model was underfitting)\n"
        "- **Not trusted:** day of week (gain below the safety margin, no weekly cycle in temperature)"
    )

# --- error analysis
with tab_err:
    st.subheader(f"Error analysis of the selected model (`{config['selected_experiment']}`)")
    c1, c2 = st.columns(2)
    with c1:
        show_figure("day3_05_beat_baseline.png", "Where it beats seasonal naive: by month and by hours ahead")
        show_figure("day3_03_peaks.png", "Daily maximum and minimum: forecast vs actual")
    with c2:
        show_figure("day3_02_bias.png", "Bias (forecast − actual) by hours ahead, hour of day and month")
        show_figure("day3_04_worst_days.png", "Worst forecasts: sudden cooling events")
    st.info("Main limitation: the worst forecasts are sudden cooling events that the past 72 h of temperature "
            "cannot announce. Next step: inputs such as dew point, pressure or wind.")
