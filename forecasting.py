# pages/forecasting.py
# ------------------------------------------------------------
# Forecasting Dashboard (3 champion plots)
#   1) Total Rentals
#   2) Top Vehicle Groups (each on its own chart)
#   3) Total Revenue
# No confidence intervals. Uses DATA_PATH only.
# ------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="Forecasting", page_icon="📈", layout="wide")

DATA_PATH = "merged_df_further_cleaned.xlsx"

COL_DATE = "Checkout Date"
COL_ID   = "row_id_for_counts"
COL_REV  = "Net Time&Dist Amount"
COL_VEH  = "Vehicle Group Rented"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel(DATA_PATH)

    # Dates
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE]).copy()
    df["__date_idx__"] = df[COL_DATE]

    # Counting key
    if COL_ID not in df.columns:
        df[COL_ID] = range(1, len(df) + 1)

    # Daily Rate proxy if missing
    if "Daily Rate" not in df.columns:
        num   = pd.to_numeric(df.get(COL_REV, np.nan), errors="coerce")
        denom = pd.to_numeric(df.get("Days Charged Count", np.nan), errors="coerce").replace(0, np.nan)
        df["Daily Rate"] = (num / denom) / 100

    df["checkout_month"] = df["__date_idx__"].dt.month
    return df

def create_time_series_features(y: pd.Series) -> pd.DataFrame:
    """Lag + calendar features for ML models."""
    X = pd.DataFrame(index=y.index)
    X["month"] = X.index.month
    X["year"]  = X.index.year
    for i in range(1, 13):
        X[f"lag_{i}"] = y.shift(i)
    return X

def build_exog(df: pd.DataFrame, idx="__date_idx__", cols=("checkout_month", "Daily Rate")):
    exog_raw = df.set_index(idx)[list(cols)]
    for c in exog_raw.columns:
        exog_raw[c] = pd.to_numeric(exog_raw[c], errors="coerce")
    exog = exog_raw.resample("M").mean().asfreq("M")
    exog = exog.fillna(method="ffill").fillna(method="bfill").fillna(0)
    return exog

def make_exog_future(exog_hist: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Repeat last-year pattern if available; else last value."""
    start = exog_hist.index[-1] + pd.DateOffset(months=1)
    future_idx = pd.date_range(start=start, periods=horizon, freq="M")
    exog_future = pd.DataFrame(index=future_idx, columns=exog_hist.columns)
    for c in exog_hist.columns:
        if len(exog_hist) >= 12:
            last_year = exog_hist[c].iloc[-12:]
            month_map = last_year.groupby(last_year.index.month).mean()
            exog_future[c] = [month_map.get(dt.month, last_year.iloc[-1]) for dt in future_idx]
        else:
            exog_future[c] = exog_hist[c].iloc[-1]
    return exog_future

def rmse(a, b):
    return sqrt(mean_squared_error(a, b))

def arima_grid(y_train, y_test, orders):
    best = (np.inf, None)
    for order in orders:
        try:
            m = ARIMA(y_train, order=order).fit()
            f = m.forecast(steps=len(y_test))
            e = rmse(y_test, f)
            if e < best[0]:
                best = (e, order)
        except Exception:
            continue
    return best  # (rmse, order)

def sarima_grid(y_train, y_test, pdq_list, seasonal_pdq_list):
    best = (np.inf, None, None)
    for order in pdq_list:
        for sorder in seasonal_pdq_list:
            try:
                m = SARIMAX(y_train, order=order, seasonal_order=sorder,
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                f = m.forecast(steps=len(y_test))
                e = rmse(y_test, f)
                if e < best[0]:
                    best = (e, order, sorder)
            except Exception:
                continue
    return best  # (rmse, order, seasonal_order)

def arimax_grid(y_train, y_test, ex_train, ex_test, orders):
    best = (np.inf, None)
    for order in orders:
        try:
            m = ARIMA(y_train, exog=ex_train, order=order).fit()
            f = m.forecast(steps=len(y_test), exog=ex_test)
            e = rmse(y_test, f)
            if e < best[0]:
                best = (e, order)
        except Exception:
            continue
    return best  # (rmse, order)

def evaluate_ml(y_train, y_test):
    """Small set of ML models; returns (best_rmse, name, model_fitted, feature_names)."""
    X_train = create_time_series_features(y_train).dropna()
    y_train2 = y_train.loc[X_train.index]

    # Build X_test aligned to the same lag structure
    X_all = create_time_series_features(pd.concat([y_train, y_test]))
    X_test = X_all.loc[y_test.index].copy()
    # Fill forward lags that cross boundary
    for i in range(1, 13):
        lag_col = f"lag_{i}"
        if lag_col in X_test.columns and lag_col not in X_train.columns:
            X_test.drop(columns=[lag_col], inplace=True, errors="ignore")

    # Ensure same columns/order
    X_test = X_test[X_train.columns]

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "SVR (RBF)": SVR(kernel="rbf", C=10.0, epsilon=0.2),
        "Linear Regression": LinearRegression()
    }

    best = (np.inf, None, None)
    for name, model in models.items():
        try:
            model.fit(X_train, y_train2)
            preds = model.predict(X_test)
            e = rmse(y_test, preds)
            if e < best[0]:
                best = (e, name, model)
        except Exception:
            continue

    return best[0], best[1], best[2], list(X_train.columns)

def iterative_ml_forecast(model, feat_cols, y_hist: pd.Series, horizon: int) -> pd.Series:
    """Roll ML forecast H steps ahead using lag features."""
    # Build features on full history
    X = create_time_series_features(y_hist).dropna()
    y_fit = y_hist.loc[X.index]
    # Refit on all history (to mirror ARIMA refit-on-full behavior)
    model.fit(X[feat_cols], y_fit)

    start = y_hist.index[-1] + pd.DateOffset(months=1)
    future_idx = pd.date_range(start=start, periods=horizon, freq="M")
    # Seed last feature vector
    last_row = X.iloc[-1].copy()

    preds = []
    for h in range(horizon):
        # Advance month/year
        current_dt = future_idx[h]
        last_row["month"] = current_dt.month
        last_row["year"]  = current_dt.year

        # Predict
        df_pred = pd.DataFrame([last_row.values], columns=last_row.index)[feat_cols]
        y_hat = float(model.predict(df_pred)[0])
        preds.append(max(0.0, y_hat))  # no negatives

        # Shift lags: lag_k <- previous lag_{k-1}; new lag_1 = y_hat
        for k in range(12, 1, -1):
            lk, lk1 = f"lag_{k}", f"lag_{k-1}"
            if lk in last_row.index and lk1 in last_row.index:
                last_row[lk] = last_row[lk1]
        if "lag_1" in last_row.index:
            last_row["lag_1"] = y_hat

    return pd.Series(preds, index=future_idx)

def choose_champion(y: pd.Series, exog: pd.DataFrame | None, horizon: int):
    """Bake-off across ARIMA, SARIMA, ARIMAX (+ small ML set). Returns a dict with champion and forecast."""
    y = y.asfreq("M").fillna(0)

    # If history short, shrink horizon
    effective_h = min(horizon, max(3, int(len(y) * 0.25)))  # keep test >=3 points
    if len(y) < 18:
        effective_h = max(3, min(horizon, len(y)//3 if len(y)//3 > 0 else 3))

    train, test = y[:-effective_h], y[-effective_h:]
    results = []

    # --- ARIMA
    orders = [(p, d, q) for p in [0, 1, 2] for d in [0, 1] for q in [0, 1, 2]]
    a_rmse, a_order = arima_grid(train, test, orders)
    results.append(("ARIMA", a_rmse, {"order": a_order}))

    # --- SARIMA (S=12)
    pdq_list = [(p, d, q) for p in [0, 1] for d in [0, 1] for q in [0, 1]]
    spdq_list = [(P, D, Q, 12) for P in [0, 1] for D in [0, 1] for Q in [0, 1]]
    s_rmse, s_order, s_sorder = sarima_grid(train, test, pdq_list, spdq_list)
    results.append(("SARIMA", s_rmse, {"order": s_order, "seasonal_order": s_sorder}))

    # --- ARIMAX if exog available
    if exog is not None and not exog.empty:
        ex_train, ex_test = exog.loc[train.index], exog.loc[test.index]
        ax_rmse, ax_order = arimax_grid(train, test, ex_train, ex_test, orders=[(p, d, q) for p in [0,1] for d in [0,1] for q in [0,1]])
        results.append(("ARIMAX", ax_rmse, {"order": ax_order}))

    # --- ML small set
    try:
        ml_rmse, ml_name, ml_model, ml_feats = evaluate_ml(train, test)
        results.append((f"ML - {ml_name}", ml_rmse, {"model": ml_model, "features": ml_feats}))
    except Exception:
        pass

    # Pick champion
    results = [r for r in results if r[1] is not None and np.isfinite(r[1])]
    if not results:
        # Fallback naive
        start = y.index[-1] + pd.DateOffset(months=1)
        idx = pd.date_range(start=start, periods=horizon, freq="M")
        return {
            "name": "Naive (flat)",
            "details": {},
            "rmse": None,
            "forecast": pd.Series([float(y.iloc[-1])] * horizon, index=idx)
        }

    champ = min(results, key=lambda r: r[1])
    name, best_rmse, details = champ

    # Refit champion on full history and forecast H ahead
    if name == "ARIMA":
        m = ARIMA(y, order=details["order"]).fit()
        fc = m.forecast(steps=horizon)
        fc = fc.clip(lower=0)
        return {"name": name, "details": details, "rmse": best_rmse, "forecast": fc}

    if name == "SARIMA":
        m = SARIMAX(y, order=details["order"], seasonal_order=details["seasonal_order"],
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=horizon)
        fc = fc.clip(lower=0)
        return {"name": name, "details": details, "rmse": best_rmse, "forecast": fc}

    if name == "ARIMAX":
        ex_hist  = exog
        ex_fut   = make_exog_future(ex_hist, horizon)
        m = ARIMA(y, exog=ex_hist, order=details["order"]).fit()
        fc = m.forecast(steps=horizon, exog=ex_fut)
        fc = pd.Series(fc, index=ex_fut.index).clip(lower=0)
        return {"name": name, "details": details, "rmse": best_rmse, "forecast": fc}

    if name.startswith("ML - "):
        ml_model  = details["model"]
        feat_cols = details["features"]
        fc = iterative_ml_forecast(ml_model, feat_cols, y, horizon)
        return {"name": name, "details": {"features": feat_cols}, "rmse": best_rmse, "forecast": fc}

    # Fallback (shouldn't hit)
    start = y.index[-1] + pd.DateOffset(months=1)
    idx = pd.date_range(start=start, periods=horizon, freq="M")
    return {"name": "Naive (flat)", "details": {}, "rmse": best_rmse, "forecast": pd.Series([float(y.iloc[-1])] * horizon, index=idx)}

def line_plot(history: pd.Series, forecast: pd.Series, title: str, y_title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="History"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast", line=dict(dash="dot")))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
        height=380
    )
    return fig

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("📈 Forecasting")

df = load_data()
min_dt = df["__date_idx__"].min()
max_dt = df["__date_idx__"].max()

with st.container():
    c1, c2, c3, c4 = st.columns([2,2,2,2])
    with c1:
        H = st.number_input("Forecast horizon (months)", min_value=3, max_value=36, value=12, step=1)
    with c2:
        top_n = st.number_input("Top N vehicle groups", min_value=1, max_value=12, value=5, step=1)
    with c3:
        st.write("History range")
        date_rng = st.date_input(
            "Filter history (optional)",
            value=(min_dt.date(), max_dt.date()),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key="fc_hist_range"
        )
    with c4:
        scale_rev = st.checkbox("Scale revenue by 100 (cents → units)", value=True)

# Apply history filter (if the user changed the dates)
if isinstance(date_rng, (list, tuple)) and len(date_rng) == 2:
    start_date, end_date = pd.to_datetime(date_rng[0]), pd.to_datetime(date_rng[1])
    df = df[(df["__date_idx__"] >= start_date) & (df["__date_idx__"] <= end_date)].copy()

if df.empty:
    st.info("No rows in the selected history range.")
    st.stop()

# ------------------------------------------------------------
# 1) Champion Forecast — Total Rentals
# ------------------------------------------------------------
st.markdown("### 1) Total Rentals — Champion Forecast")

y_rentals = (
    df.set_index("__date_idx__")
      .resample("M")[COL_ID]
      .count()
      .asfreq("M")
      .fillna(0)
)

exog = build_exog(df)
with st.spinner("Training models for total rentals…"):
    champ_r = choose_champion(y_rentals, exog, H)

# Display champion details (compact)
det_r = champ_r["details"]
det_str = ""
if champ_r["name"] in ("ARIMA", "ARIMAX"):
    det_str = f"order={det_r.get('order')}"
elif champ_r["name"] == "SARIMA":
    det_str = f"order={det_r.get('order')}, seasonal_order={det_r.get('seasonal_order')}"
elif champ_r["name"].startswith("ML - "):
    det_str = f"features={len(det_r.get('features', []))}"

mcol1, mcol2, mcol3 = st.columns([3,2,2])
mcol1.metric("Champion", champ_r["name"])
mcol2.metric("Validation RMSE", f"{champ_r['rmse']:.2f}" if champ_r["rmse"] is not None else "—")
mcol3.metric("Spec", det_str if det_str else "—")

fig_r = line_plot(y_rentals, champ_r["forecast"], "Total Rentals — Forecast", "Rentals")
st.plotly_chart(fig_r, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# 2) Champion Forecasts — Top Vehicle Groups (each on its own chart)
# ------------------------------------------------------------
st.markdown("### 2) Top Vehicle Groups — Champion Forecasts")

if COL_VEH in df.columns and df[COL_VEH].notna().sum() > 0:
    top_groups = (
        df[COL_VEH]
        .value_counts(dropna=True)
        .head(int(top_n))
        .index
        .tolist()
    )

    if len(top_groups) == 0:
        st.info("No vehicle groups found.")
    else:
        exog_hist = build_exog(df)  # same exog recipe per group (calendar + daily rate averages)

        for vg in top_groups:
            st.subheader(f"{vg}")

            y_vg = (
                df[df[COL_VEH] == vg]
                  .set_index("__date_idx__")
                  .resample("M")[COL_ID]
                  .count()
                  .asfreq("M")
                  .fillna(0)
            )

            if len(y_vg.dropna()) < 6:
                st.info("Not enough history to model this category.")
                continue

            with st.spinner(f"Training models for {vg}…"):
                champ_v = choose_champion(y_vg, exog_hist, H)

            det_v = champ_v["details"]
            det_str_v = ""
            if champ_v["name"] in ("ARIMA", "ARIMAX"):
                det_str_v = f"order={det_v.get('order')}"
            elif champ_v["name"] == "SARIMA":
                det_str_v = f"order={det_v.get('order')}, seasonal_order={det_v.get('seasonal_order')}"
            elif champ_v["name"].startswith("ML - "):
                det_str_v = f"features={len(det_v.get('features', []))}"

            c1, c2, c3 = st.columns([3,2,2])
            c1.metric("Champion", champ_v["name"])
            c2.metric("Validation RMSE", f"{champ_v['rmse']:.2f}" if champ_v["rmse"] is not None else "—")
            c3.metric("Spec", det_str_v if det_str_v else "—")

            fig_v = line_plot(y_vg, champ_v["forecast"], f"{vg} — Forecast", "Rentals")
            st.plotly_chart(fig_v, use_container_width=True)

else:
    st.info(f"Column '{COL_VEH}' not found in dataset.")

st.markdown("---")

# ------------------------------------------------------------
# 3) Champion Forecast — Total Revenue
# ------------------------------------------------------------
st.markdown("### 3) Total Revenue — Champion Forecast")

# Build revenue monthly series (sum), and optionally scale by 100
y_rev = (
    df.set_index("__date_idx__")[COL_REV]
      .astype(float)
      .resample("M")
      .sum()
      .asfreq("M")
      .fillna(0)
)

if scale_rev:
    y_rev = y_rev / 100.0

with st.spinner("Training models for total revenue…"):
    # Reuse same exog (month + Daily Rate)
    exog_rev = build_exog(df)
    champ_rev = choose_champion(y_rev, exog_rev, H)

det_re = champ_rev["details"]
det_str_re = ""
if champ_rev["name"] in ("ARIMA", "ARIMAX"):
    det_str_re = f"order={det_re.get('order')}"
elif champ_rev["name"] == "SARIMA":
    det_str_re = f"order={det_re.get('order')}, seasonal_order={det_re.get('seasonal_order')}"
elif champ_rev["name"].startswith("ML - "):
    det_str_re = f"features={len(det_re.get('features', []))}"

r1, r2, r3 = st.columns([3,2,2])
r1.metric("Champion", champ_rev["name"])
r2.metric("Validation RMSE", f"{champ_rev['rmse']:.2f}" if champ_rev["rmse"] is not None else "—")
r3.metric("Spec", det_str_re if det_str_re else "—")

fig_rev = line_plot(y_rev, champ_rev["forecast"], "Total Revenue — Forecast", "Revenue" + (" (scaled)" if scale_rev else ""))
st.plotly_chart(fig_rev, use_container_width=True)
