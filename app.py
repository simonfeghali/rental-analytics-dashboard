# app.py
# Rental Analytics Dashboard (Streamlit)
# Requirements (in requirements.txt): streamlit, pandas, numpy, plotly, openpyxl

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date

st.set_page_config(page_title="Rental Analytics", layout="wide")

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
DATA_PATH = "merged_df_further_cleaned.xlsx"  # <- per your request

LOCATION_CANDIDATES = [
    "Checkout Location", "Checkin Location", "Checkout Location District",
    "Branch", "Location"
]
VEHICLE_CAT_CANDS = [
    "Vehicle Group Rented", "Vehicle Category", "Car Group", "Car Class",
    "Make Model Code", "Vehicle Group Charged"
]
COUNTRY_CANDS = [
    "Address Country Code", "Responsible Country Code", "Responsible Billing Country",
    "Billing Country", "Renter Country", "Checkout Country"
]
BROKER_FLAG_CANDS = [
    "Broker Flag", "Is Broker", "Channel", "Booking Channel", "Rate Source", "Responsible"
]

GCC = {"AE", "SA", "QA", "KW", "OM", "BH"}  # Gulf countries

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data …")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Clean blanks -> NaN
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)

    # Dates
    for col in ["Checkout Date", "Checkin Date", "Expected Checkin Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")

    # Times -> keep as text for safety; we’ll parse hours when needed
    for col in ["Checkout Time", "Checkin Time", "Expected Checkin Time"]:
        if col in df.columns:
            s = df[col].astype(str).str.replace(".0", "", regex=False)
            s = np.where(s.str.contains(":"), s, s.str.zfill(4).str.replace(r"(\d{2})(\d{2})", r"\1:\2", regex=True))
            df[col] = s

    # Keys
    if "row_id_for_counts" not in df.columns:
        df["row_id_for_counts"] = range(1, len(df) + 1)

    # Derived date index
    df["__date_idx__"] = pd.to_datetime(df.get("Checkout Date"), errors="coerce")

    # Checkout year/month/weekday/weekend
    if "Checkout Date" in df.columns:
        dt = df["Checkout Date"]
        df["checkout_year"] = dt.dt.year
        df["checkout_month"] = dt.dt.month
        df["checkout_weekday"] = dt.dt.day_name()
        df["checkout_is_weekend"] = dt.dt.weekday >= 5

    # Location column detection
    loc_col = next((c for c in LOCATION_CANDIDATES if c in df.columns), None)
    df["__location__"] = df[loc_col] if loc_col else pd.Series(index=df.index, dtype="object")

    # Vehicle category detection
    veh_col = next((c for c in VEHICLE_CAT_CANDS if c in df.columns), None)
    df["__vehicle_cat__"] = df[veh_col] if veh_col else pd.Series(index=df.index, dtype="object")

    # Region detection
    country_col = next((c for c in COUNTRY_CANDS if c in df.columns), None)
    def classify_region(x):
        if pd.isna(x): return "Unknown"
        s = str(x).strip().upper()
        if s in GCC: return "Gulf"
        if s in {"LB", "LEBANON"}: return "Local"
        return "Other"
    df["cust_region"] = df[country_col].apply(classify_region) if country_col else "Unknown"

    # Broker vs Direct heuristic
    commission = pd.to_numeric(df.get("Commission Amount", np.nan), errors="coerce")
    trav_voucher = pd.to_numeric(df.get("Travel Agent Prepay Tour Voucher Amount", np.nan), errors="coerce")
    used_voucher = pd.to_numeric(df.get("Used Tour Voucher Amount", np.nan), errors="coerce")
    broker_mask = (pd.Series(commission).fillna(0) > 0) | (pd.Series(trav_voucher).fillna(0) > 0) | (pd.Series(used_voucher).fillna(0) > 0)
    cust_channel = np.where(broker_mask, "Broker", "Direct")
    unknown_mask = pd.Series(commission).isna() & pd.Series(trav_voucher).isna() & pd.Series(used_voucher).isna()
    cust_channel = np.where(unknown_mask, "Unknown", cust_channel)
    df["cust_channel"] = cust_channel

    # Pricing helpers
    if "Net Time&Dist Amount" in df.columns:
        # ADR (base price / day)
        denom = pd.to_numeric(df.get("Days Charged Count", np.nan), errors="coerce").replace(0, np.nan)
        df["base_price_per_day"] = pd.to_numeric(df["Net Time&Dist Amount"], errors="coerce") / denom / 100

    if "Discount %" in df.columns:
        dp = pd.to_numeric(df["Discount %"], errors="coerce")
        mx = dp.abs().max()
        df["discount_rate"] = dp / (10000.0 if (mx and mx > 100) else 100.0)
    else:
        df["discount_rate"] = np.nan

    return df

def month_resample_count(d):
    return (
        d.dropna(subset=["__date_idx__"])
         .set_index("__date_idx__")
         .resample("M")["row_id_for_counts"]
         .count()
         .rename("rentals")
         .reset_index()
    )

def month_resample_sum(d, col):
    return (
        d.dropna(subset=["__date_idx__"])
         .set_index("__date_idx__")
         .resample("M")[col]
         .sum()
         .rename("value")
         .reset_index()
    )

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error("Could not read the Excel file. Make sure it exists next to app.py and is named exactly `merged_df_further_cleaned.xlsx`.")
    st.stop()

if df["__date_idx__"].notna().any():
    MIN_DT = pd.to_datetime(df["__date_idx__"].min()).date()
    MAX_DT = pd.to_datetime(df["__date_idx__"].max()).date()
else:
    MIN_DT = date(2019, 1, 1)
    MAX_DT = date.today()

# Options for filters
loc_opts = sorted([x for x in df["__location__"].dropna().unique().tolist() if str(x).strip() != ""])
veh_opts = sorted([x for x in df["__vehicle_cat__"].dropna().unique().tolist() if str(x).strip() != ""])
chan_opts = ["Direct", "Broker", "Unknown"]
reg_opts  = ["Local", "Gulf", "Other", "Unknown"]

# ------------------------------------------------------------------
# SESSION DEFAULTS (kept stable to avoid Streamlit session errors)
# ------------------------------------------------------------------
DEFAULTS = {
    "flt_date": (MIN_DT, MAX_DT),
    "flt_location": loc_opts.copy(),         # start with all
    "flt_vehicle": veh_opts.copy(),
    "flt_channel": chan_opts.copy(),
    "flt_region": reg_opts.copy(),
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_filters():
    st.session_state["flt_date"] = (MIN_DT, MAX_DT)
    st.session_state["flt_location"] = loc_opts.copy()
    st.session_state["flt_vehicle"] = veh_opts.copy()
    st.session_state["flt_channel"] = chan_opts.copy()
    st.session_state["flt_region"] = reg_opts.copy()
    st.rerun()

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("🚗 Rental Analytics Dashboard")

# ------------------------------------------------------------------
# FILTER BAR (Main page, not sidebar)
# ------------------------------------------------------------------
with st.container():
    st.markdown("##### Filters")
    c1, c2, c3 = st.columns([1.3, 1.2, 1.2])
    c4, c5, c6 = st.columns([1.2, 1.0, 0.9])

    # Date range
    st.session_state.flt_date = c1.date_input(
        "Date range (Checkout Date)",
        value=st.session_state.flt_date,
        min_value=MIN_DT,
        max_value=MAX_DT,
        key="date_input_key"
    )
    # Location
    st.session_state.flt_location = c2.multiselect(
        "Locations",
        options=loc_opts,
        default=st.session_state.flt_location
    )
    # Vehicle category
    st.session_state.flt_vehicle = c3.multiselect(
        "Vehicle Groups",
        options=veh_opts,
        default=st.session_state.flt_vehicle
    )
    # Channel
    st.session_state.flt_channel = c4.multiselect(
        "Channel",
        options=chan_opts,
        default=st.session_state.flt_channel,
    )
    # Region
    st.session_state.flt_region = c5.multiselect(
        "Region",
        options=reg_opts,
        default=st.session_state.flt_region,
    )

    # Buttons
    apply_btn = c6.button("Apply", type="primary", use_container_width=True)
    reset_btn = c6.button("Reset filters", type="secondary", use_container_width=True)
    if reset_btn:
        reset_filters()

# ------------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------------
date_from, date_to = st.session_state.flt_date
mask = pd.Series(True, index=df.index)

if date_from and date_to:
    mask &= (df["__date_idx__"].dt.date >= date_from) & (df["__date_idx__"].dt.date <= date_to)

if st.session_state.flt_location:
    mask &= df["__location__"].isin(st.session_state.flt_location)

if st.session_state.flt_vehicle:
    mask &= df["__vehicle_cat__"].isin(st.session_state.flt_vehicle)

if st.session_state.flt_channel:
    mask &= pd.Series(df["cust_channel"]).isin(st.session_state.flt_channel)

if st.session_state.flt_region:
    mask &= pd.Series(df["cust_region"]).isin(st.session_state.flt_region)

df_filtered = df[mask].copy()

if df_filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# ------------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------------
st.divider()
st.subheader("📊 Overview KPIs")

total_rentals = int(df_filtered["row_id_for_counts"].count())
total_revenue = pd.to_numeric(df_filtered.get("Net Time&Dist Amount", 0), errors="coerce").sum() / 100
avg_daily_rate = df_filtered.get("base_price_per_day", pd.Series(np.nan)).mean()
avg_days = pd.to_numeric(df_filtered.get("Days Charged Count", np.nan), errors="coerce").mean()

top_loc = (
    df_filtered["__location__"].value_counts().idxmax() if df_filtered["__location__"].notna().any() else "—"
)
top_vehicle = (
    df_filtered["__vehicle_cat__"].value_counts().idxmax() if df_filtered["__vehicle_cat__"].notna().any() else "—"
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Rentals", f"{total_rentals:,}")
k2.metric("Revenue (sum)", f"{total_revenue:,.0f}")
k3.metric("Avg Daily Rate", "—" if pd.isna(avg_daily_rate) else f"{avg_daily_rate:,.2f}")
k4.metric("Avg Days / Rental", "—" if pd.isna(avg_days) else f"{avg_days:,.2f}")
k5.metric("Top Location", f"{top_loc}")
k6.metric("Top Vehicle Group", f"{top_vehicle}")

st.caption("KPIs computed on filtered data.")

# ------------------------------------------------------------------
# GROUPED TABS (business names)
# ------------------------------------------------------------------
tabs = st.tabs([
    "📈 Demand & Volume",
    "💰 Revenue & Pricing",
    "🧑‍🤝‍🧑 Customer & Region",
    "🚗 Fleet & Vehicle Mix",
    "🕒 Timing & Seasonality",
    "🏷️ Promotions & Discounts",
])

# ----------------- 📈 DEMAND & VOLUME -----------------
with tabs[0]:
    st.markdown("#### Demand & Volume")

    monthly = month_resample_count(df_filtered)
    fig = px.line(monthly, x="__date_idx__", y="rentals", title="Rentals per Month")
    fig.update_layout(xaxis_title="Date", yaxis_title="Rentals")
    st.plotly_chart(fig, use_container_width=True)

    yr = (
        df_filtered.assign(year=df_filtered["__date_idx__"].dt.year)
                   .groupby("year")["row_id_for_counts"].count()
                   .reset_index(name="rentals")
    )
    fig = px.bar(yr, x="year", y="rentals", title="Rentals per Year")
    fig.update_layout(xaxis_title="Year", yaxis_title="Rentals")
    st.plotly_chart(fig, use_container_width=True)

    # Monthly by top 5 locations (if any)
    if df_filtered["__location__"].notna().any():
        month_loc = (
            df_filtered.groupby([pd.Grouper(key="__date_idx__", freq="M"), "__location__"])["row_id_for_counts"]
            .count().rename("rentals").reset_index()
        )
        top5 = (month_loc.groupby("__location__")["rentals"].sum().sort_values(ascending=False).head(5).index)
        pivot_top = (
            month_loc[month_loc["__location__"].isin(top5)]
            .pivot(index="__date_idx__", columns="__location__", values="rentals")
            .fillna(0)
        )
        fig = px.line(pivot_top, title="Monthly Rentals — Top 5 Locations")
        fig.update_layout(xaxis_title="Date", yaxis_title="Rentals")
        st.plotly_chart(fig, use_container_width=True)

# ----------------- 💰 REVENUE & PRICING -----------------
with tabs[1]:
    st.markdown("#### Revenue & Pricing")

    if "Net Time&Dist Amount" in df_filtered.columns:
        rev_m = month_resample_sum(df_filtered, "Net Time&Dist Amount")
        rev_m["value"] = rev_m["value"] / 100
        fig = px.bar(rev_m, x="__date_idx__", y="value", title="Revenue per Month")
        fig.update_layout(xaxis_title="Date", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

    if "base_price_per_day" in df_filtered.columns and df_filtered["base_price_per_day"].notna().any():
        price_m = (
            df_filtered.set_index("__date_idx__")
                       .resample("M")["base_price_per_day"].mean()
                       .rename("avg_price").reset_index()
        )
        fig = px.line(price_m, x="__date_idx__", y="avg_price", title="Average Daily Rate Over Time")
        fig.update_layout(xaxis_title="Date", yaxis_title="Avg Daily Rate")
        st.plotly_chart(fig, use_container_width=True)

# ----------------- 🧑‍🤝‍🧑 CUSTOMER & REGION -----------------
with tabs[2]:
    st.markdown("#### Customer & Region")

    ch_sum = (
        df_filtered.groupby("cust_channel")["row_id_for_counts"]
                   .count().reset_index(name="rentals")
                   .sort_values("rentals", ascending=False)
    )
    fig = px.bar(ch_sum, x="cust_channel", y="rentals", title="Total Rentals by Channel")
    fig.update_layout(xaxis_title="Channel", yaxis_title="Rentals")
    st.plotly_chart(fig, use_container_width=True)

    r_sum = (
        df_filtered.groupby("cust_region")["row_id_for_counts"]
                   .count().reset_index(name="rentals")
                   .sort_values("rentals", ascending=False)
    )
    fig = px.bar(r_sum, x="cust_region", y="rentals", title="Total Rentals by Region")
    fig.update_layout(xaxis_title="Region", yaxis_title="Rentals")
    st.plotly_chart(fig, use_container_width=True)

    # Time series by region
    reg_month = (
        df_filtered.groupby([pd.Grouper(key="__date_idx__", freq="M"), "cust_region"])["row_id_for_counts"]
                   .count().reset_index()
    )
    pivot_reg = reg_month.pivot(index="__date_idx__", columns="cust_region", values="row_id_for_counts").fillna(0)
    if not pivot_reg.empty:
        fig = px.line(pivot_reg, title="Monthly Rentals by Region")
        fig.update_layout(xaxis_title="Date", yaxis_title="Rentals")
        st.plotly_chart(fig, use_container_width=True)

# ----------------- 🚗 FLEET & VEHICLE MIX -----------------
with tabs[3]:
    st.markdown("#### Fleet & Vehicle Mix")

    if df_filtered["__vehicle_cat__"].notna().any():
        cat = (
            df_filtered.groupby("__vehicle_cat__", dropna=False)["row_id_for_counts"]
                       .count().rename("rentals").reset_index()
        )
        cat["__vehicle_cat__"] = cat["__vehicle_cat__"].fillna("Unknown")
        cat["share"] = cat["rentals"] / cat["rentals"].sum()
        fig = px.bar(cat.sort_values("share", ascending=False), x="__vehicle_cat__", y="share",
                     title="Rental Share by Vehicle Category")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(xaxis_title="Vehicle Category", yaxis_title="Share of Rentals")
        st.plotly_chart(fig, use_container_width=True)

        # Top N categories over time
        topN = 10
        month_cat = (
            df_filtered.groupby([pd.Grouper(key="__date_idx__", freq="M"), "__vehicle_cat__"])["row_id_for_counts"]
                       .count().rename("rentals").reset_index()
        )
        leaders = month_cat.groupby("__vehicle_cat__")["rentals"].sum().nlargest(topN).index
        pivot_top = (
            month_cat[month_cat["__vehicle_cat__"].isin(leaders)]
            .pivot(index="__date_idx__", columns="__vehicle_cat__", values="rentals")
            .fillna(0)
        )
        fig = px.line(pivot_top, title=f"Monthly Rentals for Top {topN} Vehicle Groups")
        fig.update_layout(xaxis_title="Date", yaxis_title="Rentals")
        st.plotly_chart(fig, use_container_width=True)

# ----------------- 🕒 TIMING & SEASONALITY -----------------
with tabs[4]:
    st.markdown("#### Timing & Seasonality")

    # Weekday distribution
    if "checkout_weekday" in df_filtered.columns:
        wk = df_filtered["checkout_weekday"].value_counts().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ).reset_index()
        wk.columns = ["weekday", "count"]
        fig = px.bar(wk, x="weekday", y="count", title="Rental Frequency by Checkout Weekday")
        st.plotly_chart(fig, use_container_width=True)

    # Weekend vs weekday
    if "checkout_is_weekend" in df_filtered.columns:
        wke = df_filtered["checkout_is_weekend"].value_counts().rename_axis("is_weekend").reset_index(name="count")
        wke["label"] = np.where(wke["is_weekend"], "Weekend", "Weekday")
        fig = px.bar(wke, x="label", y="count", title="Weekend vs Weekday Rentals")
        st.plotly_chart(fig, use_container_width=True)

    # Time-of-day 3h bins
    if "Checkout Time" in df_filtered.columns:
        ct = df_filtered["Checkout Time"].astype(str)
        if ct.str.contains(":").any():
            hr = pd.to_datetime(ct, errors="coerce").dt.hour
        else:
            hr = pd.to_datetime(ct.str.zfill(4), format="%H%M", errors="coerce").dt.hour
        bins = list(range(0, 25, 3))
        labels = [f"{i:02d}:00-{i+3:02d}:00" for i in bins[:-1]]
        labels[-1] = "21:00-24:00"
        cats = pd.cut(hr, bins=bins, right=False, include_lowest=True, labels=labels)
        counts = pd.value_counts(pd.Categorical(cats, categories=labels, ordered=True), sort=False).reset_index()
        counts.columns = ["Time Bin", "Count"]
        fig = px.bar(counts, x="Time Bin", y="Count", title="Checkout Time Distribution (3-hour bins)")
        st.plotly_chart(fig, use_container_width=True)

    # Seasonality — month of year
    sm = (
        df_filtered.assign(month=df_filtered["__date_idx__"].dt.month)
                   .groupby("month")["row_id_for_counts"].count()
                   .rename("rentals").reset_index()
    )
    fig = px.bar(sm, x="month", y="rentals", title="Seasonality — Rentals by Month of Year")
    fig.update_layout(xaxis_title="Month", yaxis_title="Rentals")
    st.plotly_chart(fig, use_container_width=True)

    # Rental length histograms
    for col, ttl in [
        ("Rental Length Days", "Distribution of Rental Length (Days)"),
        ("Rental Length Hours", "Distribution of Rental Length (Hours)"),
        ("Days Charged Count", "Distribution of Days Charged"),
    ]:
        if col in df_filtered.columns:
            fig = px.histogram(df_filtered, x=col, title=ttl)
            st.plotly_chart(fig, use_container_width=True)

# ----------------- 🏷️ PROMOTIONS & DISCOUNTS -----------------
with tabs[5]:
    st.markdown("#### Promotions & Discounts")

    if "discount_rate" in df_filtered.columns and df_filtered["discount_rate"].notna().any():
        fig = px.histogram(df_filtered, x="discount_rate", nbins=40, title="Distribution of Discount Rate")
        fig.update_layout(xaxis_title="Discount Rate", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    disc_code_col = next((c for c in ["Discount Code","AWD/BCD","AWD","BCD","Promo Code"] if c in df_filtered.columns), None)
    if disc_code_col:
        top_codes = df_filtered[disc_code_col].value_counts().head(15).reset_index()
        top_codes.columns = [disc_code_col, "count"]
        fig = px.bar(top_codes, x=disc_code_col, y="count", title="Top Discount / Promo Codes")
        st.plotly_chart(fig, use_container_width=True)

st.caption("© Your company — Streamlit Dashboard")
