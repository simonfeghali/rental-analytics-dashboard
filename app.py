# app.py
# Streamlit Rental Analytics Dashboard
# Requirements (add these to requirements.txt):
# streamlit==1.37.1
# pandas>=2.0
# numpy
# plotly
# statsmodels
# scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# -------------------------------
# Config & constants
# -------------------------------
st.set_page_config(page_title="Rental Analytics", layout="wide")
DATA_PATH = "merged_df_further_cleaned.xlsx"  # <- per your instruction

CHECKOUT_DATE_COL   = "Checkout Date"
CHECKIN_DATE_COL    = "Checkin Date"
EXPECTED_DATE_COL   = "Expected Checkin Date"
DAYS_CHARGED_COL    = "Days Charged Count"
RENTAL_DAYS_COL     = "Rental Length Days"
RENTAL_HOURS_COL    = "Rental Length Hours"
CHECKOUT_TIME_COL   = "Checkout Time"
CHECKIN_TIME_COL    = "Checkin Time"
EXPECTED_TIME_COL   = "Expected Checkin Time"
BASE_PRICE_CANDS    = ["Net Time&Dist Amount", "Pre-Tax Rental Amount", "Base Rate Amount", "Daily Rate Amount", "Rate Amount"]
DISCOUNT_AMT_CANDS  = ["Discount Amount", "Promo Amount"]
DISCOUNT_PCT_CANDS  = ["Discount %"]
VEHICLE_CANDS       = ["Vehicle Group Rented", "Vehicle Category", "Car Group", "Car Class", "Vehicle Group Charged"]
LOCATION_CANDS      = ["Checkout Location", "Branch", "Checkout Location District", "Location"]
COUNTRY_CANDS       = ["Address Country Code", "Responsible Country Code", "Responsible Billing Country"]
BROKER_SIGNAL_CANDS = ["Commission Amount", "Travel Agent Prepay Tour Voucher Amount", "Used Tour Voucher Amount"]

GCC = {"AE","SA","QA","KW","OM","BH"}  # Gulf countries

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Normalize whitespace-only strings to NaN
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)

    # Parse dates (YYYYMMDD if ints/strings)
    for c in [CHECKOUT_DATE_COL, CHECKIN_DATE_COL, EXPECTED_DATE_COL]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str), format="%Y%m%d", errors="coerce")

    # Parse times (HHMM)
    def to_time(s):
        s = pd.Series(s).astype(str).str.zfill(4)
        return pd.to_datetime(s, format="%H%M", errors="coerce").dt.time

    for c in [CHECKOUT_TIME_COL, CHECKIN_TIME_COL, EXPECTED_TIME_COL]:
        if c in df.columns:
            df[c] = to_time(df[c])

    # Derived date index
    df["__date_idx__"] = pd.to_datetime(df.get(CHECKOUT_DATE_COL), errors="coerce")

    # Location, vehicle group, base price, discount fields
    loc_col = next((c for c in LOCATION_CANDS if c in df.columns), None)
    veh_col = next((c for c in VEHICLE_CANDS if c in df.columns), None)
    base_col = next((c for c in BASE_PRICE_CANDS if c in df.columns), None)
    disc_amt = next((c for c in DISCOUNT_AMT_CANDS if c in df.columns), None)
    disc_pct = next((c for c in DISCOUNT_PCT_CANDS if c in df.columns), None)

    df["__location__"]    = df[loc_col] if loc_col else pd.Series(["Unknown"]*len(df))
    df["__vehicle_cat__"] = df[veh_col] if veh_col else pd.Series(["Unknown"]*len(df))
    df["__base_amt__"]    = pd.to_numeric(df[base_col], errors="coerce") if base_col else np.nan
    df["__disc_amt__"]    = pd.to_numeric(df[disc_amt], errors="coerce") if disc_amt else np.nan

    # Discount rate (if % exists and may be scaled like 500=5.00%)
    if disc_pct and disc_pct in df.columns:
        pct = pd.to_numeric(df[disc_pct], errors="coerce")
        if pct.abs().max() > 100:
            df["__disc_rate__"] = pct / 10000.0
        else:
            df["__disc_rate__"] = pct / 100.0
    else:
        df["__disc_rate__"] = np.nan

    # Count key
    if "row_id_for_counts" not in df.columns:
        df["row_id_for_counts"] = range(1, len(df) + 1)

    # Seasonality flags
    df["checkout_year"]      = df["__date_idx__"].dt.year
    df["checkout_month_num"] = df["__date_idx__"].dt.month
    df["checkout_weekday"]   = df["__date_idx__"].dt.day_name()
    df["checkout_is_weekend"]= df["__date_idx__"].dt.weekday >= 5

    # Base price per day
    days = pd.to_numeric(df.get(DAYS_CHARGED_COL), errors="coerce")
    if days is not None and not days.isna().all():
        denom_days = days.replace(0, np.nan)
    else:
        d  = pd.to_numeric(df.get(RENTAL_DAYS_COL), errors="coerce").fillna(0)
        h  = pd.to_numeric(df.get(RENTAL_HOURS_COL), errors="coerce").fillna(0)
        denom_days = (d * 24 + h) / 24.0
        denom_days = denom_days.replace(0, np.nan)

    df["base_price_per_day"] = (df["__base_amt__"] / denom_days) / 100.0

    # Channel segmentation (Broker/Direct/Unknown)
    sigs = {c: pd.to_numeric(df.get(c), errors="coerce") for c in BROKER_SIGNAL_CANDS if c in df.columns}
    if sigs:
        commission = sigs.get("Commission Amount", pd.Series(np.nan, index=df.index))
        tav        = sigs.get("Travel Agent Prepay Tour Voucher Amount", pd.Series(np.nan, index=df.index))
        uav        = sigs.get("Used Tour Voucher Amount", pd.Series(np.nan, index=df.index))
        broker_mask = (commission.fillna(0) > 0) | (tav.fillna(0) > 0) | (uav.fillna(0) > 0)
        df["cust_channel"] = np.where(broker_mask, "Broker", "Direct")
        unknown_mask = commission.isna() & tav.isna() & uav.isna()
        df.loc[unknown_mask, "cust_channel"] = "Unknown"
    else:
        df["cust_channel"] = "Unknown"

    # Region segmentation
    country_col = next((c for c in COUNTRY_CANDS if c in df.columns), None)
    def region_from_country(x):
        if pd.isna(x): return "Unknown"
        s = str(x).strip().upper()
        if s in GCC: return "Gulf"
        if s in {"LB", "LEBANON"}: return "Local"
        return "Other"
    df["cust_region"] = df[country_col].apply(region_from_country) if country_col else "Unknown"

    return df

def safe_fig_show(fig):
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

def kpi_card(label, value, delta=None):
    st.metric(label, value, delta)

def empty_or_zero(s):
    return (s is None) or (isinstance(s, pd.Series) and s.empty)

# -------------------------------
# Load & prepare
# -------------------------------
df = load_data(DATA_PATH).copy()
df = df.dropna(subset=["__date_idx__"])
MIN_DT = df["__date_idx__"].min()
MAX_DT = df["__date_idx__"].max()

# Defaults for UI
DEFAULT_LOC   = sorted([x for x in df["__location__"].dropna().unique().tolist() if str(x) != "nan"])
DEFAULT_VEH   = sorted([x for x in df["__vehicle_cat__"].dropna().unique().tolist() if str(x) != "nan"])
DEFAULT_CHAN  = ["Direct", "Broker", "Unknown"]
DEFAULT_REG   = ["Local", "Gulf", "Other", "Unknown"]
DEFAULT_DATE  = (MIN_DT.date() if pd.notna(MIN_DT) else date(2019,1,1),
                 MAX_DT.date() if pd.notna(MAX_DT) else date.today())

# Session state init/reset
def init_state():
    if "defaults" not in st.session_state:
        st.session_state.defaults = dict(
            date=DEFAULT_DATE, loc=DEFAULT_LOC, veh=DEFAULT_VEH,
            chan=DEFAULT_CHAN, reg=DEFAULT_REG
        )
    for k, v in st.session_state.defaults.items():
        key = f"flt_{'date' if k=='date' else ('location' if k=='loc' else ('vehicle' if k=='veh' else ('channel' if k=='chan' else 'region')))}"
        if key not in st.session_state:
            st.session_state[key] = v

def reset_filters():
    st.session_state.flt_date = st.session_state.defaults["date"]
    st.session_state.flt_location = st.session_state.defaults["loc"]
    st.session_state.flt_vehicle = st.session_state.defaults["veh"]
    st.session_state.flt_channel = st.session_state.defaults["chan"]
    st.session_state.flt_region = st.session_state.defaults["reg"]
    st.rerun()

init_state()

# -------------------------------
# UI — Filters (main page)
# -------------------------------
st.title("Rental Analytics Dashboard")

with st.container():
    c1, c2, c3 = st.columns([1.2, 1, 1], vertical_alignment="top")

    with c1:
        st.markdown("**Date range (Checkout Date)**")
        st.session_state.flt_date = st.date_input(
            "date_range",
            value=st.session_state.flt_date,
            min_value=DEFAULT_DATE[0],
            max_value=DEFAULT_DATE[1],
            format="YYYY/MM/DD",
            label_visibility="collapsed",
        )

        st.markdown("**Locations**")
        st.session_state.flt_location = st.multiselect(
            "locations",
            options=DEFAULT_LOC,
            default=st.session_state.flt_location,
            label_visibility="collapsed",
        )

        st.markdown("**Vehicle Groups**")
        st.session_state.flt_vehicle = st.multiselect(
            "vehicle_groups",
            options=DEFAULT_VEH,
            default=st.session_state.flt_vehicle,
            label_visibility="collapsed",
        )

    with c2:
        st.markdown("**Channel**")
        st.session_state.flt_channel = st.multiselect(
            "channel",
            options=DEFAULT_CHAN,
            default=st.session_state.flt_channel,
            label_visibility="collapsed",
        )

        st.markdown("**Region**")
        st.session_state.flt_region = st.multiselect(
            "region",
            options=DEFAULT_REG,
            default=st.session_state.flt_region,
            label_visibility="collapsed",
        )

    with c3:
        st.write("")
        apply_clicked = st.button("Apply", type="primary", use_container_width=True)
        st.write("")
        reset_clicked = st.button("Reset filters", use_container_width=True)
        if reset_clicked:
            reset_filters()

# -------------------------------
# Apply filters (compare Timestamp to Timestamp)
# -------------------------------
date_from, date_to = st.session_state.flt_date
start_ts = pd.to_datetime(date_from)
end_ts   = pd.to_datetime(date_to)

mask = (
    (df["__date_idx__"] >= start_ts) &
    (df["__date_idx__"] <= end_ts)
)
if st.session_state.flt_location:
    mask &= df["__location__"].isin(st.session_state.flt_location)
if st.session_state.flt_vehicle:
    mask &= df["__vehicle_cat__"].isin(st.session_state.flt_vehicle)
if st.session_state.flt_channel:
    mask &= df["cust_channel"].isin(st.session_state.flt_channel)
if st.session_state.flt_region:
    mask &= df["cust_region"].isin(st.session_state.flt_region)

df_f = df[mask].copy()

# -------------------------------
# Executive KPIs
# -------------------------------
st.header("Executive KPIs")

total_rentals = int(df_f["row_id_for_counts"].count())
total_revenue = float((df_f["__base_amt__"].sum() or 0) / 100.0)
adr = float(df_f["base_price_per_day"].mean()) if not df_f["base_price_per_day"].dropna().empty else 0.0
avg_len_days = float(pd.to_numeric(df_f.get(RENTAL_DAYS_COL), errors="coerce").mean() or 0)

colk1, colk2, colk3, colk4 = st.columns(4)
with colk1: kpi_card("Total Rentals", f"{total_rentals:,}")
with colk2: kpi_card("Revenue (T&D)", f"{total_revenue:,.0f}")
with colk3: kpi_card("Avg Daily Rate", f"{adr:,.1f}")
with colk4: kpi_card("Avg Rental Length (days)", f"{avg_len_days:,.2f}")

# quick tops
top_loc = df_f["__location__"].value_counts()
top_loc = top_loc.idxmax() if not top_loc.empty else "—"
top_veh = df_f["__vehicle_cat__"].value_counts()
top_veh = top_veh.idxmax() if not top_veh.empty else "—"
colt1, colt2 = st.columns(2)
with colt1: st.caption(f"Top Location: **{top_loc}**")
with colt2: st.caption(f"Top Vehicle Group: **{top_veh}**")

st.divider()

# -------------------------------
# Demand & Seasonality
# -------------------------------
st.subheader("Demand & Seasonality")

# Rentals per month
rentals_per_month = (
    df_f.set_index("__date_idx__")
        .resample("M")["row_id_for_counts"]
        .count()
        .rename("rentals")
        .to_frame()
        .reset_index()
)
fig1 = px.line(rentals_per_month, x="__date_idx__", y="rentals", title="Rentals per Month")
fig1.update_layout(xaxis_title="Date", yaxis_title="Rentals")
safe_fig_show(fig1)

# Rentals per year
rentals_per_year = (
    df_f.assign(year=df_f["__date_idx__"].dt.year)
        .groupby("year", as_index=False)["row_id_for_counts"].count()
        .rename(columns={"row_id_for_counts":"rentals"})
)
fig_year = px.bar(rentals_per_year, x="year", y="rentals", title="Rentals per Year")
fig_year.update_layout(xaxis_title="Year", yaxis_title="Rentals")
safe_fig_show(fig_year)

# Seasonality (month of year)
seasonality_moy = (
    df_f.assign(m=df_f["__date_idx__"].dt.month)
        .groupby("m", as_index=False)["row_id_for_counts"].count()
        .rename(columns={"row_id_for_counts":"avg_rentals"})
)
fig_moy = px.bar(seasonality_moy, x="m", y="avg_rentals", title="Seasonality — Rentals by Month of Year")
fig_moy.update_layout(xaxis_title="Month", yaxis_title="Rentals")
safe_fig_show(fig_moy)

st.divider()

# -------------------------------
# Vehicle Mix
# -------------------------------
st.subheader("Vehicle Mix")

if "__vehicle_cat__" in df_f.columns:
    # Share by category
    cat_counts = (df_f.groupby("__vehicle_cat__", dropna=False)["row_id_for_counts"]
                    .count().rename("rentals").reset_index())
    total = cat_counts["rentals"].sum()
    cat_counts["share"] = cat_counts["rentals"] / total if total else 0
    fig_cat = px.bar(cat_counts.sort_values("share", ascending=False),
                     x="__vehicle_cat__", y="share",
                     title="Rental Share per Vehicle Category",
                     labels={"__vehicle_cat__":"Vehicle Category", "share":"Share"})
    fig_cat.update_yaxes(tickformat=".0%")
    fig_cat.update_layout(xaxis={"categoryorder":"total descending"})
    safe_fig_show(fig_cat)

    # Top N vehicle groups time series
    rentals_month_vehicle = (
        df_f.groupby([pd.Grouper(key="__date_idx__", freq="M"), "__vehicle_cat__"])["row_id_for_counts"]
            .count().rename("rentals").reset_index()
    )
    top_n = 10
    top_veh_groups = (rentals_month_vehicle.groupby("__vehicle_cat__")["rentals"]
                        .sum().sort_values(ascending=False).head(top_n).index)
    pivot_top = (rentals_month_vehicle[rentals_month_vehicle["__vehicle_cat__"].isin(top_veh_groups)]
                    .pivot(index="__date_idx__", columns="__vehicle_cat__", values="rentals").fillna(0))
    if not pivot_top.empty:
        fig_veh_ts = px.line(pivot_top, title=f"Monthly Rentals for Top {top_n} Vehicle Groups")
        fig_veh_ts.update_layout(xaxis_title="Date", yaxis_title="Rentals")
        safe_fig_show(fig_veh_ts)

st.divider()

# -------------------------------
# Customer Segments
# -------------------------------
st.subheader("Customer Segments")

# Channel totals
channel_summary = (df_f.groupby("cust_channel")["row_id_for_counts"]
                   .count().rename("rentals").to_frame())
channel_summary["share"] = channel_summary["rentals"] / channel_summary["rentals"].sum() if channel_summary["rentals"].sum() else 0
fig_chan = px.bar(channel_summary.reset_index(), x="cust_channel", y="rentals",
                  title="Total Rentals by Channel", labels={"cust_channel":"Channel"})
fig_chan.update_layout(xaxis={'categoryorder':'total descending'}, yaxis_title="Rentals")
safe_fig_show(fig_chan)

# Region totals
region_summary = (df_f.groupby("cust_region")["row_id_for_counts"]
                  .count().rename("rentals").to_frame())
region_summary["share"] = region_summary["rentals"] / region_summary["rentals"].sum() if region_summary["rentals"].sum() else 0
fig_reg = px.bar(region_summary.reset_index(), x="cust_region", y="rentals",
                 title="Total Rentals by Region", labels={"cust_region":"Region"})
fig_reg.update_layout(xaxis={'categoryorder':'total descending'}, yaxis_title="Rentals")
safe_fig_show(fig_reg)

# Monthly TS by channel & region
monthly_channel = (df_f.groupby([pd.Grouper(key="__date_idx__", freq="M"), "cust_channel"])["row_id_for_counts"]
                      .count().reset_index())
pivot_channel = (monthly_channel.pivot(index="__date_idx__", columns="cust_channel", values="row_id_for_counts")
                                .sort_index().fillna(0))
if not pivot_channel.empty:
    fig_channel_ts = px.line(pivot_channel, title="Monthly Rentals by Channel")
    fig_channel_ts.update_layout(xaxis_title="Date", yaxis_title="Rentals")
    safe_fig_show(fig_channel_ts)

monthly_region = (df_f.groupby([pd.Grouper(key="__date_idx__", freq="M"), "cust_region"])["row_id_for_counts"]
                     .count().reset_index())
pivot_region = (monthly_region.pivot(index="__date_idx__", columns="cust_region", values="row_id_for_counts")
                              .sort_index().fillna(0))
if not pivot_region.empty:
    fig_region_ts = px.line(pivot_region, title="Monthly Rentals by Region")
    fig_region_ts.update_layout(xaxis_title="Date", yaxis_title="Rentals")
    safe_fig_show(fig_region_ts)

st.divider()

# -------------------------------
# Operational Patterns
# -------------------------------
st.subheader("Operational Patterns")

# Weekday vs weekend
weekday_counts = df_f["checkout_weekday"].value_counts().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
).fillna(0).reset_index()
weekday_counts.columns = ["weekday", "count"]
fig_wd = px.bar(weekday_counts, x="weekday", y="count", title="Rental Frequency by Checkout Weekday")
fig_wd.update_layout(xaxis_title="Weekday", yaxis_title="Rentals")
safe_fig_show(fig_wd)

weekend_counts = df_f["checkout_is_weekend"].value_counts().reindex([False, True]).fillna(0).reset_index()
weekend_counts.columns = ["is_weekend", "count"]
weekend_counts["label"] = weekend_counts["is_weekend"].map({False:"Weekday", True:"Weekend"})
fig_we = px.bar(weekend_counts, x="label", y="count", title="Rental Frequency: Weekend vs Weekday",
                labels={"label":"", "count":"Rentals"})
safe_fig_show(fig_we)

# Checkout time 3-hour bins
ct = df_f[CHECKOUT_TIME_COL].astype(str)
if ct.str.contains(":").any():
    hours = pd.to_datetime(ct, errors="coerce").dt.hour
else:
    hours = pd.to_datetime(ct.str.zfill(4), format="%H%M", errors="coerce").dt.hour
bins = list(range(0, 25, 3))
labels = [f"{i:02d}:00-{i+3:02d}:00" for i in bins[:-1]]
labels[-1] = "21:00-24:00"
cats = pd.cut(hours, bins=bins, right=False, include_lowest=True, labels=labels)
time_bin_counts = pd.value_counts(pd.Categorical(cats, categories=labels, ordered=True), sort=False).fillna(0).astype(int)
df_bins = pd.DataFrame({"Time Bin": labels, "Count": time_bin_counts.values})
fig_bins = px.bar(df_bins, x="Time Bin", y="Count", title="Rental Frequency by Checkout Time (3-hour bins)")
safe_fig_show(fig_bins)

# Rental length distributions
colh1, colh2, colh3 = st.columns(3)
with colh1:
    if RENTAL_DAYS_COL in df_f.columns:
        fig_days = px.histogram(df_f, x=RENTAL_DAYS_COL, title="Distribution: Rental Length (Days)")
        safe_fig_show(fig_days)
with colh2:
    if RENTAL_HOURS_COL in df_f.columns:
        fig_hours = px.histogram(df_f, x=RENTAL_HOURS_COL, title="Distribution: Rental Length (Hours)")
        safe_fig_show(fig_hours)
with colh3:
    if DAYS_CHARGED_COL in df_f.columns:
        fig_charged = px.histogram(df_f, x=DAYS_CHARGED_COL, title="Distribution: Days Charged")
        safe_fig_show(fig_charged)

st.divider()

# -------------------------------
# Pricing & Discounting
# -------------------------------
st.subheader("Pricing & Discounting")

# Monthly aggregates for price sensitivity
agg = (df_f.set_index("__date_idx__")
          .resample("M")
          .agg(rentals=("row_id_for_counts","count"),
               avg_base_price=("base_price_per_day","mean"),
               avg_discount_rate=("__disc_rate__","mean"))
          .reset_index())

fig_r = px.bar(agg, x="__date_idx__", y="rentals", title="Monthly Rentals")
fig_r.update_layout(xaxis_title="Date", yaxis_title="Rentals")
safe_fig_show(fig_r)

if "avg_base_price" in agg.columns:
    fig_bp = px.bar(agg, x="__date_idx__", y="avg_base_price",
                    title="Average Base Price per Day (Monthly)")
    fig_bp.update_layout(xaxis_title="Date", yaxis_title="Price per Day")
    safe_fig_show(fig_bp)

if "avg_discount_rate" in agg.columns:
    fig_dr = px.bar(agg, x="__date_idx__", y="avg_discount_rate",
                    title="Average Discount Rate (Monthly)")
    fig_dr.update_layout(xaxis_title="Date", yaxis_title="Discount Rate")
    safe_fig_show(fig_dr)

# Histograms
colp1, colp2 = st.columns(2)
with colp1:
    if "__disc_rate__" in df_f.columns and not df_f["__disc_rate__"].dropna().empty:
        fig_dh = px.histogram(df_f, x="__disc_rate__", title="Distribution of Discount Rate")
        fig_dh.update_layout(xaxis_title="Discount Rate", yaxis_title="Frequency")
        safe_fig_show(fig_dh)
with colp2:
    if "base_price_per_day" in df_f.columns and not df_f["base_price_per_day"].dropna().empty:
        fig_ph = px.histogram(df_f, x="base_price_per_day", title="Distribution of Base Price per Day")
        fig_ph.update_layout(xaxis_title="Price per Day", yaxis_title="Frequency")
        safe_fig_show(fig_ph)

# -------------------------------
# Footer
# -------------------------------
st.caption("Grouped: Executive KPIs • Demand & Seasonality • Vehicle Mix • Customer Segments • Operational Patterns • Pricing & Discounting")
