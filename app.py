# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Rental Analytics Dashboard", layout="wide")

DATA_PATH = "data/merged_df_further_cleaned.xlsx"

# --------------------
# Data loading & prep
# --------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Dates & times
    for col in ["Checkout Date", "Checkin Date", "Expected Checkin Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Ensure time-like columns are strings -> hour
    if "Checkout Time" in df.columns:
        ct = df["Checkout Time"].astype(str)
        if ct.str.contains(":").any():
            df["checkout_hour"] = pd.to_datetime(ct, errors="coerce").dt.hour
        else:
            df["checkout_hour"] = pd.to_datetime(ct.str.zfill(4), format="%H%M", errors="coerce").dt.hour

    # Row id for counts
    if "row_id_for_counts" not in df.columns:
        df["row_id_for_counts"] = range(1, len(df) + 1)

    # Date index
    df["__date_idx__"] = df.get("Checkout Date", pd.NaT)

    # Vehicle group
    VEHICLE_CAT = "Vehicle Group Rented" if "Vehicle Group Rented" in df.columns else None

    # Base price/day
    base_col = "Net Time&Dist Amount" if "Net Time&Dist Amount" in df.columns else None
    if base_col is not None:
        if "Days Charged Count" in df.columns:
            denom = pd.to_numeric(df["Days Charged Count"], errors="coerce").replace(0, np.nan)
            df["base_price_per_day"] = (pd.to_numeric(df[base_col], errors="coerce") / denom) / 100.0
        else:
            days = pd.to_numeric(df.get("Rental Length Days", 0), errors="coerce").fillna(0)
            hours = pd.to_numeric(df.get("Rental Length Hours", 0), errors="coerce").fillna(0)
            dur_days = (days * 24 + hours) / 24.0
            df["base_price_per_day"] = (pd.to_numeric(df[base_col], errors="coerce") / dur_days.replace(0, np.nan)) / 100.0

    # Discount %
    if "Discount %" in df.columns:
        m = pd.to_numeric(df["Discount %"], errors="coerce").abs().max()
        df["discount_rate"] = (pd.to_numeric(df["Discount %"], errors="coerce") / (10000.0 if (m is not None and m > 100) else 100.0))

    # Commission signals -> Broker vs Direct
    commission = pd.to_numeric(df.get("Commission Amount", np.nan), errors="coerce")
    trav_voucher = pd.to_numeric(df.get("Travel Agent Prepay Tour Voucher Amount", np.nan), errors="coerce")
    used_voucher = pd.to_numeric(df.get("Used Tour Voucher Amount", np.nan), errors="coerce")
    broker_mask = (commission.fillna(0) > 0) | (trav_voucher.fillna(0) > 0) | (used_voucher.fillna(0) > 0)
    df["cust_channel"] = np.where(broker_mask, "Broker", "Direct")
    unknown_mask = commission.isna() & trav_voucher.isna() & used_voucher.isna()
    df.loc[unknown_mask, "cust_channel"] = "Unknown"

    # Region flags (Gulf / Local / Other / Unknown)
    country_col = None
    for c in ["Address Country Code", "Responsible Country Code", "Responsible Billing Country", "Renter Country", "Billing Country"]:
        if c in df.columns:
            country_col = c
            break
    GCC = {"AE","SA","QA","KW","OM","BH"}
    def region_from_country(x):
        if pd.isna(x): return "Unknown"
        s = str(x).strip().upper()
        if s in GCC: return "Gulf"
        if s in {"LB", "LEBANON"}: return "Local"
        return "Other"
    df["cust_region"] = df[country_col].apply(region_from_country) if country_col else "Unknown"

    # Seasonal flags
    if "__date_idx__" in df.columns:
        d = df["__date_idx__"]
        df["checkout_year"] = d.dt.year
        df["checkout_month_num"] = d.dt.month
        df["checkout_month"] = d.dt.to_period("M")
        df["checkout_weekday"] = d.dt.day_name()
        df["checkout_is_weekend"] = d.dt.weekday >= 5
        # Summer
        df["is_summer"] = d.dt.month.isin([6,7,8])
        # Christmas/New Year (Dec15-Jan7)
        m = d.dt.month
        day = d.dt.day
        df["is_christmas_newyear"] = ((m==12) & (day>=15)) | ((m==1) & (day<=7))
        # Eid windows (approx 2019-2024)
        eid_ranges = [
            ("2019-06-03","2019-06-06"), ("2019-08-11","2019-08-14"),
            ("2020-05-24","2020-05-26"), ("2020-07-31","2020-08-03"),
            ("2021-05-13","2021-05-16"), ("2021-07-20","2021-07-23"),
            ("2022-05-02","2022-05-05"), ("2022-07-09","2022-07-12"),
            ("2023-04-20","2023-04-23"), ("2023-06-28","2023-07-01"),
            ("2024-04-09","2024-04-12"), ("2024-06-16","2024-06-19"),
        ]
        eid_mask = False
        for a,b in eid_ranges:
            eid_mask = eid_mask | ((d >= pd.to_datetime(a)) & (d <= pd.to_datetime(b)))
        df["is_eid"] = eid_mask

    # Location pick
    for cand in ["Checkout Location", "Branch", "Location", "Checkout Location District"]:
        if cand in df.columns:
            df["__location__"] = df[cand]
            break
    if "__location__" not in df.columns:
        df["__location__"] = "Unknown"

    # Vehicle cat
    df["__vehicle_cat__"] = df["Vehicle Group Rented"] if "Vehicle Group Rented" in df.columns else "Unknown"

    # Revenue column scaled to currency units
    if "Net Time&Dist Amount" in df.columns:
        df["__revenue__"] = pd.to_numeric(df["Net Time&Dist Amount"], errors="coerce") / 100.0

    return df

df_full = load_data(DATA_PATH)

# --------------------
# Sidebar filters
# --------------------
st.sidebar.header("Filters")

# Date range
min_date = pd.to_datetime(df_full["__date_idx__"].min())
max_date = pd.to_datetime(df_full["__date_idx__"].max())
date_range = st.sidebar.date_input(
    "Checkout Date Range",
    value=(min_date.date() if pd.notna(min_date) else None, max_date.date() if pd.notna(max_date) else None),
)

# Location filter
locations = sorted(df_full["__location__"].dropna().unique().tolist())
loc_sel = st.sidebar.multiselect("Location", options=locations, default=locations[: min(10, len(locations))])

# Vehicle group filter
veh_groups = sorted(df_full["__vehicle_cat__"].dropna().unique().tolist())
veh_sel = st.sidebar.multiselect("Vehicle Group", options=veh_groups, default=veh_groups[: min(10, len(veh_groups))])

# Channel & Region
channel_opts = ["Broker", "Direct", "Unknown"]
region_opts = ["Gulf", "Local", "Other", "Unknown"]
ch_sel = st.sidebar.multiselect("Channel", channel_opts, default=channel_opts)
rg_sel = st.sidebar.multiselect("Region", region_opts, default=region_opts)

# Apply filters
df = df_full.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["__date_idx__"] >= start) & (df["__date_idx__"] <= end)]
if loc_sel:
    df = df[df["__location__"].isin(loc_sel)]
if veh_sel:
    df = df[df["__vehicle_cat__"].isin(veh_sel)]
if ch_sel:
    df = df[df["cust_channel"].isin(ch_sel)]
if rg_sel:
    df = df[df["cust_region"].isin(rg_sel)]

# --------------------
# KPIs
# --------------------
st.title("📊 Rental Analytics Dashboard")

def safe_count(s): return int(pd.to_numeric(s, errors="coerce").count())

total_rentals = safe_count(df["row_id_for_counts"])
active_months = df["checkout_month"].nunique() if "checkout_month" in df.columns else 0
total_revenue = df["__revenue__"].sum() if "__revenue__" in df.columns else 0.0
adr = df["base_price_per_day"].mean() if "base_price_per_day" in df.columns else np.nan
weekend_share = (df["checkout_is_weekend"].mean() if "checkout_is_weekend" in df.columns else np.nan)
broker_share = (df["cust_channel"].eq("Broker").mean() if "cust_channel" in df.columns else np.nan)
gulf_share = (df["cust_region"].eq("Gulf").mean() if "cust_region" in df.columns else np.nan)
avg_days = pd.to_numeric(df.get("Rental Length Days", np.nan), errors="coerce").mean()
top_loc = df["__location__"].value_counts().idxtop() if not df["__location__"].empty else "—"
top_veh = df["__vehicle_cat__"].value_counts().idxtop() if not df["__vehicle_cat__"].empty else "—"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Rentals", f"{total_rentals:,}")
k2.metric("Total Revenue", f"{total_revenue:,.0f}")
k3.metric("ADR (Avg Base/Day)", "—" if pd.isna(adr) else f"{adr:,.2f}")
k4.metric("Active Months", f"{active_months}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Weekend Share", "—" if pd.isna(weekend_share) else f"{weekend_share:.0%}")
k6.metric("Broker Share", "—" if pd.isna(broker_share) else f"{broker_share:.0%}")
k7.metric("Gulf Share", "—" if pd.isna(gulf_share) else f"{gulf_share:.0%}")
k8.metric("Avg Rental Length (days)", "—" if pd.isna(avg_days) else f"{avg_days:.2f}")

k9, k10 = st.columns(2)
k9.metric("Top Location", top_loc)
k10.metric("Top Vehicle Group", top_veh)

st.divider()

# --------------------
# EDA Section
# --------------------
st.header("Exploratory Data Analysis")

# 1) Rentals per Month (overall)
if "__date_idx__" in df.columns:
    rentals_per_month = (
        df.dropna(subset=["__date_idx__"])
          .set_index("__date_idx__")
          .resample("M")["row_id_for_counts"]
          .count()
          .rename("rentals")
          .to_frame()
    )
    st.subheader("Rentals per Month (All Locations)")
    st.plotly_chart(px.line(rentals_per_month, y="rentals"), use_container_width=True)

# 2) Rentals per Year
if "checkout_year" in df.columns:
    rentals_per_year = (
        df.groupby("checkout_year", dropna=False)["row_id_for_counts"]
          .count().rename("rentals").reset_index()
    )
    st.subheader("Rentals per Year")
    st.plotly_chart(px.bar(rentals_per_year, x="checkout_year", y="rentals"), use_container_width=True)

# 3) Seasonality by Month-of-Year
if "__date_idx__" in df.columns:
    seasonality_moy = (
        df.assign(month=df["__date_idx__"].dt.month)
          .groupby("month")["row_id_for_counts"]
          .count().rename("avg_rentals").reset_index()
    )
    st.subheader("Seasonality — Rentals by Month of Year")
    st.plotly_chart(px.bar(seasonality_moy, x="month", y="avg_rentals"), use_container_width=True)

# 4) Holiday/Seasonal Shares
if {"is_eid","is_christmas_newyear","is_summer"}.issubset(df.columns):
    holiday_summary = pd.DataFrame({
        "Period": ["Eid","Christmas/New Year","Summer"],
        "share": [
            df["is_eid"].mean(),
            df["is_christmas_newyear"].mean(),
            df["is_summer"].mean()
        ]
    })
    st.subheader("Share of Rentals During Holiday/Seasonal Periods")
    st.plotly_chart(px.bar(holiday_summary, x="Period", y="share"), use_container_width=True)

# 5) Rentals by Location over Time (Top 5)
if "__location__" in df.columns and "__date_idx__" in df.columns:
    rentals_month_loc = (
        df.dropna(subset=["__date_idx__"])
          .groupby([pd.Grouper(key="__date_idx__", freq="M"), "__location__"])["row_id_for_counts"]
          .count().rename("rentals").reset_index()
    )
    top_locs = (rentals_month_loc.groupby("__location__")["rentals"].sum()
                              .sort_values(ascending=False).head(5).index)
    pivot_top = (rentals_month_loc[rentals_month_loc["__location__"].isin(top_locs)]
                 .pivot(index="__date_idx__", columns="__location__", values="rentals")
                 .fillna(0))
    st.subheader("Monthly Rentals — Top 5 Locations")
    st.plotly_chart(px.line(pivot_top), use_container_width=True)

# 6) Vehicle Category Utilization / Shares
if "__vehicle_cat__" in df.columns:
    cat_counts = (df.groupby("__vehicle_cat__", dropna=False)["row_id_for_counts"]
                    .count().rename("rentals").reset_index())
    cat_counts["__vehicle_cat__"] = cat_counts["__vehicle_cat__"].fillna("Unknown")
    cat_counts["rental_share"] = cat_counts["rentals"] / cat_counts["rentals"].sum()
    st.subheader("Rental Share per Vehicle Category")
    fig_cat = px.bar(cat_counts.sort_values("rental_share", ascending=False),
                     x="__vehicle_cat__", y="rental_share", labels={"__vehicle_cat__":"Vehicle Group"})
    fig_cat.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Top N time series
    top_n = st.slider("Top N Vehicle Groups (time series)", 3, 15, 10)
    rentals_month_vehicle = (
        df.groupby([pd.Grouper(key="__date_idx__", freq="M"), "__vehicle_cat__"])["row_id_for_counts"]
          .count().rename("rentals").reset_index()
    )
    top_vehicle_groups = (rentals_month_vehicle.groupby("__vehicle_cat__")["rentals"].sum()
                          .sort_values(ascending=False).head(top_n).index)
    pivot_top_vehicles = (rentals_month_vehicle[rentals_month_vehicle["__vehicle_cat__"].isin(top_vehicle_groups)]
                 .pivot(index="__date_idx__", columns="__vehicle_cat__", values="rentals")
                 .fillna(0))
    st.subheader(f"Monthly Rentals for Top {top_n} Vehicle Groups")
    st.plotly_chart(px.line(pivot_top_vehicles), use_container_width=True)

# 7) Weekday vs Weekend / Time-of-day bins
if "checkout_weekday" in df.columns:
    st.subheader("Rental Frequency by Checkout Weekday")
    weekday_counts = df["checkout_weekday"].value_counts().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    )
    st.plotly_chart(px.bar(weekday_counts, labels={"index":"Weekday","value":"Rentals"}), use_container_width=True)

if "checkout_is_weekend" in df.columns:
    st.subheader("Rental Frequency: Weekend vs Weekday")
    wk_df = df["checkout_is_weekend"].value_counts().rename_axis("is_weekend").reset_index(name="count")
    wk_df["label"] = wk_df["is_weekend"].map({True:"Weekend", False:"Weekday"})
    st.plotly_chart(px.bar(wk_df, x="label", y="count"), use_container_width=True)

if "checkout_hour" in df.columns:
    st.subheader("Rental Frequency by Checkout Time (3-hour bins)")
    bins = list(range(0, 25, 3))
    labels = [f"{i:02d}:00-{i+3:02d}:00" for i in bins[:-1]]
    labels[-1] = "21:00-24:00"
    cats = pd.cut(df["checkout_hour"], bins=bins, right=False, include_lowest=True, labels=labels)
    cats = pd.Categorical(cats, categories=labels, ordered=True)
    time_bins = pd.value_counts(cats, sort=False, dropna=False).reindex(labels).fillna(0).astype(int)
    st.plotly_chart(px.bar(time_bins, labels={"index":"Time Bin","value":"Rentals"}), use_container_width=True)

# 8) Rental length distributions
for col, title in [("Rental Length Days","Distribution of Rental Length (Days)"),
                   ("Rental Length Hours","Distribution of Rental Length (Hours)"),
                   ("Days Charged Count","Distribution of Days Charged")]:
    if col in df.columns:
        st.subheader(title)
        st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

# 9) Channel & Region splits and time series
if "cust_channel" in df.columns:
    st.subheader("Total Rentals by Channel (Broker vs Direct)")
    ch_sum = (df.groupby("cust_channel")["row_id_for_counts"].count().reset_index(name="rentals"))
    st.plotly_chart(px.bar(ch_sum, x="cust_channel", y="rentals"), use_container_width=True)

    st.subheader("Monthly Rentals by Channel — Time Series")
    monthly_channel = (df.groupby([pd.Grouper(key="__date_idx__", freq="M"), "cust_channel"])["row_id_for_counts"]
                         .count().reset_index())
    pivot_channel = (monthly_channel
        .pivot(index="__date_idx__", columns="cust_channel", values="row_id_for_counts")
        .sort_index().fillna(0))
    st.plotly_chart(px.line(pivot_channel), use_container_width=True)

if "cust_region" in df.columns:
    st.subheader("Total Rentals by Region (Gulf / Local / Other / Unknown)")
    rg_sum = (df.groupby("cust_region")["row_id_for_counts"].count().reset_index(name="rentals"))
    st.plotly_chart(px.bar(rg_sum, x="cust_region", y="rentals"), use_container_width=True)

    st.subheader("Monthly Rentals by Region — Time Series")
    monthly_region = (df.groupby([pd.Grouper(key="__date_idx__", freq="M"), "cust_region"])["row_id_for_counts"]
                        .count().reset_index())
    pivot_region = (monthly_region
        .pivot(index="__date_idx__", columns="cust_region", values="row_id_for_counts")
        .sort_index().fillna(0))
    st.plotly_chart(px.line(pivot_region), use_container_width=True)

# 10) Pricing & Discounts
if "base_price_per_day" in df.columns:
    st.subheader("Average Base Price per Day — Monthly")
    agg = (df.groupby(pd.Grouper(key="__date_idx__", freq="M"))
           .agg(avg_base_price=("base_price_per_day","mean"),
                rentals=("row_id_for_counts","count"))
           .reset_index())
    st.plotly_chart(px.bar(agg, x="__date_idx__", y="avg_base_price"), use_container_width=True)

if "discount_rate" in df.columns:
    st.subheader("Average Discount Rate — Monthly")
    agg2 = (df.groupby(pd.Grouper(key="__date_idx__", freq="M"))
            .agg(avg_discount_rate=("discount_rate","mean"))
            .reset_index())
    st.plotly_chart(px.bar(agg2, x="__date_idx__", y="avg_discount_rate"), use_container_width=True)

    st.subheader("Distribution of Discount Rate")
    st.plotly_chart(px.histogram(df, x="discount_rate"), use_container_width=True)

# 11) Revenue over time (if present)
if "__revenue__" in df.columns and "__date_idx__" in df.columns:
    st.subheader("Total Revenue — Monthly")
    rev_m = (df.set_index("__date_idx__").resample("M")["__revenue__"].sum().reset_index())
    st.plotly_chart(px.bar(rev_m, x="__date_idx__", y="__revenue__",
                           labels={"__revenue__":"Revenue"}), use_container_width=True)

st.caption("Tip: refine filters in the sidebar to slice KPIs & charts by date, location, vehicle group, channel, and region.")
