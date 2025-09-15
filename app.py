import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="Rental Analytics Dashboard", page_icon="🚗", layout="wide")
DATA_PATH = "merged_df_further_cleaned.xlsx"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_top_value(s: pd.Series, default="—"):
    if s is None:
        return default
    s = s.dropna()
    return default if s.empty else s.value_counts().idxmax()

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)

    # Dates
    if "Checkout Date" in df.columns:
        df["Checkout Date"] = pd.to_datetime(df["Checkout Date"], errors="coerce")
    if "Checkin Date" in df.columns:
        df["Checkin Date"] = pd.to_datetime(df["Checkin Date"], errors="coerce")

    df["__date_idx__"] = df["Checkout Date"]
    df["row_id_for_counts"] = range(1, len(df) + 1)

    # Location
    LOC_CANDS = ["Checkout Location", "Checkout Location code", "Branch", "Location"]
    loc_col = next((c for c in LOC_CANDS if c in df.columns), None)
    df["__location__"] = df[loc_col].fillna("Unknown") if loc_col else "Unknown"

    # Channel (Broker vs Direct via money signals)
    COMM = df.get("Commission Amount", pd.Series(np.nan, index=df.index))
    TRAV = df.get("Travel Agent Prepay Tour Voucher Amount", pd.Series(np.nan, index=df.index))
    USED = df.get("Used Tour Voucher Amount", pd.Series(np.nan, index=df.index))
    broker_mask = (COMM.fillna(0) > 0) | (TRAV.fillna(0) > 0) | (USED.fillna(0) > 0)
    df["cust_channel"] = np.where(broker_mask, "Broker", "Direct")

    # Region (Gulf vs Local vs Other)
    GCC = {"AE","SA","QA","KW","OM","BH"}
    country_col = next(
        (c for c in ["Address Country Code", "Responsible Country Code", "Responsible Billing Country"] if c in df.columns),
        None
    )
    def region_from_country(x):
        if pd.isna(x): return "Unknown"
        s = str(x).strip().upper()
        if s in GCC: return "Gulf"
        if s in {"LB", "LEBANON"}: return "Local"
        return "Other"
    df["cust_region"] = df[country_col].apply(region_from_country) if country_col else "Unknown"

    return df

df = load_data()

# ------------------------------------------------------------
# Defaults & session state for filters (use LIST, not tuple)
# ------------------------------------------------------------
MIN_DT = pd.to_datetime(df["__date_idx__"]).min()
MAX_DT = pd.to_datetime(df["__date_idx__"]).max()

def _init_state():
    if "flt_date" not in st.session_state:
        st.session_state.flt_date = [MIN_DT.date(), MAX_DT.date()]  # LIST
    if "flt_loc" not in st.session_state:
        st.session_state.flt_loc = []
    if "flt_channel" not in st.session_state:
        st.session_state.flt_channel = []
    if "flt_region" not in st.session_state:
        st.session_state.flt_region = []
    if "flt_vehicle" not in st.session_state:
        st.session_state.flt_vehicle = []

def _reset_filters():
    st.session_state.flt_date = [MIN_DT.date(), MAX_DT.date()]  # keep type = LIST
    st.session_state.flt_loc = []
    st.session_state.flt_channel = []
    st.session_state.flt_region = []
    st.session_state.flt_vehicle = []

_init_state()

# ------------------------------------------------------------
# Header & Filters (main page)
# ------------------------------------------------------------
st.title("🚗 Rental Analytics Dashboard")

with st.container():
    top_left, top_mid, top_right = st.columns([6, 3, 1])
    with top_left:
        st.subheader("Filters")

        c1, c2, c3 = st.columns([2,2,2])
        with c1:
            st.date_input(
                "Date range",
                value=st.session_state.flt_date,
                min_value=MIN_DT.date(),
                max_value=MAX_DT.date(),
                key="flt_date",
            )
        with c2:
            st.multiselect(
                "Location",
                options=sorted(df["__location__"].dropna().unique()),
                default=st.session_state.flt_loc,
                key="flt_loc",
            )
            st.multiselect(
                "Channel",
                options=sorted(df["cust_channel"].dropna().unique()),
                default=st.session_state.flt_channel,
                key="flt_channel",
            )
        with c3:
            st.multiselect(
                "Region",
                options=sorted(df["cust_region"].dropna().unique()),
                default=st.session_state.flt_region,
                key="flt_region",
            )
            st.multiselect(
                "Vehicle Group",
                options=sorted(df.get("Vehicle Group Rented", pd.Series(dtype=object)).dropna().unique()),
                default=st.session_state.flt_vehicle,
                key="flt_vehicle",
            )

    with top_right:
        st.write("")
        st.write("")
        st.button("🔄 Reset filters", use_container_width=True, on_click=_reset_filters)

# ------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------
date_start, date_end = st.session_state.flt_date
mask = df["__date_idx__"].between(pd.to_datetime(date_start), pd.to_datetime(date_end))

if st.session_state.flt_loc:
    mask &= df["__location__"].isin(st.session_state.flt_loc)
if st.session_state.flt_channel:
    mask &= df["cust_channel"].isin(st.session_state.flt_channel)
if st.session_state.flt_region:
    mask &= df["cust_region"].isin(st.session_state.flt_region)
if st.session_state.flt_vehicle and "Vehicle Group Rented" in df.columns:
    mask &= df["Vehicle Group Rented"].isin(st.session_state.flt_vehicle)

df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.info("No rows match the current filters.")
    st.stop()

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Total Rentals", f"{len(df_filtered):,}")

total_rev = df_filtered.get("Net Time&Dist Amount", pd.Series(dtype=float)).sum() / 100
k2.metric("Total Revenue", f"{total_rev:,.0f}")

if {"Days Charged Count", "Net Time&Dist Amount"}.issubset(df_filtered.columns):
    adr = (
        df_filtered["Net Time&Dist Amount"] /
        df_filtered["Days Charged Count"].replace(0, np.nan)
    ).mean() / 100
    k3.metric("Avg Daily Rate", f"{adr:,.0f}")
else:
    k3.metric("Avg Daily Rate", "—")

k4, k5, k6 = st.columns(3)
wkend_share = (df_filtered["Checkout Date"].dt.weekday >= 5).mean() * 100
k4.metric("Weekend Share", f"{wkend_share:,.1f}%")
k5.metric("Top Vehicle", safe_top_value(df_filtered.get("Vehicle Group Rented")))
k6.metric("Top Location", safe_top_value(df_filtered.get("__location__")))

st.markdown("---")

# ------------------------------------------------------------
# EDA
# ------------------------------------------------------------
st.header("📊 Exploratory Data Analysis")

# Rentals per Month
monthly = (
    df_filtered.dropna(subset=["__date_idx__"])
    .set_index("__date_idx__")
    .resample("M")["row_id_for_counts"]
    .count()
    .rename("rentals")
    .reset_index()
)

fig = px.line(monthly, x="__date_idx__", y="rentals", title="Rentals per Month")
fig.update_layout(xaxis_title="Date", yaxis_title="Rentals")
st.plotly_chart(fig, use_container_width=True)


# Rentals per Year (fixed)
yearly = (
    df_filtered.assign(year=df_filtered["__date_idx__"].dt.year)
    .groupby("year", dropna=False)["row_id_for_counts"]
    .count()
    .reset_index(name="rentals")
)
st.plotly_chart(
    px.bar(yearly, x="year", y="rentals", title="Rentals per Year"),
    use_container_width=True
)

# Seasonality (month of year) (fixed)
seasonality = (
    df_filtered.assign(month=df_filtered["__date_idx__"].dt.month)
    .groupby("month", dropna=False)["row_id_for_counts"]
    .count()
    .reset_index(name="rentals")
)
st.plotly_chart(
    px.bar(seasonality, x="month", y="rentals", title="Seasonality by Month"),
    use_container_width=True
)

# Vehicle mix
if "Vehicle Group Rented" in df_filtered.columns:
    vc = (
        df_filtered["Vehicle Group Rented"]
        .value_counts()
        .head(10)
        .rename_axis("vehicle_group")
        .reset_index(name="count")
    )
    st.plotly_chart(
        px.bar(vc, x="vehicle_group", y="count", title="Top Vehicle Groups"),
        use_container_width=True
    )

# Channel breakdown
ch = (
    df_filtered["cust_channel"].value_counts()
    .rename_axis("channel").reset_index(name="count")
)
st.plotly_chart(
    px.bar(ch, x="channel", y="count", title="Channel Breakdown"),
    use_container_width=True
)

# Region breakdown
rg = (
    df_filtered["cust_region"].value_counts()
    .rename_axis("region").reset_index(name="count")
)
st.plotly_chart(
    px.bar(rg, x="region", y="count", title="Region Breakdown"),
    use_container_width=True
)

# Weekday counts
wd = (
    df_filtered["Checkout Date"].dt.day_name().value_counts()
    .rename_axis("weekday").reset_index(name="count")
)
category_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
wd["weekday"] = pd.Categorical(wd["weekday"], categories=category_order, ordered=True)
wd = wd.sort_values("weekday")
st.plotly_chart(
    px.bar(wd, x="weekday", y="count", title="Rentals by Weekday"),
    use_container_width=True
)

# Rental length distributions
if "Rental Length Days" in df_filtered.columns:
    st.plotly_chart(
        px.histogram(df_filtered, x="Rental Length Days", title="Distribution of Rental Length (Days)"),
        use_container_width=True
    )
if "Rental Length Hours" in df_filtered.columns:
    st.plotly_chart(
        px.histogram(df_filtered, x="Rental Length Hours", title="Distribution of Rental Length (Hours)"),
        use_container_width=True
    )
if "Days Charged Count" in df_filtered.columns:
    st.plotly_chart(
        px.histogram(df_filtered, x="Days Charged Count", title="Distribution of Days Charged"),
        use_container_width=True
    )
