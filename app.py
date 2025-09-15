import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Rental Analytics Dashboard",
    page_icon="🚗",
    layout="wide",
)

DATA_PATH = "merged_df_further_cleaned.xlsx"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_top_value(s: pd.Series, default="—"):
    if s is None or s.empty:
        return default
    s = s.dropna()
    if s.empty:
        return default
    vc = s.value_counts(dropna=True)
    if vc.empty:
        return default
    return vc.idxmax()

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
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

    # Channel
    COMM = df.get("Commission Amount", pd.Series(np.nan, index=df.index))
    TRAV = df.get("Travel Agent Prepay Tour Voucher Amount", pd.Series(np.nan, index=df.index))
    USED = df.get("Used Tour Voucher Amount", pd.Series(np.nan, index=df.index))
    broker_mask = (COMM.fillna(0) > 0) | (TRAV.fillna(0) > 0) | (USED.fillna(0) > 0)
    df["cust_channel"] = np.where(broker_mask, "Broker", "Direct")

    # Region
    GCC = {"AE","SA","QA","KW","OM","BH"}
    country_col = next((c for c in ["Address Country Code", "Responsible Country Code", "Responsible Billing Country"] if c in df.columns), None)
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
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Filters")
min_date, max_date = df["__date_idx__"].min(), df["__date_idx__"].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date])

loc_filter = st.sidebar.multiselect("Location", options=df["__location__"].unique())
channel_filter = st.sidebar.multiselect("Channel", options=df["cust_channel"].unique())
region_filter = st.sidebar.multiselect("Region", options=df["cust_region"].unique())
vehicle_filter = st.sidebar.multiselect("Vehicle Group", options=df.get("Vehicle Group Rented", pd.Series()).dropna().unique())

# Apply filters
mask = (df["__date_idx__"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
if loc_filter:
    mask &= df["__location__"].isin(loc_filter)
if channel_filter:
    mask &= df["cust_channel"].isin(channel_filter)
if region_filter:
    mask &= df["cust_region"].isin(region_filter)
if vehicle_filter and "Vehicle Group Rented" in df.columns:
    mask &= df["Vehicle Group Rented"].isin(vehicle_filter)

df_filtered = df[mask]

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
st.title("🚗 Rental Analytics Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals", f"{len(df_filtered):,}")
col2.metric("Total Revenue", f"{df_filtered.get('Net Time&Dist Amount', pd.Series()).sum()/100:,.0f}")
if "Days Charged Count" in df_filtered.columns and "Net Time&Dist Amount" in df_filtered.columns:
    adr = (df_filtered["Net Time&Dist Amount"] / df_filtered["Days Charged Count"].replace(0,np.nan)).mean()/100
    col3.metric("Avg Daily Rate", f"{adr:,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Weekend Share", f"{(df_filtered['Checkout Date'].dt.weekday>=5).mean()*100:,.1f}%")
col5.metric("Top Vehicle", safe_top_value(df_filtered.get("Vehicle Group Rented")))
col6.metric("Top Location", safe_top_value(df_filtered.get("__location__")))

# ------------------------------------------------------------
# EDA Charts
# ------------------------------------------------------------
st.header("📊 Exploratory Data Analysis")

# Rentals per Month
monthly = df_filtered.dropna(subset=["__date_idx__"]).set_index("__date_idx__").resample("M")["row_id_for_counts"].count()
fig1 = px.line(monthly, y="row_id_for_counts", title="Rentals per Month")
st.plotly_chart(fig1, use_container_width=True)

# Rentals per Year
yearly = df_filtered.groupby(df_filtered["__date_idx__"].dt.year)["row_id_for_counts"].count()
fig2 = px.bar(yearly, y="row_id_for_counts", title="Rentals per Year")
st.plotly_chart(fig2, use_container_width=True)

# Seasonality
seasonality = df_filtered.groupby(df_filtered["__date_idx__"].dt.month)["row_id_for_counts"].count()
fig3 = px.bar(seasonality, y="row_id_for_counts", title="Seasonality by Month")
st.plotly_chart(fig3, use_container_width=True)

# Vehicle Groups
if "Vehicle Group Rented" in df_filtered.columns:
    vc = df_filtered["Vehicle Group Rented"].value_counts().head(10)
    fig4 = px.bar(vc, y=vc.values, x=vc.index, title="Top Vehicle Groups")
    st.plotly_chart(fig4, use_container_width=True)

# Channel vs Direct
fig5 = px.bar(df_filtered["cust_channel"].value_counts(), title="Channel Breakdown")
st.plotly_chart(fig5, use_container_width=True)

# Region
fig6 = px.bar(df_filtered["cust_region"].value_counts(), title="Region Breakdown")
st.plotly_chart(fig6, use_container_width=True)

# Weekday / Weekend
fig7 = px.bar(df_filtered["Checkout Date"].dt.day_name().value_counts(), title="Rentals by Weekday")
st.plotly_chart(fig7, use_container_width=True)

# Rental Lengths
if "Rental Length Days" in df_filtered.columns:
    fig8 = px.histogram(df_filtered, x="Rental Length Days", title="Distribution of Rental Length (Days)")
    st.plotly_chart(fig8, use_container_width=True)

