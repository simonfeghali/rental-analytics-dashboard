# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Rental Analytics Dashboard", layout="wide")

# --- Load pre-cleaned dataset directly from repo ---
@st.cache_data
def load_data():
    df = pd.read_excel("merged_df_further_cleaned.xlsx")  # <-- keep file in repo under /data
    # Ensure datetime columns
    if "Checkout Date" in df.columns:
        df["Checkout Date"] = pd.to_datetime(df["Checkout Date"], errors="coerce")
        df["checkout_month"] = df["Checkout Date"].dt.to_period("M")
    df["row_id_for_counts"] = range(1, len(df)+1)
    df["__date_idx__"] = df["Checkout Date"]
    return df

df = load_data()

# --- KPIs ---
st.title("📊 Rental Analytics Dashboard")

total_rentals = len(df)
unique_vehicles = df["Vehicle Group Rented"].nunique() if "Vehicle Group Rented" in df else 0
date_range = f"{df['Checkout Date'].min().date()} → {df['Checkout Date'].max().date()}"

col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals", f"{total_rentals:,}")
col2.metric("Unique Vehicle Groups", unique_vehicles)
col3.metric("Date Range", date_range)

# --- EDA Charts ---
st.header("Exploratory Data Analysis")

# Rentals per Month
rentals_per_month = (
    df.dropna(subset=["__date_idx__"])
      .set_index("__date_idx__")
      .resample("M")["row_id_for_counts"].count()
)
fig1 = px.line(rentals_per_month, y="row_id_for_counts", title="Rentals per Month")
st.plotly_chart(fig1, use_container_width=True)

# Rentals per Year
rentals_per_year = df.groupby(df["Checkout Date"].dt.year)["row_id_for_counts"].count()
fig2 = px.bar(rentals_per_year, y="row_id_for_counts", title="Rentals per Year")
st.plotly_chart(fig2, use_container_width=True)

# Seasonality
seasonality = df.groupby(df["Checkout Date"].dt.month)["row_id_for_counts"].count()
fig3 = px.bar(seasonality, y="row_id_for_counts", title="Seasonality by Month")
st.plotly_chart(fig3, use_container_width=True)

# Vehicle Groups
if "Vehicle Group Rented" in df.columns:
    vehicle_counts = df["Vehicle Group Rented"].value_counts().head(10)
    fig4 = px.bar(vehicle_counts, y=vehicle_counts.values, x=vehicle_counts.index,
                  title="Top Vehicle Groups")
    st.plotly_chart(fig4, use_container_width=True)

