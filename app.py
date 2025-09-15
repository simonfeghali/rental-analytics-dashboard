# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Rental Analytics Dashboard", layout="wide")

# --- File Upload ---
st.sidebar.header("Upload Excel Files")
file1 = st.sidebar.file_uploader("Upload 19 20.xlsx", type=["xlsx"])
file2 = st.sidebar.file_uploader("Upload 22 23 24.xlsx", type=["xlsx"])

if file1 and file2:
    # Read all sheets from both files
    df_19_20 = pd.concat([pd.read_excel(file1, sheet_name=s) for s in pd.ExcelFile(file1).sheet_names], ignore_index=True)
    df_22_23_24 = pd.concat([pd.read_excel(file2, sheet_name=s) for s in pd.ExcelFile(file2).sheet_names], ignore_index=True)

    # Merge
    merged_df = pd.concat([df_19_20, df_22_23_24], ignore_index=True)

    st.success(f"Data loaded! Shape: {merged_df.shape}")

    # --- Cleaning (shortened from your notebook) ---
    for col in merged_df.select_dtypes(include='object').columns:
        merged_df[col] = merged_df[col].replace(r'^\s*$', np.nan, regex=True)

    # Drop columns with >20% NaN
    missing_perc = merged_df.isnull().mean() * 100
    merged_df = merged_df.drop(columns=missing_perc[missing_perc > 20].index)

    # Example: parse dates
    if "Checkout Date" in merged_df.columns:
        merged_df["Checkout Date"] = pd.to_datetime(merged_df["Checkout Date"], errors="coerce")
        merged_df["checkout_month"] = merged_df["Checkout Date"].dt.to_period("M")

    df = merged_df.copy()
    df["row_id_for_counts"] = range(1, len(df)+1)
    df["__date_idx__"] = df["Checkout Date"]

    # --- EDA Visuals ---
    st.header("📊 Exploratory Data Analysis")

    # Rentals per Month
    rentals_per_month = df.dropna(subset=["__date_idx__"]).set_index("__date_idx__").resample("M")["row_id_for_counts"].count()
    fig1 = px.line(rentals_per_month, y="row_id_for_counts", title="Rentals per Month")
    st.plotly_chart(fig1, use_container_width=True)

    # Rentals by Year
    rentals_per_year = df.groupby(df["Checkout Date"].dt.year)["row_id_for_counts"].count()
    fig2 = px.bar(rentals_per_year, y="row_id_for_counts", title="Rentals per Year")
    st.plotly_chart(fig2, use_container_width=True)

    # Seasonality
    seasonality = df.groupby(df["Checkout Date"].dt.month)["row_id_for_counts"].count()
    fig3 = px.bar(seasonality, y="row_id_for_counts", title="Seasonality by Month")
    st.plotly_chart(fig3, use_container_width=True)

    # Vehicle Categories
    if "Vehicle Group Rented" in df.columns:
        vehicle_counts = df["Vehicle Group Rented"].value_counts().head(10)
        fig4 = px.bar(vehicle_counts, y=vehicle_counts.values, x=vehicle_counts.index, title="Top Vehicle Groups")
        st.plotly_chart(fig4, use_container_width=True)

else:
    st.info("⬅️ Please upload both Excel files to begin.")
