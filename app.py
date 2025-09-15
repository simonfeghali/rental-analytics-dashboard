# app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import date

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="Rental Analytics Dashboard", page_icon="🚗", layout="wide")
DATA_PATH = "merged_df_further_cleaned.xlsx"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_top_value(s: pd.Series | None, default: str = "—"):
    if s is None:
        return default
    s = s.dropna()
    return default if s.empty else s.value_counts().idxmax()

def parse_checkout_hour(series: pd.Series) -> pd.Series:
    """Robustly extract hour from Checkout Time stored as 'HHMM', 'HH:MM[:SS]', python time, or number."""
    if series.dtype == "O":
        s = series.astype(str).str.strip()
        if s.str.contains(":").any():
            return pd.to_datetime(s, errors="coerce").dt.hour
        # numeric as 'HHMM'
        s = s.str.replace(r"\.0+$", "", regex=True).str.zfill(4)
        return pd.to_datetime(s, format="%H%M", errors="coerce").dt.hour
    try:
        # python time dtype
        return pd.to_datetime(series.astype(str), errors="coerce").dt.hour
    except Exception:
        return pd.to_datetime(series, errors="coerce").dt.hour

def compute_discount_rate(df: pd.DataFrame) -> pd.Series:
    """Return discount rate between 0 and 1 if possible."""
    if "Discount %" in df.columns:
        pct = pd.to_numeric(df["Discount %"], errors="coerce")
        return np.where(pct.abs().max() > 100, pct / 10000.0, pct / 100.0)
    if {"Discount Amount", "Net Time&Dist Amount"}.issubset(df.columns):
        num = pd.to_numeric(df["Discount Amount"], errors="coerce")
        den = pd.to_numeric(df["Net Time&Dist Amount"], errors="coerce").replace(0, np.nan)
        return (num / den).clip(lower=0, upper=1)
    return pd.Series(np.nan, index=df.index)

def bar_or_kpi(series: pd.Series, title: str, top_n: int | None = None,
               order: list[str] | None = None, unit: str = "rentals"):
    """
    Render a bar chart for a categorical series unless only one category remains.
    Then render a KPI showing that category and its share.
    """
    s = series.dropna()
    if s.nunique() <= 1:
        if s.empty:
            st.info(f"No data for {title} with current filters.")
            return
        cat = str(s.iloc[0]) if s.nunique() == 0 else str(s.mode(dropna=True).iloc[0])
        cnt = int(len(s))
        share = (cnt / len(series.dropna())) * 100 if len(series.dropna()) else 0.0
        st.metric(f"{title} — {cat}", f"{cnt:,} {unit} ({share:.1f}%)")
        return

    vc = s.value_counts()
    if top_n:
        vc = vc.head(top_n)
    df_plot = vc.rename_axis("category").reset_index(name="count")
    if order:
        df_plot["category"] = pd.Categorical(df_plot["category"], categories=order, ordered=True)
        df_plot = df_plot.sort_values("category")
    fig = px.bar(df_plot, x="category", y="count", title=title)
    fig.update_layout(xaxis_title=series.name or "category", yaxis_title="count")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)

    # Dates
    if "Checkout Date" in df.columns:
        df["Checkout Date"] = pd.to_datetime(df["Checkout Date"], errors="coerce")
    if "Checkin Date" in df.columns:
        df["Checkin Date"] = pd.to_datetime(df["Checkin Date"], errors="coerce")
    df["__date_idx__"] = df["Checkout Date"]
    df["row_id_for_counts"] = range(1, len(df) + 1)

    # Location (explicit new name first)
    LOC_CANDS = [
        "Checkout Location Name",  # <-- your note
        "Checkout Location",
        "Checkout Location code",
        "Branch",
        "Location",
    ]
    loc_col = next((c for c in LOC_CANDS if c in df.columns), None)
    df["__location__"] = df[loc_col].fillna("Unknown") if loc_col else "Unknown"

    # Vehicle column
    VEH_CANDS = ["Vehicle Group Rented", "Vehicle Group Charged", "Vehicle Category", "Car Group", "Car Class"]
    veh_col = next((c for c in VEH_CANDS if c in df.columns), None)
    df["__vehicle__"] = df[veh_col].fillna("Unknown") if veh_col else "Unknown"

    # Channel (Broker vs Direct via monetary signals)
    COMM = pd.to_numeric(df.get("Commission Amount"), errors="coerce")
    TRAV = pd.to_numeric(df.get("Travel Agent Prepay Tour Voucher Amount"), errors="coerce")
    USED = pd.to_numeric(df.get("Used Tour Voucher Amount"), errors="coerce")
    broker_mask = (COMM.fillna(0) > 0) | (TRAV.fillna(0) > 0) | (USED.fillna(0) > 0)
    df["cust_channel"] = np.where(broker_mask, "Broker", "Direct")
    unknown_mask = COMM.isna() & TRAV.isna() & USED.isna()
    df.loc[unknown_mask, "cust_channel"] = "Unknown"

    # Region (Gulf vs Local vs Other)
    GCC = {"AE", "SA", "QA", "KW", "OM", "BH"}
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

    # Seasonal flags
    d = df["__date_idx__"]
    df["is_summer"] = d.dt.month.isin([6, 7, 8])
    df["is_christmas_newyear"] = ((d.dt.month == 12) & (d.dt.day >= 15)) | ((d.dt.month == 1) & (d.dt.day <= 7))
    eid_ranges = [
        ("2019-06-03","2019-06-06"), ("2019-08-11","2019-08-14"),
        ("2020-05-24","2020-05-26"), ("2020-07-31","2020-08-03"),
        ("2021-05-13","2021-05-16"), ("2021-07-20","2021-07-23"),
        ("2022-05-02","2022-05-05"), ("2022-07-09","2022-07-12"),
        ("2023-04-20","2023-04-23"), ("2023-06-28","2023-07-01"),
        ("2024-04-09","2024-04-12"), ("2024-06-16","2024-06-19"),
    ]
    eid_mask = pd.Series(False, index=df.index)
    for a, b in eid_ranges:
        eid_mask = eid_mask | ((d >= pd.to_datetime(a)) & (d <= pd.to_datetime(b)))
    df["is_eid"] = eid_mask

    # Pricing features
    df["discount_rate"] = compute_discount_rate(df)
    if "Net Time&Dist Amount" in df.columns:
        den = pd.to_numeric(df.get("Days Charged Count"), errors="coerce").replace(0, np.nan)
        base = pd.to_numeric(df["Net Time&Dist Amount"], errors="coerce")
        df["base_price_per_day"] = (base / den) / 100.0

    return df

# ------------------------------------------------------------
# Data & defaults
# ------------------------------------------------------------
df = load_data()
MIN_DT = pd.to_datetime(df["__date_idx__"]).min().date()
MAX_DT = pd.to_datetime(df["__date_idx__"]).max().date()
VEHICLE_COL_EXISTS = "__vehicle__" in df.columns

def _init_state():
    if "flt_date" not in st.session_state:
        st.session_state.flt_date = [MIN_DT, MAX_DT]  # list
    if "flt_loc" not in st.session_state:
        st.session_state.flt_loc = []
    if "flt_channel" not in st.session_state:
        st.session_state.flt_channel = []
    if "flt_region" not in st.session_state:
        st.session_state.flt_region = []
    if "flt_vehicle" not in st.session_state:
        st.session_state.flt_vehicle = []

def _reset_filters():
    st.session_state.flt_date = [MIN_DT, MAX_DT]
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
    left, mid, right = st.columns([6, 3, 1])

    with left:
        st.subheader("Filters")
        c1, c2, c3 = st.columns([2, 2, 2])

        with c1:
            st.date_input(
                "Date range (Checkout Date)",
                value=st.session_state.flt_date,
                min_value=MIN_DT,
                max_value=MAX_DT,
                key="flt_date",
            )

        with c2:
            st.multiselect(
                "Locations",
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
            if VEHICLE_COL_EXISTS:
                st.multiselect(
                    "Vehicle Groups",
                    options=sorted(df["__vehicle__"].dropna().unique()),
                    default=st.session_state.flt_vehicle,
                    key="flt_vehicle",
                )

    with right:
        st.write("")
        st.write("")
        st.button("🔄 Reset filters", use_container_width=True, on_click=_reset_filters)

# ------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------
date_start, date_end = st.session_state.flt_date
date_start = pd.to_datetime(date_start)
date_end = pd.to_datetime(date_end)

mask = df["__date_idx__"].between(date_start, date_end, inclusive="both")
if st.session_state.flt_loc:
    mask &= df["__location__"].isin(st.session_state.flt_loc)
if st.session_state.flt_channel:
    mask &= df["cust_channel"].isin(st.session_state.flt_channel)
if st.session_state.flt_region:
    mask &= df["cust_region"].isin(st.session_state.flt_region)
if VEHICLE_COL_EXISTS and st.session_state.flt_vehicle:
    mask &= df["__vehicle__"].isin(st.session_state.flt_vehicle)

df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.info("No rows match the current filters.")
    st.stop()

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Total Rentals", f"{len(df_filtered):,}")

total_rev = pd.to_numeric(df_filtered.get("Net Time&Dist Amount"), errors="coerce").sum() / 100.0
k2.metric("Total Revenue", f"{total_rev:,.0f}")

if {"Days Charged Count", "Net Time&Dist Amount"}.issubset(df_filtered.columns):
    adr = (
        pd.to_numeric(df_filtered["Net Time&Dist Amount"], errors="coerce") /
        pd.to_numeric(df_filtered["Days Charged Count"], errors="coerce").replace(0, np.nan)
    ).mean() / 100.0
    k3.metric("Avg Daily Rate (ADR)", f"{adr:,.0f}")
else:
    k3.metric("Avg Daily Rate (ADR)", "—")

k4, k5, k6 = st.columns(3)
wkend_share = (df_filtered["Checkout Date"].dt.weekday >= 5).mean() * 100
k4.metric("Weekend Share", f"{wkend_share:,.1f}%")
k5.metric("Top Vehicle", safe_top_value(df_filtered.get("__vehicle__")))
k6.metric("Top Location", safe_top_value(df_filtered.get("__location__")))

st.markdown("---")

# ------------------------------------------------------------
# TABS (business groupings)
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand & Seasonality",
    "🧑‍🤝‍🧑 Customer Mix",
    "🚘 Fleet Mix",
    "🕒 Time Patterns",
    "💵 Price & Discounting",
])

# ------------------------------------------------------------
# TAB 1 — Demand & Seasonality
# ------------------------------------------------------------
with tab1:
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

    # Rentals per Year
    yearly = (
        df_filtered.assign(year=df_filtered["__date_idx__"].dt.year)
        .groupby("year", dropna=False)["row_id_for_counts"]
        .count()
        .reset_index(name="rentals")
    )
    st.plotly_chart(px.bar(yearly, x="year", y="rentals", title="Rentals per Year"),
                    use_container_width=True)

    # Seasonality (month of year)
    seasonality = (
        df_filtered.assign(month=df_filtered["__date_idx__"].dt.month)
        .groupby("month", dropna=False)["row_id_for_counts"]
        .count()
        .reset_index(name="rentals")
    )
    st.plotly_chart(px.bar(seasonality, x="month", y="rentals", title="Seasonality by Month"),
                    use_container_width=True)

    # Holiday / Seasonal period shares
    def share(mask: pd.Series) -> float:
        return (mask.sum() / len(mask)) if len(mask) else np.nan
    holiday_summary = pd.DataFrame({
        "Period": ["share_eid", "share_christmas_newyear", "share_summer"],
        "share": [
            share(df_filtered["is_eid"]),
            share(df_filtered["is_christmas_newyear"]),
            share(df_filtered["is_summer"]),
        ],
    })
    fig_h = px.bar(holiday_summary, x="Period", y="share",
                   title="Share of Rentals During Holiday/Seasonal Periods")
    fig_h.update_layout(xaxis_title="Period", yaxis_title="Share of Rentals")
    fig_h.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_h, use_container_width=True)

# ------------------------------------------------------------
# TAB 2 — Customer Mix
# ------------------------------------------------------------
with tab2:
    bar_or_kpi(df_filtered["cust_channel"], "Channel Breakdown")
    bar_or_kpi(df_filtered["cust_region"], "Region Breakdown")
    # Top 10 locations (or KPI)
    bar_or_kpi(df_filtered["__location__"], "Top Locations", top_n=10)

# ------------------------------------------------------------
# TAB 3 — Fleet Mix
# ------------------------------------------------------------
with tab3:
    if VEHICLE_COL_EXISTS:
        bar_or_kpi(df_filtered["__vehicle__"], "Top Vehicle Groups", top_n=10)

        # Monthly rentals for top 5 vehicle groups (if enough categories)
        top5 = (
            df_filtered["__vehicle__"]
            .value_counts()
            .head(5)
            .index.tolist()
        )
        if len(top5) > 1:
            rentals_month_vehicle = (
                df_filtered[df_filtered["__vehicle__"].isin(top5)]
                .groupby([pd.Grouper(key="__date_idx__", freq="M"), "__vehicle__"])["row_id_for_counts"]
                .count()
                .rename("rentals")
                .reset_index()
            )
            pivot_top = rentals_month_vehicle.pivot(index="__date_idx__", columns="__vehicle__", values="rentals").fillna(0)
            fig_v = px.line(pivot_top, title="Monthly Rentals — Top 5 Vehicle Groups")
            fig_v.update_layout(xaxis_title="Date", yaxis_title="Rentals")
            st.plotly_chart(fig_v, use_container_width=True)

# ------------------------------------------------------------
# TAB 4 — Time Patterns
# ------------------------------------------------------------
with tab4:
    # Weekend vs Weekday
    wk_label = np.where(df_filtered["Checkout Date"].dt.weekday >= 5, "Weekend", "Weekday")
    bar_or_kpi(pd.Series(wk_label, name="is_weekend"), "Rental Frequency: Weekend vs Weekday",
               order=["Weekday", "Weekend"])

    # Rentals by weekday (ordered)
    bar_or_kpi(
        df_filtered["Checkout Date"].dt.day_name(),
        "Rentals by Weekday",
        order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    # Checkout time bins (3-hour)
    if "Checkout Time" in df_filtered.columns:
        hours = parse_checkout_hour(df_filtered["Checkout Time"])
        bins = list(range(0, 25, 3))  # 0..24 step 3
        labels = [f"{i:02d}:00-{i+3:02d}:00" for i in bins[:-1]]
        labels[-1] = "21:00-24:00"
        cats = pd.cut(hours, bins=bins, right=False, include_lowest=True, labels=labels)
        cnt = pd.value_counts(pd.Categorical(cats, categories=labels, ordered=True), sort=False)
        time_bin_counts_df = pd.DataFrame({"Time Bin": labels, "Count": cnt.values})
        fig_tb = px.bar(time_bin_counts_df, x="Time Bin", y="Count",
                        title="Rental Frequency by Checkout Time (3-hour Bins)")
        fig_tb.update_layout(xaxis={"categoryorder": "array", "categoryarray": labels})
        st.plotly_chart(fig_tb, use_container_width=True)

    # Rental length distributions
    cols = st.columns(3)
    if "Rental Length Days" in df_filtered.columns:
        with cols[0]:
            st.plotly_chart(
                px.histogram(df_filtered, x="Rental Length Days", title="Distribution of Rental Length (Days)"),
                use_container_width=True,
            )
    if "Rental Length Hours" in df_filtered.columns:
        with cols[1]:
            st.plotly_chart(
                px.histogram(df_filtered, x="Rental Length Hours", title="Distribution of Rental Length (Hours)"),
                use_container_width=True,
            )
    if "Days Charged Count" in df_filtered.columns:
        with cols[2]:
            st.plotly_chart(
                px.histogram(df_filtered, x="Days Charged Count", title="Distribution of Days Charged"),
                use_container_width=True,
            )

# ------------------------------------------------------------
# TAB 5 — Price & Discounting
# ------------------------------------------------------------
with tab5:
    # Monthly aggregates for pricing if fields exist
    df_price = df_filtered.copy()
    has_base = "base_price_per_day" in df_price.columns and df_price["base_price_per_day"].notna().any()
    has_disc = "discount_rate" in df_price.columns and df_price["discount_rate"].notna().any()

    # Distributions
    cols = st.columns(2)
    if has_base:
        with cols[0]:
            fig_b = px.histogram(df_price, x="base_price_per_day", title="Distribution of Base Price per Day")
            fig_b.update_layout(xaxis_title="Base Price per Day", yaxis_title="Frequency")
            st.plotly_chart(fig_b, use_container_width=True)
    if has_disc:
        with cols[1]:
            fig_d = px.histogram(df_price, x="discount_rate", title="Distribution of Discount Rate")
            fig_d.update_layout(xaxis_title="Discount Rate", yaxis_title="Frequency")
            st.plotly_chart(fig_d, use_container_width=True)

    # Monthly time series for rentals / avg price / avg discount
    df_price["__date_idx__"] = pd.to_datetime(df_price["__date_idx__"], errors="coerce")
    agg = (
        df_price.dropna(subset=["__date_idx__"])
        .groupby(pd.Grouper(key="__date_idx__", freq="M"))
        .agg(rentals=("row_id_for_counts", "count"),
             avg_base_price=("base_price_per_day", "mean") if has_base else ("row_id_for_counts", "size"),
             avg_discount_rate=("discount_rate", "mean") if has_disc else ("row_id_for_counts", "size"))
        .reset_index()
    )

    col_ts1, col_ts2, col_ts3 = st.columns(3)
    with col_ts1:
        st.plotly_chart(px.bar(agg, x="__date_idx__", y="rentals", title="Monthly Rentals"),
                        use_container_width=True)
    if has_base:
        with col_ts2:
            st.plotly_chart(px.bar(agg, x="__date_idx__", y="avg_base_price",
                                   title="Average Base Price per Day (Monthly)"),
                            use_container_width=True)
    if has_disc:
        with col_ts3:
            st.plotly_chart(px.bar(agg, x="__date_idx__", y="avg_discount_rate",
                                   title="Average Discount Rate (Monthly)"),
                            use_container_width=True)
