# app.py
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
def safe_top_value(s: pd.Series | None, default="—"):
    if s is None:
        return default
    s = s.dropna()
    return default if s.empty else s.value_counts().idxmax()

def add_seasonal_flags(d: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = d.copy()
    m = d[date_col].dt.month
    day = d[date_col].dt.day
    d["is_summer"] = m.isin([6, 7, 8])
    d["is_christmas_newyear"] = ((m == 12) & (day >= 15)) | ((m == 1) & (day <= 7))
    # Eid ranges (approx; 2019–2024)
    eid_ranges = [
        ("2019-06-03","2019-06-06"), ("2019-08-11","2019-08-14"),
        ("2020-05-24","2020-05-26"), ("2020-07-31","2020-08-03"),
        ("2021-05-13","2021-05-16"), ("2021-07-20","2021-07-23"),
        ("2022-05-02","2022-05-05"), ("2022-07-09","2022-07-12"),
        ("2023-04-20","2023-04-23"), ("2023-06-28","2023-07-01"),
        ("2024-04-09","2024-04-12"), ("2024-06-16","2024-06-19"),
    ]
    mask = False
    for a, b in eid_ranges:
        mask = mask | ((d[date_col] >= pd.to_datetime(a)) & (d[date_col] <= pd.to_datetime(b)))
    d["is_eid"] = mask
    return d

def compute_pricing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create base_price_per_day and discount_rate if possible."""
    df = df.copy()
    # Base amount candidates (prefer Net Time&Dist Amount)
    base_cands = [
        "Net Time&Dist Amount", "Pre-Tax Rental Amount", "Daily Rate Amount",
        "Rate Amount", "Base Rate Amount"
    ]
    base_col = next((c for c in base_cands if c in df.columns), None)

    # Daily denominator
    if "Days Charged Count" in df.columns:
        denom = pd.to_numeric(df["Days Charged Count"], errors="coerce").replace(0, np.nan)
    else:
        days = pd.to_numeric(df.get("Rental Length Days", np.nan), errors="coerce")
        hours = pd.to_numeric(df.get("Rental Length Hours", np.nan), errors="coerce")
        denom = (days.fillna(0) * 24 + hours.fillna(0)) / 24.0
        denom = denom.replace(0, np.nan)

    if base_col:
        base_vals = pd.to_numeric(df[base_col], errors="coerce")
        # Most exports store cents; if values look huge, divide by 100
        scale = 100 if base_vals.median(skipna=True) > 500 else 1
        df["base_price_per_day"] = (base_vals / denom) / scale

    # Discount %
    disc_col = None
    for c in ["Discount %", "Discount Percent", "Discount Rate"]:
        if c in df.columns:
            disc_col = c
            break
    if disc_col:
        dvals = pd.to_numeric(df[disc_col], errors="coerce").abs()
        if dvals.max(skipna=True) and dvals.max(skipna=True) > 100:  # basis points
            df["discount_rate"] = dvals / 10000.0
        else:
            df["discount_rate"] = dvals / 100.0
    else:
        df["discount_rate"] = np.nan

    return df

@st.cache_data(show_spinner=True)
def load_data() -> tuple[pd.DataFrame, str]:
    df = pd.read_excel(DATA_PATH)

    # Dates
    if "Checkout Date" in df.columns:
        df["Checkout Date"] = pd.to_datetime(df["Checkout Date"], errors="coerce")
    if "Checkin Date" in df.columns:
        df["Checkin Date"] = pd.to_datetime(df["Checkin Date"], errors="coerce")

    df["__date_idx__"] = df["Checkout Date"]
    df["row_id_for_counts"] = range(1, len(df) + 1)

    # Location (explicit name + fallbacks)
    LOC_CANDS = ["Checkout Location Name", "Checkout Location", "Checkout Location code", "Branch", "Location"]
    loc_col = next((c for c in LOC_CANDS if c in df.columns), None)
    if loc_col:
        df["__location__"] = df[loc_col].fillna("Unknown")
    else:
        df["__location__"] = "Unknown"

    # Vehicle group column used throughout
    VEH_CANDS = ["Vehicle Group Rented", "Vehicle Group Charged", "Vehicle Category", "Car Group", "Car Class"]
    veh_col = next((c for c in VEH_CANDS if c in df.columns), None)

    # Channel (Broker vs Direct via money signals)
    COMM = df.get("Commission Amount", pd.Series(np.nan, index=df.index))
    TRAV = df.get("Travel Agent Prepay Tour Voucher Amount", pd.Series(np.nan, index=df.index))
    USED = df.get("Used Tour Voucher Amount", pd.Series(np.nan, index=df.index))
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
        if pd.isna(x):
            return "Unknown"
        s = str(x).strip().upper()
        if s in GCC:
            return "Gulf"
        if s in {"LB", "LEBANON"}:
            return "Local"
        return "Other"

    df["cust_region"] = df[country_col].apply(region_from_country) if country_col else "Unknown"

    # Pricing columns
    df = compute_pricing_columns(df)

    return df, (veh_col or "")

df, VEHICLE_COL = load_data()

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
# Header & Filters (main page, not sidebar)
# ------------------------------------------------------------
st.title("🚗 Rental Analytics Dashboard")

with st.container():
    top_left, top_mid, top_right = st.columns([6, 3, 1])
    with top_left:
        st.subheader("Filters")

        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            st.date_input(
                "Date range (Checkout Date)",
                value=st.session_state.flt_date,
                min_value=MIN_DT.date(),
                max_value=MAX_DT.date(),
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
            st.multiselect(
                "Vehicle Groups" if VEHICLE_COL else "Vehicle Groups (not found)",
                options=sorted(df[VEHICLE_COL].dropna().unique()) if VEHICLE_COL else [],
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
date_vals = st.session_state.flt_date
if isinstance(date_vals, (list, tuple)) and len(date_vals) == 2:
    date_start, date_end = date_vals
else:
    date_start = date_end = date_vals

mask = df["__date_idx__"].between(pd.to_datetime(date_start), pd.to_datetime(date_end))
if st.session_state.flt_loc:
    mask &= df["__location__"].isin(st.session_state.flt_loc)
if st.session_state.flt_channel:
    mask &= df["cust_channel"].isin(st.session_state.flt_channel)
if st.session_state.flt_region:
    mask &= df["cust_region"].isin(st.session_state.flt_region)
if st.session_state.flt_vehicle and VEHICLE_COL:
    mask &= df[VEHICLE_COL].isin(st.session_state.flt_vehicle)

df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.info("No rows match the current filters.")
    st.stop()

# ------------------------------------------------------------
# KPIs (always visible)
# ------------------------------------------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Total Rentals", f"{len(df_filtered):,}")

total_rev = pd.to_numeric(df_filtered.get("Net Time&Dist Amount", 0), errors="coerce").sum() / 100
k2.metric("Total Revenue", f"{total_rev:,.0f}")

if {"Days Charged Count", "Net Time&Dist Amount"}.issubset(df_filtered.columns):
    adr = (
        pd.to_numeric(df_filtered["Net Time&Dist Amount"], errors="coerce")
        / pd.to_numeric(df_filtered["Days Charged Count"], errors="coerce").replace(0, np.nan)
    ).mean() / 100
    k3.metric("Avg Daily Rate", f"{adr:,.0f}")
else:
    k3.metric("Avg Daily Rate", "—")

k4, k5, k6 = st.columns(3)
wkend_share = (df_filtered["Checkout Date"].dt.weekday >= 5).mean() * 100
k4.metric("Weekend Share", f"{wkend_share:,.1f}%")
k5.metric("Top Vehicle", safe_top_value(df_filtered.get(VEHICLE_COL)))  # uses detected vehicle column
k6.metric("Top Location", safe_top_value(df_filtered.get("__location__")))

st.markdown("---")

# ------------------------------------------------------------
# Tabs (grouped business views)
# ------------------------------------------------------------
tab_overview, tab_trends, tab_mix, tab_pricing, tab_time = st.tabs(
    ["Overview", "Volume & Trends", "Customer & Fleet Mix", "Pricing & Discounts", "Time Patterns"]
)

# ---------- Overview ----------
with tab_overview:
    st.subheader("Top Trend")
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

# ---------- Volume & Trends ----------
with tab_trends:
    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
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

# ---------- Customer & Fleet Mix ----------
with tab_mix:
    c1, c2 = st.columns(2)
    with c1:
        if VEHICLE_COL:
            vc = (
                df_filtered[VEHICLE_COL].value_counts().head(10)
                .rename_axis("vehicle_group")
                .reset_index(name="count")
            )
            st.plotly_chart(
                px.bar(vc, x="vehicle_group", y="count", title="Top Vehicle Groups"),
                use_container_width=True
            )
        loc_top = (
            df_filtered["__location__"].value_counts().head(10)
            .rename_axis("location").reset_index(name="rentals")
        )
        st.plotly_chart(
            px.bar(loc_top, x="location", y="rentals", title="Top 10 Locations"),
            use_container_width=True
        )
    with c2:
        ch = (
            df_filtered["cust_channel"].value_counts()
            .rename_axis("channel").reset_index(name="count")
        )
        st.plotly_chart(
            px.bar(ch, x="channel", y="count", title="Channel Breakdown"),
            use_container_width=True
        )

        rg = (
            df_filtered["cust_region"].value_counts()
            .rename_axis("region").reset_index(name="count")
        )
        st.plotly_chart(
            px.bar(rg, x="region", y="count", title="Region Breakdown"),
            use_container_width=True
        )

# ---------- Pricing & Discounts ----------
with tab_pricing:
    st.caption("Prices shown are normalized per day where possible; currency assumed from source.")
    p1, p2 = st.columns(2)
    with p1:
        if "base_price_per_day" in df_filtered.columns:
            st.plotly_chart(
                px.histogram(
                    df_filtered, x="base_price_per_day",
                    nbins=60, title="Distribution of Base Price per Day"
                ).update_layout(xaxis_title="Base Price per Day", yaxis_title="Frequency"),
                use_container_width=True
            )
        else:
            st.info("Base price per day not available from source columns.")
    with p2:
        if "discount_rate" in df_filtered.columns:
            st.plotly_chart(
                px.histogram(
                    df_filtered, x="discount_rate",
                    nbins=60, title="Distribution of Discount Rate"
                ).update_layout(xaxis_title="Discount Rate", yaxis_title="Frequency"),
                use_container_width=True
            )
        else:
            st.info("Discount rate not available from source columns.")

# ---------- Time Patterns ----------
with tab_time:
    df_time = add_seasonal_flags(df_filtered, "__date_idx__")

    t1, t2 = st.columns(2)

    with t1:
        # Weekday distribution
        wd = (
            df_time["Checkout Date"].dt.day_name().value_counts()
            .rename_axis("weekday").reset_index(name="count")
        )
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        wd["weekday"] = pd.Categorical(wd["weekday"], categories=order, ordered=True)
        wd = wd.sort_values("weekday")
        st.plotly_chart(
            px.bar(wd, x="weekday", y="count", title="Rentals by Weekday"),
            use_container_width=True
        )

        # Weekend vs weekday share
        weekend_counts = pd.Series(
            np.where(df_time["Checkout Date"].dt.weekday >= 5, "Weekend", "Weekday")
        ).value_counts().rename_axis("is_weekend").reset_index(name="count")
        st.plotly_chart(
            px.bar(weekend_counts, x="is_weekend", y="count", title="Rental Frequency: Weekend vs Weekday"),
            use_container_width=True
        )

    with t2:
        # Checkout time → 3-hour bins
        ct = df_time["Checkout Time"].astype(str)
        if ct.str.contains(":").any():  # like "19:24:00"
            hours = pd.to_datetime(ct, errors="coerce").dt.hour
        else:                            # like 1924 or "1924"
            hours = pd.to_datetime(ct.str.zfill(4), format="%H%M", errors="coerce").dt.hour

        bins = list(range(0, 25, 3))
        labels = [f"{i:02d}:00-{i+3:02d}:00" for i in bins[:-1]]
        labels[-1] = "21:00-24:00"
        cats = pd.cut(hours, bins=bins, right=False, include_lowest=True, labels=labels)
        cats = pd.Categorical(cats, categories=labels, ordered=True)
        time_bin_counts = pd.Series(cats).value_counts(sort=False, dropna=False).reindex(labels).fillna(0).astype(int)
        df_bins = pd.DataFrame({"Time Bin": labels, "Count": time_bin_counts.values})

        st.plotly_chart(
            px.bar(df_bins, x="Time Bin", y="Count", title="Rental Frequency by Checkout Time (3-hour Bins)")
            .update_layout(xaxis={"categoryorder": "array", "categoryarray": labels}),
            use_container_width=True
        )

    st.markdown("---")
    # Holiday / seasonal share
    def share(mask: pd.Series) -> float:
        return (mask.sum() / len(mask)) if len(mask) else np.nan

    holiday_summary = pd.DataFrame({
        "share_eid": [share(df_time["is_eid"])],
        "share_christmas_newyear": [share(df_time["is_christmas_newyear"])],
        "share_summer": [share(df_time["is_summer"])],
    }).T.rename(columns={0: "share"}).reset_index(names="period")

    st.plotly_chart(
        px.bar(holiday_summary, x="period", y="share",
               title="Share of Rentals During Holiday/Seasonal Periods")
        .update_layout(xaxis_title="Period", yaxis_title="Share of Rentals"),
        use_container_width=True
    )

    # Rental length distributions (extra)
    extra1, extra2, extra3 = st.columns(3)
    if "Rental Length Days" in df_time.columns:
        extra1.plotly_chart(
            px.histogram(df_time, x="Rental Length Days", title="Rental Length (Days)"),
            use_container_width=True
        )
    if "Rental Length Hours" in df_time.columns:
        extra2.plotly_chart(
            px.histogram(df_time, x="Rental Length Hours", title="Rental Length (Hours)"),
            use_container_width=True
        )
    if "Days Charged Count" in df_time.columns:
        extra3.plotly_chart(
            px.histogram(df_time, x="Days Charged Count", title="Days Charged"),
            use_container_width=True
        )
