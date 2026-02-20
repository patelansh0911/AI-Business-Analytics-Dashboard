import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Business Analytics & Forecasting", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])
    return df

def kpis(df: pd.DataFrame) -> dict:
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    orders = df["Order_ID"].nunique()
    aov = total_sales / orders if orders else 0
    margin = (total_profit / total_sales) if total_sales else 0
    return {
        "Total Sales": total_sales,
        "Total Profit": total_profit,
        "Orders": orders,
        "Avg Order Value": aov,
        "Profit Margin": margin
    }

def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # ensure column exists and is datetime
    d.columns = d.columns.str.strip()
    
    # Debug: print available columns if Order_Date is missing
    if "Order_Date" not in d.columns:
        st.error(f"Order_Date not found. Available columns: {list(d.columns)}")
        return pd.DataFrame(columns=["Order_Date", "Sales"])

    d["Order_Date"] = pd.to_datetime(d["Order_Date"], errors="coerce")
    d = d.dropna(subset=["Order_Date"])
    
    # Check if we have any data after filtering
    if len(d) == 0:
        st.warning("No data available for the selected filters.")
        return pd.DataFrame(columns=["Order_Date", "Sales"])

    # Group by date and sum sales - use resample to properly preserve the date column
    daily = d.set_index("Order_Date").resample("D")["Sales"].sum().reset_index()
    daily = daily.sort_values("Order_Date")
    
    return daily

def add_time_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["day_ordinal"] = x["Order_Date"].map(pd.Timestamp.toordinal)
    x["dow"] = x["Order_Date"].dt.dayofweek
    x["month"] = x["Order_Date"].dt.month
    x["year"] = x["Order_Date"].dt.year
    return x

def forecast_sales(daily: pd.DataFrame, horizon_days: int = 30):
    # Train on historical daily sales
    data = add_time_features(daily)
    X = data[["day_ordinal", "dow", "month", "year"]]
    y = data["Sales"]

    # Simple, strong baseline model
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2
    )
    model.fit(X, y)

    # quick backtest (last 30 days if possible)
    backtest_days = min(30, len(data) // 5) if len(data) >= 50 else 0
    mae = None
    if backtest_days >= 7:
        train = data.iloc[:-backtest_days]
        test = data.iloc[-backtest_days:]
        m2 = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
        m2.fit(train[["day_ordinal", "dow", "month", "year"]], train["Sales"])
        pred = m2.predict(test[["day_ordinal", "dow", "month", "year"]])
        mae = mean_absolute_error(test["Sales"], pred)

    # future frame
    last_date = daily["Order_Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    future = pd.DataFrame({"Order_Date": future_dates})
    future = add_time_features(future)
    future_pred = model.predict(future[["day_ordinal", "dow", "month", "year"]])

    fc = pd.DataFrame({"Order_Date": future_dates, "Forecast_Sales": future_pred})
    return fc, mae

st.title("📊 AI-Powered Business Analytics & Forecasting Dashboard")

with st.sidebar:
    st.header("Data")
    data_path = st.text_input("CSV path", "business_sales_dataset.csv")
    st.caption("Tip: keep the dataset in the same folder as app.py, or provide full path.")

df = load_data(data_path)

# Filters
st.subheader("Filters")
c1, c2, c3, c4 = st.columns(4)
with c1:
    region = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
with c2:
    category = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
with c3:
    segment = st.multiselect("Segment", sorted(df["Segment"].unique()), default=sorted(df["Segment"].unique()))
with c4:
    date_min = df["Order_Date"].min().date()
    date_max = df["Order_Date"].max().date()
    date_range = st.date_input("Date range", (date_min, date_max), min_value=date_min, max_value=date_max)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

f = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Segment"].isin(segment)) &
    (df["Order_Date"].between(start_date, end_date))
].copy()

# KPIs
k = kpis(f)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Sales", f"₹{k['Total Sales']:,.0f}")
k2.metric("Total Profit", f"₹{k['Total Profit']:,.0f}")
k3.metric("Orders", f"{k['Orders']:,}")
k4.metric("Avg Order Value", f"₹{k['Avg Order Value']:,.0f}")
k5.metric("Profit Margin", f"{k['Profit Margin']*100:.1f}%")

st.divider()

# Charts row 1
left, right = st.columns((2, 1))

with left:
    st.subheader("Sales Trend")
    daily = build_daily_series(f)
    fig = px.line(daily, x="Order_Date", y="Sales", title="Daily Sales")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Sales by Category")
    cat = f.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    fig = px.bar(cat, x="Category", y="Sales", title="Category Sales")
    st.plotly_chart(fig, use_container_width=True)

# Charts row 2
c1, c2 = st.columns(2)
with c1:
    st.subheader("Region Performance")
    r = f.groupby("Region", as_index=False)[["Sales","Profit"]].sum().sort_values("Sales", ascending=False)
    fig = px.bar(r, x="Region", y="Sales", title="Sales by Region")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Profit vs Sales")
    fig = px.scatter(f, x="Sales", y="Profit", color="Category", title="Profitability Scatter", hover_data=["Region","Segment","Sub_Category"])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Forecasting
st.subheader("🔮 Forecasting")
h = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1)
daily_all = build_daily_series(f)
fc, mae = forecast_sales(daily_all, horizon_days=h)

combo = daily_all.rename(columns={"Sales":"Actual_Sales"})[["Order_Date","Actual_Sales"]].merge(fc, on="Order_Date", how="outer")
fig = px.line(combo, x="Order_Date", y=["Actual_Sales","Forecast_Sales"], title="Actual vs Forecast Sales")
st.plotly_chart(fig, use_container_width=True)

if mae is not None:
    st.caption(f"Backtest (last ~30 days) MAE: {mae:,.2f}")

# Insights
st.subheader("💡 Quick Insights")
top_cat = cat.iloc[0]["Category"] if len(cat) else "N/A"
top_region = r.iloc[0]["Region"] if len(r) else "N/A"
st.write(f"- Top category by sales: **{top_cat}**")
st.write(f"- Top region by sales: **{top_region}**")
st.write("- Use the filters to compare performance across segments, categories, and regions.")
