from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.db_utils import DATABASE_PATH, get_conn, load_data_to_sqlite


st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%); border: 1px solid #dce8ff; border-radius: 16px; padding: 0.8rem 1rem; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);}
    [data-testid="stMetricLabel"] {color: #334155 !important; font-weight: 700 !important;}
    [data-testid="stMetricValue"] {color: #0f172a !important; font-weight: 800 !important; font-size: 1.45rem !important;}
    .stSubheader > div {color: #0f172a !important; font-weight: 800 !important;}
    .stSuccess {background: linear-gradient(135deg, #ecfdf3, #f0fdf4); border-left: 4px solid #16a34a; border-radius: 12px; color: #166534;}
    .stInfo {background: linear-gradient(135deg, #eff6ff, #f8fbff); border-left: 4px solid #2563eb; border-radius: 12px; color: #1d4ed8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Analytics Dashboard")
st.caption("A polished business intelligence view of historical sales performance, holiday effects, and forecast alignment.")

if not DATABASE_PATH.exists():
    load_data_to_sqlite()


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with get_conn() as conn:
        weekly = pd.read_sql(
            """
            SELECT Date, SUM(Weekly_Sales) AS Total_Sales
            FROM sales
            GROUP BY Date
            ORDER BY Date
            """,
            conn,
            parse_dates=["Date"],
        )
        monthly = (
            weekly.assign(Month=weekly["Date"].dt.to_period("M"))
            .groupby("Month", as_index=False)["Total_Sales"]
            .sum()
        )
        monthly["Month"] = monthly["Month"].dt.to_timestamp()
        top_stores = pd.read_sql(
            """
            SELECT Store, SUM(Weekly_Sales) AS Total_Sales
            FROM sales
            GROUP BY Store
            ORDER BY Total_Sales DESC
            LIMIT 10
            """,
            conn,
        )
        top_departments = pd.read_sql(
            """
            SELECT Dept, SUM(Weekly_Sales) AS Total_Sales
            FROM sales
            GROUP BY Dept
            ORDER BY Total_Sales DESC
            LIMIT 15
            """,
            conn,
        )
        holiday = pd.read_sql(
            """
            SELECT IsHoliday, AVG(Weekly_Sales) AS Average_Sales
            FROM sales
            GROUP BY IsHoliday
            """,
            conn,
        )
        holiday["Holiday_Label"] = holiday["IsHoliday"].map({0: "No", 1: "Yes"})
    return weekly, monthly, top_stores, top_departments, holiday


weekly, monthly, top_stores, top_departments, holiday = load_dashboard_data()

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Total Sales", f"${weekly['Total_Sales'].sum():,.0f}")
metric_col2.metric("Average Weekly Sales", f"${weekly['Total_Sales'].mean():,.0f}")
metric_col3.metric("Peak Week", weekly.loc[weekly["Total_Sales"].idxmax(), "Date"].date().strftime("%Y-%m-%d"))
metric_col4.metric("Weeks Tracked", f"{len(weekly):,}")

st.divider()

chart_col1, chart_col2 = st.columns([1.4, 0.8], gap="large")
with chart_col1:
    sales_trend = px.line(
        weekly,
        x="Date",
        y="Total_Sales",
        title="Weekly Sales Trend",
        labels={"Date": "Week", "Total_Sales": "Weekly Sales"},
    )
    sales_trend.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(sales_trend, use_container_width=True)

with chart_col2:
    holiday_chart = px.bar(
        holiday,
        x="Holiday_Label",
        y="Average_Sales",
        title="Holiday vs Non-Holiday Sales",
        labels={"Holiday_Label": "Holiday Week", "Average_Sales": "Average Sales"},
    )
    holiday_chart.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(holiday_chart, use_container_width=True)

st.markdown("<h4 style='margin-bottom: 0.5rem; color:#0f172a;'>Forecast visibility</h4>", unsafe_allow_html=True)
forecast_path = ROOT / "models" / "monthly_demand_forecast.csv"
if forecast_path.exists():
    forecast_df = pd.read_csv(forecast_path, parse_dates=["Month"])
    monthly_forecast = (
        forecast_df.groupby("Month", as_index=False)
        .agg(Actual_Monthly_Demand=("Actual_Monthly_Demand", "sum"), Predicted_Monthly_Demand=("Predicted_Monthly_Demand", "sum"))
    )
    forecast_fig = px.line(
        monthly_forecast,
        x="Month",
        y=["Actual_Monthly_Demand", "Predicted_Monthly_Demand"],
        markers=True,
        title="Monthly Actual vs Forecasted Demand",
        labels={"value": "Demand", "variable": "Series"},
    )
    forecast_fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(forecast_fig, use_container_width=True)
else:
    st.info("The monthly forecast export is not present yet. Run the training workflow to produce the forecast file.")

st.divider()

bottom_col1, bottom_col2 = st.columns(2, gap="large")
with bottom_col1:
    top_store_chart = px.bar(
        top_stores,
        x="Store",
        y="Total_Sales",
        title="Top Stores by Sales",
        labels={"Total_Sales": "Sales"},
    )
    top_store_chart.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(top_store_chart, use_container_width=True)

with bottom_col2:
    top_department_chart = px.bar(
        top_departments,
        x="Dept",
        y="Total_Sales",
        title="Top Departments by Sales",
        labels={"Total_Sales": "Sales"},
    )
    top_department_chart.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(top_department_chart, use_container_width=True)

st.success("This dashboard now combines historical sales performance, holiday effect insights, and predictive trend visibility in one professional view.")
