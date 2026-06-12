from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.db_utils import load_raw_data
from src.feature_eng import SEASON_MAP, build_label_mappings
from src.model_utils import estimate_inventory_risk, load_model_bundle, prepare_prediction_input


MODELS_DIR = ROOT / "models"
FORECAST_FILE = MODELS_DIR / "monthly_demand_forecast.csv"


st.set_page_config(page_title="Demand Forecast", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%); border: 1px solid #dce8ff; border-radius: 16px; padding: 0.8rem 1rem; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);}
    [data-testid="stMetricLabel"] {color: #334155 !important; font-weight: 700 !important;}
    [data-testid="stMetricValue"] {color: #0f172a !important; font-weight: 800 !important; font-size: 1.45rem !important;}
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stSlider > label, .stCheckbox > label {color: #0f172a !important; font-weight: 700 !important;}
    .stButton > button {background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; border-radius: 10px; padding: 0.5rem 1rem; font-weight: 700;}
    .stButton > button:hover {background: linear-gradient(135deg, #1d4ed8, #1e40af); color: white;}
    .forecast-panel {background: linear-gradient(135deg, #f8fbff, #eef6ff); border: 1px solid #c7d2fe; border-radius: 16px; padding: 1rem; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05); color: #0f172a;}
    .chart-card {background: linear-gradient(135deg, #ffffff, #f8fbff); border: 1px solid #c7d2fe; border-radius: 18px; padding: 0.95rem; margin-top: 1rem; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Demand Forecast")
st.caption("Create a business-ready demand forecast with the trained model and review how the prediction compares with the saved monthly forecast history.")

try:
    model, metadata = load_model_bundle()
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.stop()


@st.cache_data
def load_reference_labels(metadata: dict | None = None) -> tuple[dict, dict, dict]:
    if metadata and metadata.get("label_mappings"):
        label_mappings = metadata["label_mappings"]
        return (
            label_mappings.get("Store", {}),
            label_mappings.get("Dept", {}),
            label_mappings.get("Type", {}),
        )

    train_raw, _, _, stores_raw = load_raw_data()
    reference_df = train_raw.copy()
    if "Type" not in reference_df.columns and stores_raw is not None:
        reference_df = reference_df.merge(stores_raw[["Store", "Type"]], on="Store", how="left")

    mappings = build_label_mappings(reference_df)
    return mappings.get("Store", {}), mappings.get("Dept", {}), mappings.get("Type", {})


store_map, dept_map, type_map = load_reference_labels(metadata)


@st.cache_data
def load_forecast_history() -> pd.DataFrame:
    if FORECAST_FILE.exists():
        history = pd.read_csv(FORECAST_FILE, parse_dates=["Month"])
        history = history.sort_values(["Store", "Dept", "Month"])
        return history
    return pd.DataFrame(columns=["Store", "Dept", "Month", "Actual_Monthly_Demand", "Predicted_Monthly_Demand"])


forecast_history = load_forecast_history()

st.markdown("<div class='forecast-panel'><h4 style='margin:0 0 0.5rem 0; color:#0f172a;'>Forecast input</h4><p style='margin:0; color:#334155;'>Adjust the store, department, and business conditions to generate a practical demand forecast.</p></div>", unsafe_allow_html=True)

with st.form("forecast_form"):
    col1, col2 = st.columns(2)
    with col1:
        store_options = list(store_map.keys())
        store = st.selectbox(
            "Store",
            options=store_options,
            format_func=lambda value: store_map.get(value, f"Store {value}"),
        )
        dept_options = list(dept_map.keys())
        dept = st.selectbox(
            "Department",
            options=dept_options,
            format_func=lambda value: dept_map.get(value, f"Department {value}"),
        )
        type_options = list(type_map.keys())
        store_type = st.selectbox(
            "Store Type",
            options=type_options,
            format_func=lambda value: type_map.get(value, f"Store Type {value}"),
        )
        size = st.number_input("Store Size", min_value=1000, max_value=250000, value=150000, step=1000)
        current_inventory_value = st.number_input(
            "Current Inventory Value",
            min_value=0.0,
            value=25000.0,
            step=1000.0,
        )
    with col2:
        year = st.number_input("Year", min_value=2010, max_value=2030, value=2013, step=1)
        month = st.slider("Month", 1, 12, 6)
        week = st.slider("Week", 1, 52, 24)
        is_holiday = st.checkbox("Holiday Week")
        temperature = st.slider("Temperature", 0.0, 110.0, 65.0)
        fuel_price = st.slider("Fuel Price", 1.0, 6.0, 3.5)

    submitted = st.form_submit_button("Predict Demand", type="primary")

if submitted:
    season = SEASON_MAP.get(month, "Summer")
    quarter = (month - 1) // 3 + 1
    user_inputs = {
        "Store": store,
        "Dept": dept,
        "Type": store_type,
        "Size": size,
        "Year": year,
        "Month": month,
        "Week": week,
        "Quarter": quarter,
        "Day": 1,
        "Season": season,
        "IsHoliday": int(is_holiday),
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
    }

    X = prepare_prediction_input(user_inputs, metadata)
    predicted_sales = max(0.0, float(model.predict(X)[0]))
    risk = estimate_inventory_risk(predicted_sales, current_inventory_value)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Weekly Demand", f"${predicted_sales:,.0f}")
    c2.metric("Suggested Reorder Value", f"${risk['reorder_value']:,.0f}")
    c3.metric("Risk Level", risk["risk"])

    st.caption(
        f"Selected inputs: {store_map.get(store, f'Store {store}')}, {dept_map.get(dept, f'Department {dept}')}, {type_map.get(store_type, f'Store Type {store_type}')}, {'Holiday Week' if is_holiday else 'Non-Holiday Week'}"
    )

    st.markdown(
        f"""
        <div class='forecast-panel' style='margin-top: 0.8rem;'><b>Planning insight:</b> Coverage ratio is <b>{risk['coverage_ratio']:.2f}</b>. The inventory recommendation is based on the forecasted demand and the stock value you entered.</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='chart-card'><h4 style='margin:0 0 0.5rem 0; color:#0f172a;'>Forecast context</h4><p style='margin:0 0 0.7rem 0; color:#475569;'>The chart below is shown in a full-width panel so the forecast trend is easier to read during presentations.</p></div>", unsafe_allow_html=True)
if not forecast_history.empty:
    store_history = forecast_history[(forecast_history["Store"] == store) & (forecast_history["Dept"] == dept)]
    if not store_history.empty:
        recent_history = store_history.sort_values("Month").tail(12)
        fig = px.line(
            recent_history,
            x="Month",
            y=["Actual_Monthly_Demand", "Predicted_Monthly_Demand"],
            markers=True,
            labels={"value": "Demand", "variable": "Series"},
            title=f"Actual vs Predicted Demand for {store_map.get(store, f'Store {store}')}, {dept_map.get(dept, f'Department {dept}')}",
        )
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("Saved forecast history for this store and department combination is not available yet.")
else:
    st.caption("Forecast history file was not found. Run the training workflow to generate the monthly demand forecast export.")

st.divider()

if not forecast_history.empty:
    st.markdown("<h4 style='margin-bottom: 0.5rem; color:#0f172a;'>Recent historical forecast snapshot</h4>", unsafe_allow_html=True)
    sample = forecast_history[(forecast_history["Store"] == store) & (forecast_history["Dept"] == dept)]
    if not sample.empty:
        preview = sample.sort_values("Month").tail(8).copy()
        preview["Monthly_Difference"] = preview["Monthly_Difference"].round(2)
        st.dataframe(preview[["Month", "Actual_Monthly_Demand", "Predicted_Monthly_Demand", "Monthly_Difference"]], use_container_width=True)
    else:
        st.caption("No matching forecast rows were found for the selected combination.")
