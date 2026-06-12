from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]


st.set_page_config(
    page_title="Sales & Demand Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: transparent; border: none; box-shadow: none;}
    [data-testid="stMetricValue"] {font-size: 1.7rem !important; font-weight: 800 !important; color: #0f172a !important;}
    [data-testid="stMetricLabel"] {color: #334155 !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.06em;}
    .hero-card {background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #3b82f6 100%); border-radius: 24px; padding: 1.7rem; color: white; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.25);}
    .stat-card {background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%); border: 1px solid #c7d2fe; border-radius: 18px; padding: 1rem 1.1rem; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08); min-height: 138px;}
    .stat-value {font-size: 1.75rem; font-weight: 800; color: #0f172a; margin: 0.3rem 0 0.2rem 0;}
    .stat-label {color: #1e293b; font-size: 0.88rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;}
    .info-box {background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%); border-left: 4px solid #2563eb; border-radius: 14px; padding: 0.95rem 1rem; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05); color: #0f172a;}
    .nav-box {background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%); border: 1px solid #c7d2fe; border-radius: 14px; padding: 0.85rem 0.95rem; margin-bottom: 0.7rem; color: #0f172a; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);}
    .nav-box b {color: #0f172a;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_home_summary() -> dict:
    summary = {
        "training_records": 0,
        "stores": 0,
        "departments": 0,
        "forecast_accuracy": 0.0,
    }

    try:
        train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
        stores = pd.read_csv(ROOT / "data" / "raw" / "stores.csv")
        metrics = pd.read_csv(ROOT / "models" / "model_metrics.csv")
        xgb_metrics = metrics.loc[metrics["model"] == "XGBoost"].iloc[0]
        summary = {
            "training_records": int(len(train)),
            "stores": int(len(stores)),
            "departments": int(train["Dept"].nunique()),
            "forecast_accuracy": round(float(xgb_metrics["Forecast_Accuracy_%"]), 1),
        }
    except FileNotFoundError:
        pass

    return summary


summary = load_home_summary()

with st.container():
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom: 0.25rem;">Sales & Demand Intelligence Platform</h1>
            <p style="margin: 0; font-size: 1.02rem; opacity: 0.95;">A premium forecasting workspace for Walmart demand planning, inventory risk, and executive-ready analytics.</p>
            <p style="margin: 0.55rem 0 0 0; font-size: 0.95rem; opacity: 0.96;"><b>Highlights:</b> XGBoost forecasting • SHAP explainability • real-time dashboard views • recruiter-ready presentation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

metric_cards = [
    ("🧾", "Training Records", f"{summary['training_records']:,}", "Rows used to train the forecasting pipeline"),
    ("🏬", "Stores", f"{summary['stores']}", "Retail locations included in the analysis"),
    ("🧩", "Departments", f"{summary['departments']}", "Product departments covered by the model"),
    ("🎯", "Forecast Accuracy", f"{summary['forecast_accuracy']}%", "Model accuracy achieved on the test set"),
]

cols = st.columns(4)
for col, (icon, label, value, subtext) in zip(cols, metric_cards):
    with col:
        st.markdown(
            f"""
            <div class="stat-card">
                <div style="font-size: 1.3rem;">{icon}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
                <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.35rem;">{subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

col_left, col_right = st.columns([1.7, 1.0], gap="large")
with col_left:
    st.subheader("What you can do here")
    st.write(
        "Use the sidebar pages to move from demand forecasting to a complete business analytics view. "
        "The experience is now connected to the trained model artifacts and the saved forecast output, so the dashboard behaves like a real planning tool."
    )
    st.success("The forecast page predicts weekly demand using the trained XGBoost model, and the analytics page visualizes real sales and forecast trends.")

with col_right:
    st.subheader("Quick navigation")
    st.markdown("<div class='nav-box'>📈 <b>Demand Forecast</b><br/>Simulate store and department demand with the trained model.</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-box'>📊 <b>Analytics Dashboard</b><br/>Review sales trends, holiday behaviors, and forecast visibility.</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-box'>🧠 <b>Model Performance</b><br/>Explore the forecasting metrics and business impact of the model.</div>", unsafe_allow_html=True)

st.divider()

st.subheader("Operational workflow")
workflow_col1, workflow_col2, workflow_col3 = st.columns(3)
with workflow_col1:
    st.markdown("<div class='info-box'><b>1. Forecast demand</b><br/>Generate a weekly demand prediction for any store and department combination.</div>", unsafe_allow_html=True)
with workflow_col2:
    st.markdown("<div class='info-box'><b>2. Review business impact</b><br/>Check inventory pressure, reorder guidance, and risk signals from the forecast.</div>", unsafe_allow_html=True)
with workflow_col3:
    st.markdown("<div class='info-box'><b>3. Monitor performance</b><br/>Compare sales movement, trends, and forecasting accuracy through the dashboard.</div>", unsafe_allow_html=True)
