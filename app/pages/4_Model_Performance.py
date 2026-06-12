from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


MODELS_DIR = ROOT / "models"

st.title("Model Performance")

metrics_path = MODELS_DIR / "model_metrics.csv"
final_metrics_path = MODELS_DIR / "final_test_metrics.csv"
forecast_path = MODELS_DIR / "monthly_demand_forecast.csv"

if not metrics_path.exists():
    st.warning("Model metrics were not found. Run `train_model.ipynb` or `python train_model.py` first.")
    st.stop()

metrics = pd.read_csv(metrics_path)
st.subheader("Model Comparison")
st.dataframe(metrics, use_container_width=True)

metric_cols = [col for col in ["MAE", "RMSE", "WMAE", "WAPE", "Forecast_Accuracy_%"] if col in metrics.columns]
selected_metric = st.selectbox("Comparison Metric", metric_cols, index=metric_cols.index("WMAE") if "WMAE" in metric_cols else 0)
st.plotly_chart(
    px.bar(metrics, x="model", y=selected_metric, title=f"Model Comparison by {selected_metric}"),
    use_container_width=True,
)

if final_metrics_path.exists():
    final_metrics = pd.read_csv(final_metrics_path).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final MAE", f"${final_metrics['MAE']:,.0f}")
    c2.metric("Final RMSE", f"${final_metrics['RMSE']:,.0f}")
    c3.metric("Final R2", f"{final_metrics['R2']:.3f}")
    c4.metric("Final WAPE", f"{final_metrics['WAPE']:.2%}")

if forecast_path.exists():
    forecast = pd.read_csv(forecast_path)
    st.subheader("Monthly Demand Forecast Sample")
    st.dataframe(forecast.head(50), use_container_width=True)
