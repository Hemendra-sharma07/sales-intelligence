# Sales & Demand Intelligence Platform

Machine learning project for Walmart store-department sales forecasting, monthly demand planning, model explainability, and business analytics.

## Dataset

This project uses the Walmart Store Sales Forecasting dataset.

Place these files in `data/raw/`:

- `train.csv`
- `test.csv`
- `features.csv`
- `stores.csv`

## Project Structure

```text
sales-intelligence/
├── data/
│   ├── raw/
│   └── database/
├── models/
├── app/
│   ├── main.py
│   └── pages/
├── src/
│   ├── db_utils.py
│   ├── feature_eng.py
│   └── model_utils.py
├── train_model.ipynb
├── requirements.txt
└── README.md
```

## Main Workflow

1. Load raw Walmart CSV files.
2. Perform EDA on sales, stores, departments, holidays, promotions, and economic variables.
3. Clean missing values:
   - Markdown columns are filled with `0`.
   - Numeric null values are filled with mean values.
   - Categorical null values are filled with mode values.
4. Engineer date, promotion, lag, rolling, and encoded categorical features.
5. Detect target outliers using box plots and IQR.
6. Split data by time into train, validation, and test periods.
7. Train regression models and select the best model.
8. Evaluate with MAE, RMSE, R2, WMAE, WAPE, and forecast accuracy percentage.
9. Save the final model and metadata.
10. Generate monthly demand forecasts from weekly predictions.

## Run Locally

```bash
python -m venv sales_venv
sales_venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook train_model.ipynb
streamlit run app/main.py
```

## Train From Script

```bash
python train_model.py
```

The script creates:

- `data/database/inventory.db`
- `models/xgb_demand_model.pkl`
- `models/shap_explainer.pkl`
- `models/model_features.pkl`
- `models/model_metrics.csv`
- `models/monthly_demand_forecast.csv`

## Notes

The main prediction target is `Weekly_Sales`, which is sales value in dollars. Monthly demand is created by aggregating weekly predictions. Inventory risk is estimated later using forecasted demand and user-provided inventory assumptions.
