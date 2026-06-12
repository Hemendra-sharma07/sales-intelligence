import numpy as np
import pandas as pd


MARKDOWN_COLUMNS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
NUMERIC_IMPUTE_COLUMNS = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]
NUMERIC_IMPUTE_STRATEGIES = {
    "Temperature": "mean",
    "Fuel_Price": "mean",
    "CPI": "median",
    "Unemployment": "median",
    "Size": "median",
}
SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


def merge_walmart_data(
    sales_df: pd.DataFrame,
    features_df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge train/test sales rows with feature and store metadata."""
    df = sales_df.merge(
        features_df,
        on=["Store", "Date"],
        how="left",
        suffixes=("", "_feature"),
    )
    if "IsHoliday_feature" in df.columns:
        df = df.drop(columns=["IsHoliday_feature"])
    df = df.merge(stores_df, on="Store", how="left")
    return df


def clean_walmart_data(
    df: pd.DataFrame,
    impute_values: dict | None = None,
    type_mode: str | None = None,
    remove_negative_sales: bool = True,
) -> tuple[pd.DataFrame, dict, str]:
    """Clean data and replace blank/NA/null values using business-safe defaults."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(bool).astype(int)

    for col in MARKDOWN_COLUMNS:
        if col in df.columns:
            markdown_values = pd.to_numeric(df[col], errors="coerce")
            positive_values = markdown_values[markdown_values > 0].dropna()
            if not positive_values.empty:
                fill_value = float(positive_values.median())
            else:
                fallback_values = markdown_values.dropna()
                fill_value = float(fallback_values.median()) if not fallback_values.empty else 0.0
            df[col] = markdown_values.fillna(fill_value)
            if impute_values is None:
                impute_values = {}
            impute_values[col] = fill_value

    if impute_values is None:
        impute_values = {}
        for col in NUMERIC_IMPUTE_COLUMNS:
            if col in df.columns:
                numeric_values = pd.to_numeric(df[col], errors="coerce")
                strategy = NUMERIC_IMPUTE_STRATEGIES.get(col, "median")
                if strategy == "mean":
                    impute_values[col] = float(numeric_values.mean())
                else:
                    impute_values[col] = float(numeric_values.median())

    for col, value in impute_values.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(value)

    if "Type" in df.columns:
        if type_mode is None:
            type_mode = df["Type"].mode(dropna=True).iloc[0]
        df["Type"] = df["Type"].fillna(type_mode)
    else:
        type_mode = type_mode or "A"

    if "Weekly_Sales" in df.columns:
        df["Weekly_Sales"] = pd.to_numeric(df["Weekly_Sales"], errors="coerce")
        if remove_negative_sales:
            df = df[df["Weekly_Sales"] >= 0].copy()

    return df, impute_values, type_mode


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create date and season features."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["ISO_Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["Day_Of_Month"] = df["Date"].dt.day
    df["Season"] = df["Month"].map(SEASON_MAP)
    return df


def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create promotion features from markdown columns."""
    df = df.copy()
    available_markdowns = [col for col in MARKDOWN_COLUMNS if col in df.columns]
    df["Total_Markdown_Amount"] = df[available_markdowns].sum(axis=1) if available_markdowns else 0
    df["Has_Markdown_Flag"] = (df["Total_Markdown_Amount"] > 0).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "Weekly_Sales",
) -> pd.DataFrame:
    """Add leakage-safe lag and rolling features for historical training rows."""
    if group_cols is None:
        group_cols = ["Store", "Dept"]

    df = df.sort_values(group_cols + ["Date"]).copy()
    grouped = df.groupby(group_cols, sort=False)[target_col]
    target_default = float(pd.to_numeric(df[target_col], errors="coerce").median())

    for lag in [1, 4, 8, 13, 52]:
        df[f"Sales_Lag_{lag}_Weeks"] = grouped.shift(lag).fillna(target_default)

    df["Sales_Rolling_Mean_4_Weeks"] = grouped.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    df["Sales_Rolling_Std_4_Weeks"] = grouped.transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())
    df["Sales_Rolling_Mean_8_Weeks"] = grouped.transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    df["Sales_Rolling_Std_8_Weeks"] = grouped.transform(lambda s: s.shift(1).rolling(8, min_periods=2).std())
    df["Sales_Rolling_Mean_13_Weeks"] = grouped.transform(lambda s: s.shift(1).rolling(13, min_periods=1).mean())

    rolling_columns = [
        "Sales_Rolling_Mean_4_Weeks",
        "Sales_Rolling_Std_4_Weeks",
        "Sales_Rolling_Mean_8_Weeks",
        "Sales_Rolling_Std_8_Weeks",
        "Sales_Rolling_Mean_13_Weeks",
    ]
    for col in rolling_columns:
        df[col] = df[col].fillna(float(df[col].median()))

    return df


def apply_iqr_filter(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """Remove target outliers using the IQR rule."""
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = df[target_col].between(lower, upper)
    bounds = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower": float(lower),
        "upper": float(upper),
        "removed_rows": int((~mask).sum()),
    }
    return df[mask].copy(), bounds


def build_model_matrix(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    """Create one-hot encoded model features and optional target."""
    df = df.copy()
    categorical_cols = [col for col in ["Type", "Season"] if col in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)

    blocked_cols = {
        target_col,
        "Date",
    }
    if feature_columns is None:
        feature_columns = [
            col
            for col in df.columns
            if col not in blocked_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

    X = df.reindex(columns=feature_columns)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(X.mean(numeric_only=True))
    X = X.fillna(0)
    y = df[target_col] if target_col in df.columns else None
    return X, y, feature_columns
