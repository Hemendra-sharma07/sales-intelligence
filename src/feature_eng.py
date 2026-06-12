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

STORE_LABELS = {
    1: "North Plaza Store",
    2: "Downtown Market",
    3: "Riverfront Superstore",
    4: "Westfield Center",
    5: "Sunset Retail Hub",
    6: "Midtown Express",
    7: "Harbor Point Store",
    8: "Green Valley Market",
    9: "Lakeside Outlet",
    10: "Cedar Grove Store",
    11: "Elm Street Shop",
    12: "Oak Ridge Market",
    13: "Pine Avenue Store",
    14: "Central Square Retail",
    15: "Hilltop Warehouse",
    16: "Brookside Store",
    17: "Maple Grove Market",
    18: "Crestview Shop",
    19: "Summit Point Store",
    20: "Valley View Market",
    21: "Highland Express",
    22: "Parkside Retail",
    23: "Orchard Lane Store",
    24: "Beacon Hill Market",
    25: "Meadowbrook Shop",
    26: "Royal Oak Store",
    27: "Silver Creek Market",
    28: "Fairway Plaza",
    29: "Stonebridge Store",
    30: "Garden City Market",
    31: "Liberty Square Store",
    32: "Heritage Plaza",
    33: "Canyon Ridge Store",
    34: "Copper Creek Market",
    35: "Timberline Store",
    36: "Magnolia Park Shop",
    37: "Golden Gate Store",
    38: "Brookfield Market",
    39: "Franklin Square Store",
    40: "Westbrook Market",
    41: "Eastgate Retail",
    42: "Pinecrest Store",
    43: "Canyon View Market",
    44: "Redwood Square Store",
    45: "Fairfield Hub",
}

DEPARTMENT_LABELS = {
    1: "Electronics",
    2: "Home Appliances",
    3: "Furniture",
    4: "Decor & Lighting",
    5: "Outdoor & Gardening",
    6: "Sports & Fitness",
    7: "Automotive",
    8: "Toys & Games",
    9: "Books & Media",
    10: "Clothing",
    11: "Shoes",
    12: "Jewelry & Accessories",
    13: "Beauty & Personal Care",
    14: "Health & Wellness",
    15: "Pet Supplies",
    16: "Grocery",
    17: "Bakery",
    18: "Dairy",
    19: "Frozen Foods",
    20: "Pantry Staples",
    21: "Beverages",
    22: "Household Essentials",
    23: "Cleaning Supplies",
    24: "Baby & Kids",
    25: "Office Supplies",
    26: "Tech Accessories",
    27: "Musical Instruments",
    28: "Craft & Hobby",
    29: "Party Supplies",
    30: "Seasonal Decor",
    31: "Luggage",
    32: "Camping Gear",
    33: "Lawn & Garden",
    34: "Tools & Hardware",
    35: "Building Materials",
    36: "Pharmacy",
    37: "Optical",
    38: "Mattress & Sleep",
    39: "Kitchenware",
    40: "Dining & Entertaining",
    41: "Bath & Shower",
    42: "Bedding",
    43: "Laundry & Fabric Care",
    44: "Paper Goods",
    45: "Small Appliances",
    46: "Smart Home",
    47: "Video Games",
    48: "Mobile Phones",
    49: "Cameras & Photography",
    50: "Wearable Tech",
    51: "Fitness Wear",
    52: "Men's Apparel",
    53: "Women's Apparel",
    54: "Kids Apparel",
    55: "Footwear",
    56: "Handbags",
    57: "Watches",
    58: "Makeup",
    59: "Skin Care",
    60: "Hair Care",
    61: "Oral Care",
    62: "Vitamins",
    63: "First Aid",
    64: "Pet Food",
    65: "Animal Care",
    66: "Fresh Produce",
    67: "Meat & Seafood",
    68: "Deli",
    69: "Refrigerated",
    70: "Bread & Bakery",
    71: "Coffee & Tea",
    72: "Snacks",
    73: "Condiments",
    74: "Canned Goods",
    75: "Soups & Broths",
    76: "Frozen Meals",
    77: "Ice Cream",
    78: "International Foods",
    79: "Gift Cards",
    80: "Seasonal Gifts",
    81: "Specialty Foods",
}


def build_label_mappings(df: pd.DataFrame) -> dict[str, dict]:
    """Create human-friendly labels for categorical values used in training and the app."""
    mappings: dict[str, dict] = {}

    def find_column(candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    if "Store" in df.columns:
        store_values = sorted({int(value) for value in pd.to_numeric(df["Store"], errors="coerce").dropna().unique()})
        mappings["Store"] = {
            value: STORE_LABELS.get(value, f"Store {value}") for value in store_values
        }

    if "Dept" in df.columns:
        dept_values = sorted({int(value) for value in pd.to_numeric(df["Dept"], errors="coerce").dropna().unique()})
        mappings["Dept"] = {
            value: DEPARTMENT_LABELS.get(value, f"Department {value}") for value in dept_values
        }

    if "Type" in df.columns:
        type_values = [value for value in df["Type"].dropna().astype(str).unique() if str(value) != "nan"]
        type_lookup = {"A": "Value Format", "B": "Mid-Market Format", "C": "Premium Format"}
        mappings["Type"] = {
            value: type_lookup.get(str(value), f"Store Type {value}") for value in sorted(type_values)
        }

    holiday_col = find_column(["IsHoliday", "IsHoliday_x", "IsHoliday_y"])
    if holiday_col is not None:
        mappings["IsHoliday"] = {0: "Regular Week", 1: "Holiday Week"}

    return mappings


def add_display_labels(df: pd.DataFrame, mappings: dict[str, dict] | None = None) -> pd.DataFrame:
    """Append label columns such as Store_Label and Dept_Label without changing the model inputs."""
    df = df.copy()
    if mappings is None:
        mappings = build_label_mappings(df)

    for column in ["Store", "Dept", "Type", "IsHoliday"]:
        if column in df.columns and column in mappings:
            df[f"{column}_Label"] = df[column].map(lambda value: mappings[column].get(value, str(value)))

    return df


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
