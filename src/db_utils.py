from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATABASE_DIR = PROJECT_ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "inventory.db"


def load_raw_data(raw_dir: Path | str = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four Walmart CSV files from data/raw."""
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(raw_dir / "test.csv", parse_dates=["Date"])
    features = pd.read_csv(raw_dir / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(raw_dir / "stores.csv")
    return train, test, features, stores


def load_data_to_sqlite(
    raw_dir: Path | str = RAW_DATA_DIR,
    db_path: Path | str = DATABASE_PATH,
) -> Path:
    """Create the SQLite database used by the dashboard and chat pages."""
    train, test, features, stores = load_raw_data(raw_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        train.to_sql("sales", conn, if_exists="replace", index=False)
        test.to_sql("future_sales", conn, if_exists="replace", index=False)
        features.to_sql("features", conn, if_exists="replace", index=False)
        stores.to_sql("stores", conn, if_exists="replace", index=False)

    print(f"sales table:        {len(train):,} rows loaded")
    print(f"future_sales table: {len(test):,} rows loaded")
    print(f"features table:     {len(features):,} rows loaded")
    print(f"stores table:       {len(stores):,} rows loaded")
    print(f"Database created:   {db_path}")
    return db_path


def get_conn(db_path: Path | str = DATABASE_PATH) -> sqlite3.Connection:
    """Return a SQLite connection to the project database."""
    return sqlite3.connect(Path(db_path))


def read_sales_master(db_path: Path | str = DATABASE_PATH) -> pd.DataFrame:
    """Read the merged historical sales table from SQLite."""
    query = """
        SELECT
            s.Store,
            s.Dept,
            s.Date,
            s.Weekly_Sales,
            s.IsHoliday,
            f.Temperature,
            f.Fuel_Price,
            f.MarkDown1,
            f.MarkDown2,
            f.MarkDown3,
            f.MarkDown4,
            f.MarkDown5,
            f.CPI,
            f.Unemployment,
            st.Type,
            st.Size
        FROM sales s
        LEFT JOIN features f
            ON s.Store = f.Store AND s.Date = f.Date
        LEFT JOIN stores st
            ON s.Store = st.Store
    """
    with get_conn(db_path) as conn:
        return pd.read_sql(query, conn, parse_dates=["Date"])


if __name__ == "__main__":
    load_data_to_sqlite()
