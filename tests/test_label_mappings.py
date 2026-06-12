import pandas as pd

from src.feature_eng import build_label_mappings


def test_build_label_mappings_returns_human_friendly_labels():
    df = pd.DataFrame(
        {
            "Store": [1, 2, 1],
            "Dept": [10, 20, 10],
            "Type": ["A", "B", "A"],
            "IsHoliday": [0, 1, 0],
        }
    )

    mappings = build_label_mappings(df)

    assert mappings["Store"][1] == "North Plaza Store"
    assert mappings["Dept"][10] == "Clothing"
    assert mappings["Type"]["A"] == "Value Format"
    assert mappings["IsHoliday"][1] == "Holiday Week"
