import pandas as pd


def ajouter_variables(df: pd.DataFrame) -> pd.DataFrame:
    resultat = df.copy()
    resultat["total_nights"] = (
        resultat["stays_in_weekend_nights"]
        + resultat["stays_in_week_nights"]
    )
    resultat["has_children"] = (
        pd.to_numeric(resultat["children"], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype("int8")
    )
    return resultat
