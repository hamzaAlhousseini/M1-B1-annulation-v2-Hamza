"""Entraînement du modèle d'annulation v2 — Escale Atlantique (mission FastIA).

Tâche 9 : reporte ici la démarche finale mise au point dans tes notebooks.

"""

import json
import os
import time
from datetime import date
from pathlib import Path

import joblib
import pandas as pd  
import sklearn
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, text
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_selector
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_MONTHS = 4
TARGET = "is_canceled"

load_dotenv(ROOT / ".env")


def load_data(engine) -> pd.DataFrame:
    query = text("""
    SELECT
        r.*,
        h.code AS hotel
    FROM reservations AS r
    INNER JOIN hotels AS h
        ON r.hotel_id = h.hotel_id
    """)

    df = pd.read_sql(query, engine)

    # Convertir la date d'arrivée
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])

    df.head()

    return df

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_limite = df["arrival_date"].max() - pd.DateOffset(months=4)

    # Entraînement sur les données passées
    train = df[df["arrival_date"] <= date_limite].copy()

    # Test sur les données les plus récentes
    test = df[df["arrival_date"] > date_limite].copy()

    return train, test


def build_pipeline() -> Pipeline:
    """
    Construit un pipeline de classification basé sur une forêt aléatoire.
    Gère les valeurs manquantes et encode les variables catégorielles.
    """
    # Prétraitement des variables catégorielles
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Prétraitement des variables numériques
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1))
    ])

    # Combiner les prétraitements
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline,
             make_column_selector(dtype_include=["object", "category"])),
            ("numerical", numerical_pipeline,
             make_column_selector(dtype_include=["number"]))
        ]
    )

    # Pipeline complet
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=3
        ))
    ])

    return pipeline

def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Calcule les métriques de classification sur les données fournies."""
    y_pred = model.predict(X)

    metrics = {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = round(roc_auc_score(y, y_proba), 4)

    return metrics


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"])
    df = load_data(engine)
    train, test = split_train_test(df)
    print(f"Entraînement : {len(train)} réservations | test : {len(test)} réservations")

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    model = build_pipeline()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = round(time.perf_counter() - start, 1)

    metrics = evaluate(model, X_test, y_test)
    print("Métriques sur le jeu de test :", metrics)

    model_path = MODEL_DIR / "cancellation_model_v2.joblib"
    joblib.dump(model, model_path)
    info = {
        "model_name": "cancellation_model",
        "version": "2.0.0",
        "trained_on": date.today().isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "features": list(X_train.columns),
        "target": TARGET,
        "train_period": [str(train["arrival_date"].min()), str(train["arrival_date"].max())],
        "test_period": [str(test["arrival_date"].min()), str(test["arrival_date"].max())],
        "hyperparameters": {
            k: v for k, v in model.get_params().items()
            if k.startswith("classifier__") and isinstance(v, (int, float, str, bool, type(None)))
        },
        "test_metrics": metrics,
        "training_seconds": training_seconds,
        "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
    }
    (MODEL_DIR / "model_info_v2.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Modèle et fiche sauvegardés dans", MODEL_DIR)


if __name__ == "__main__":
    main()
