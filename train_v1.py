import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from train import evaluate, load_data, MODEL_DIR
from model.train_baseline import build_pipeline, FEATURES, TARGET


def prepare_features_target_and_dates():
    # Créer une connexion à la base de données SQLite
    engine = create_engine(os.environ["DATABASE_URL"])

    # Charger les données
    df = load_data(engine)

    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(
            "Variables attendues absentes du jeu de donnees : "
            + ", ".join(missing_features)
        )

    # Utiliser les colonnes de la recette v1, avant tout encodage.
    X = df[FEATURES].copy()
    y = df[TARGET]
    arrival_dates = df["arrival_date"]

    print("Jeu de données préparé pour le modèle :")
    print("Nombre d'échantillons :", X.shape[0])

    return X, y, arrival_dates


def separate_features_target_and_dates():
    X, y, arrival_dates = prepare_features_target_and_dates()

    # Séparation des données
    return train_test_split(
        X,
        y,
        arrival_dates,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def separate_features_and_target():
    X_train, X_test, y_train, y_test, _, _ = separate_features_target_and_dates()
    return X_train, X_test, y_train, y_test


def main():   

    X_train, X_test, y_train, y_test = separate_features_and_target()

    model = build_pipeline()
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print("Métriques sur le jeu de test :", metrics)

    # Sauvegarde du modèle
    joblib.dump(model, MODEL_DIR / "cancellation_model_v1.joblib")

if __name__ == "__main__":
    main()