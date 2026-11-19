import os
import time

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model.train_baseline import FEATURES, build_pipeline as build_v1_pipeline
from pipeline import creer_pipeline
from train import MODEL_DIR, TARGET, load_data, split_train_test


def evaluer_une_fois(nom, modele, X_train, y_train, X_test, y_test, chemin):
    debut = time.perf_counter()
    modele.fit(X_train, y_train)
    temps_entrainement = time.perf_counter() - debut

    joblib.dump(modele, chemin)
    taille_octets = chemin.stat().st_size

    predictions = modele.predict(X_test)
    probabilites = modele.predict_proba(X_test)[:, 1]
    metriques = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilites),
    }

    return {
        "name": nom,
        "metrics": metriques,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "training_seconds": temps_entrainement,
        "model_size_bytes": taille_octets,
        "model_path": chemin,
    }


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"])
    train_df, test_df = split_train_test(load_data(engine))
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print(
        f"Découpage commun : {len(X_train)} réservations d'entraînement, "
        f"{len(X_test)} de test."
    )
    print(
        "Période d'entraînement : "
        f"{train_df['arrival_date'].min().date()} au "
        f"{train_df['arrival_date'].max().date()}"
    )
    print(
        "Période de test : "
        f"{test_df['arrival_date'].min().date()} au "
        f"{test_df['arrival_date'].max().date()}"
    )
    print("Métrique principale : F1 (compromis précision/rappel).")

    colonnes_numeriques = X_train.select_dtypes(include=np.number).columns.tolist()
    colonnes_categorielles = X_train.select_dtypes(exclude=np.number).columns.tolist()
    v2_pipeline = creer_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=3,
        ),
        colonnes_numeriques,
        colonnes_categorielles,
    ).set_params(modele__max_depth=20, modele__min_samples_leaf=2)

    resultats = [
        evaluer_une_fois(
            "v1 rejouée",
            build_v1_pipeline(),
            X_train,
            y_train,
            X_test,
            y_test,
            MODEL_DIR / "cancellation_model_v1_replayed.joblib",
        ),
        evaluer_une_fois(
            "v2 réglée",
            v2_pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            MODEL_DIR / "cancellation_model_v2_final.joblib",
        ),
    ]

    lignes = []
    for resultat in resultats:
        metriques = resultat["metrics"]
        lignes.append(
            {
                "Modèle": resultat["name"],
                "Accuracy": metriques["accuracy"],
                "Précision": metriques["precision"],
                "Rappel": metriques["recall"],
                "F1": metriques["f1"],
                "ROC-AUC": metriques["roc_auc"],
                "Temps entraînement (s)": resultat["training_seconds"],
                "Taille modèle (MB)": resultat["model_size_bytes"] / 1_000_000,
            }
        )
        print(f"\nMatrice de confusion — {resultat['name']} (lignes=réel, colonnes=prédit):")
        print(pd.DataFrame(
            resultat["confusion_matrix"],
            index=["Réel 0", "Réel 1"],
            columns=["Prédit 0", "Prédit 1"],
        ))
        print("Modèle sauvegardé :", resultat["model_path"])

    print("\nComparaison finale sur le test :")
    print(
        pd.DataFrame(lignes).to_string(
            index=False,
            float_format=lambda valeur: f"{valeur:.4f}",
        )
    )


if __name__ == "__main__":
    main()
