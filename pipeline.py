import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from model.pipeline_features import ajouter_variables
from train import MODEL_DIR, evaluate
from train_v1 import separate_features_and_target


def creer_preparation(
    colonnes_numeriques: list[str],
    colonnes_categorielles: list[str],
    *,
    mise_a_echelle: bool = False,
    sortie_dense: bool = False,
) -> ColumnTransformer:
    etapes_numeriques = [("impute", SimpleImputer(strategy="median"))]
    if mise_a_echelle:
        etapes_numeriques.append(("scale", StandardScaler()))

    pipeline_categoriel = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not sortie_dense,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=etapes_numeriques),
                colonnes_numeriques,
            ),
            ("cat", pipeline_categoriel, colonnes_categorielles),
        ],
        sparse_threshold=0 if sortie_dense else 0.3,
    )


def creer_pipeline(
    modele,
    colonnes_numeriques: list[str],
    colonnes_categorielles: list[str],
    *,
    mise_a_echelle: bool = False,
    sortie_dense: bool = False,
) -> Pipeline:
    preparation = creer_preparation(
        colonnes_numeriques + ["total_nights", "has_children"],
        colonnes_categorielles,
        mise_a_echelle=mise_a_echelle,
        sortie_dense=sortie_dense,
    )
    return Pipeline(
        steps=[
            ("variables", FunctionTransformer(ajouter_variables, validate=False)),
            ("preparation", preparation),
            ("modele", modele),
        ]
    )


def main() -> None:
    X_train, X_test, y_train, y_test = separate_features_and_target()

    colonnes_numeriques = X_train.select_dtypes(include=np.number).columns.tolist()
    colonnes_categorielles = X_train.select_dtypes(exclude=np.number).columns.tolist()

    candidats = [
        (
            "Régression logistique",
            LogisticRegression(max_iter=1000, solver="liblinear"),
            True,
            False,
            {
                "modele__C": [0.1, 1.0, 10.0],
                "modele__class_weight": [None, "balanced"],
            },
        ),
        (
            "Forêt aléatoire",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=3,
            ),
            False,
            False,
            {
                "modele__max_depth": [None, 20],
                "modele__min_samples_leaf": [2, 5],
            },
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(
                max_iter=150,
                random_state=42,
            ),
            False,
            True,
            {
                "modele__max_iter": [100, 150],
                "modele__learning_rate": [0.05, 0.1],
            },
        ),
    ]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = {
        "f1": "f1",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
    }

    pipelines = {}
    grilles = {}
    lignes_resultats = []
    for nom, modele, mise_a_echelle, sortie_dense, grille in candidats:
        pipeline = creer_pipeline(
            modele,
            colonnes_numeriques,
            colonnes_categorielles,
            mise_a_echelle=mise_a_echelle,
            sortie_dense=sortie_dense,
        )
        resultats = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        pipelines[nom] = pipeline
        grilles[nom] = grille
        lignes_resultats.append(
            {
                "Modèle": nom,
                "F1 entraînement": np.mean(resultats["train_f1"]),
                "F1 moyen": np.mean(resultats["test_f1"]),
                "Écart train-validation": (
                    np.mean(resultats["train_f1"])
                    - np.mean(resultats["test_f1"])
                ),
                "Écart-type F1": np.std(resultats["test_f1"]),
                "ROC-AUC moyen": np.mean(resultats["test_roc_auc"]),
                "Accuracy moyenne": np.mean(resultats["test_accuracy"]),
            }
        )

    comparaison = pd.DataFrame(lignes_resultats).sort_values(
        by="F1 moyen",
        ascending=False,
    )
    print("Comparaison des modèles (validation croisée stratifiée, 3 folds) :")
    print(comparaison.to_string(index=False, float_format=lambda valeur: f"{valeur:.4f}"))

    meilleur_nom = comparaison.iloc[0]["Modèle"]
    print(f"\nRéglage des hyperparamètres : {meilleur_nom}")
    recherche = GridSearchCV(
        estimator=pipelines[meilleur_nom],
        param_grid=grilles[meilleur_nom],
        scoring="f1",
        cv=cv,
        n_jobs=1,
        return_train_score=True,
        refit=False,
        error_score="raise",
    )
    recherche.fit(X_train, y_train)

    cles_parametres = list(grilles[meilleur_nom])
    trace = pd.DataFrame(recherche.cv_results_)
    colonnes_trace = [
        *(f"param_{cle}" for cle in cles_parametres),
        "mean_train_score",
        "std_train_score",
        "mean_test_score",
        "std_test_score",
    ]
    trace = trace[colonnes_trace].rename(
        columns={
            **{
                f"param_{cle}": cle.removeprefix("modele__")
                for cle in cles_parametres
            },
            "mean_train_score": "F1 entraînement moyen",
            "std_train_score": "Écart-type F1 entraînement",
            "mean_test_score": "F1 validation moyen",
            "std_test_score": "Écart-type F1 validation",
        }
    )
    trace["Écart train-validation"] = (
        trace["F1 entraînement moyen"] - trace["F1 validation moyen"]
    )
    trace = trace.sort_values("F1 validation moyen", ascending=False)

    print("\nTrace complète des essais GridSearchCV :")
    print(trace.to_string(index=False, float_format=lambda valeur: f"{valeur:.4f}"))
    chemin_trace = MODEL_DIR / "pipeline_grid_search_results.csv"
    trace.to_csv(chemin_trace, index=False, encoding="utf-8-sig")

    seuil_ecart = 0.05
    essais_stables = trace[trace["Écart train-validation"] <= seuil_ecart]
    if essais_stables.empty:
        meilleur_essai = trace.iloc[0]
        print(
            "\nAucun essai ne respecte l'écart maximal de "
            f"{seuil_ecart:.2f} ; sélection du meilleur F1 validation."
        )
    else:
        meilleur_essai = essais_stables.iloc[0]
        print(
            "\nSélection du meilleur F1 validation parmi les essais dont "
            f"l'écart est <= {seuil_ecart:.2f}."
        )

    parametres_retenus = recherche.cv_results_["params"][int(meilleur_essai.name)]
    meilleur_pipeline = clone(pipelines[meilleur_nom]).set_params(**parametres_retenus)
    meilleur_pipeline.fit(X_train, y_train)

    print("\nParamètres retenus :", parametres_retenus)
    print(
        "F1 entraînement / validation : "
        f"{meilleur_essai['F1 entraînement moyen']:.4f} / "
        f"{meilleur_essai['F1 validation moyen']:.4f} "
        f"(écart {meilleur_essai['Écart train-validation']:.4f})"
    )
    print("Trace sauvegardée dans :", chemin_trace)

    print(f"\nMeilleur modèle selon le F1 CV : {meilleur_nom}")
    print("Scores sur le jeu de test :", evaluate(meilleur_pipeline, X_test, y_test))

    chemin_modele = MODEL_DIR / "cancellation_model_pipeline.joblib"
    joblib.dump(meilleur_pipeline, chemin_modele)
    print("Pipeline complet sauvegardé dans :", chemin_modele)


if __name__ == "__main__":
    main()
