from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from train_v1 import prepare_features_target_and_dates
from model.train_baseline import build_pipeline
import pandas as pd
from IPython.display import display


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv, label: str):
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = cross_validate(
        model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=True
    )

    scores = pd.DataFrame({
        "Accuracy": results["test_accuracy"],
        "Precision": results["test_precision"],
        "Recall": results["test_recall"],
        "Train F1": results["train_f1"],
        "Validation F1": results["test_f1"],
        "Écart train-validation": results["train_f1"] - results["test_f1"],
        "ROC-AUC": results["test_roc_auc"]
    })

    print(f"\n{label} — résultats par fold (test_f1 = validation) :")
    display(scores)

    print(f"\n{label} — moyenne et écart-type par métrique :")
    display(scores.agg(["mean", "std"]).T)


def search_hyperparameters(X: pd.DataFrame, y: pd.Series):
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )
    param_grid = {
        "classifier__max_iter": [100, 200],
        "classifier__learning_rate": [0.05, 0.1],
    }

    grid_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=False,
    )
    grid_search.fit(X, y)

    results = pd.DataFrame(grid_search.cv_results_)[[
        "param_classifier__max_iter",
        "param_classifier__learning_rate",
        "mean_test_score",
        "std_test_score",
    ]].rename(columns={
        "param_classifier__max_iter": "max_iter",
        "param_classifier__learning_rate": "learning_rate",
        "mean_test_score": "F1 moyen",
        "std_test_score": "Écart-type F1",
    })

    print("\nRésultats de GridSearchCV :")
    display(results)


def main():
    X, y, arrival_dates = prepare_features_target_and_dates()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        dates_train,
        _,
    ) = train_test_split(
        X,
        y,
        arrival_dates,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print("Jeu de test :", len(X_test), "echantillons")

    model_v1 = build_pipeline()

    stratified_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )
    evaluate(model_v1, X_train, y_train, stratified_cv, "StratifiedKFold")

    chronological_order = dates_train.sort_values(kind="stable").index
    X_train_chronological = X_train.loc[chronological_order]
    y_train_chronological = y_train.loc[chronological_order]
    time_series_cv = TimeSeriesSplit(n_splits=5)
    evaluate(
        model_v1,
        X_train_chronological,
        y_train_chronological,
        time_series_cv,
        "TimeSeriesSplit",
    )

    search_hyperparameters(X_train, y_train)

    model_v1.fit(X_train, y_train)
    model_v1.predict(X_test)



if __name__ == "__main__":
    main()





