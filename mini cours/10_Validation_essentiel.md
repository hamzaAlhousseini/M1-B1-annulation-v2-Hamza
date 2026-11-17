# 10 — Valider et régler un modèle : l'essentiel

> **Utilisé dans** : M1-B1 (tâches 3, 7 et 8), puis les modules 4 et 6.
*

## 1. Découper selon l'usage


Un modèle ne sert à rien s'il est bon **sur les données qu'il a déjà vues**. Ce qui compte, c'est sa performance sur des données **nouvelles**, comme celles qu'il rencontrera en production.

On met donc de côté un **jeu de test**, qu'on n'utilise **qu'une seule fois, à la fin**, pour le verdict. Si on le consulte pour faire des choix (quelle variable garder, quel réglage), il cesse d'être « nouveau » et le score final devient trop optimiste.

Comment choisir ce jeu de test ? **En imitant l'usage réel du modèle.**

| Usage du modèle | Découpage adapté |
|---|---|
| Prédire pour des individus « au hasard » d'une même période | aléatoire (`train_test_split`, avec `stratify=y` pour garder la même proportion de classes) |
| Prédire **le futur** à partir du passé | **temporel** : on entraîne sur le passé, on teste sur la période la plus récente |
| Prédire pour de nouveaux clients, de nouveaux sites… | par groupe : aucun groupe n'est à la fois dans l'entraînement et dans le test |

Avec un découpage aléatoire sur un problème temporel, le modèle s'entraîne sur des données « du futur » et le score est flatteur. En production, il déçoit.

```python
date_limite = df["date"].max() - pd.DateOffset(months=4)
train = df[df["date"] <= date_limite]
test = df[df["date"] > date_limite]
```

## 2. La validation croisée : choisir sans toucher au test

Pour comparer des modèles ou des réglages, il faut un score fiable **sans utiliser le test**. La **validation croisée** découpe le jeu d'entraînement en *k* parties (*folds*) : on entraîne sur *k − 1* parties et on évalue sur la dernière, *k* fois de suite.

```
fold 1 : [VAL][ train ][ train ][ train ][ train ]
fold 2 : [ train ][VAL][ train ][ train ][ train ]
  ...
fold 5 : [ train ][ train ][ train ][ train ][VAL]
                                        → 5 scores → moyenne et écart-type
```

```python
from sklearn.model_selection import cross_validate

scores = cross_validate(
    pipeline, X_train, y_train,
    cv=5,
    scoring=["f1", "recall", "precision"],
    return_train_score=True,          # pour détecter le surapprentissage (section 4)
    n_jobs=-1,
)
pd.DataFrame(scores).mean()
```

**L'écart-type compte autant que la moyenne** : un score de 0,80 ± 0,01 est plus fiable que 0,82 ± 0,08.

Pour un problème temporel, `TimeSeriesSplit` respecte l'ordre chronologique (on valide toujours sur une période postérieure à l'entraînement). Il faut alors que les données soient **triées par date**.

## 3. Régler les hyperparamètres

Les **hyperparamètres** se fixent avant l'entraînement : profondeur maximale d'un arbre, nombre d'arbres, force de régularisation… Le modèle ne les apprend pas : c'est à toi de les choisir, en testant des combinaisons par validation croisée.

```python
from sklearn.model_selection import GridSearchCV

grille = {
    "classifier__max_depth": [4, 8, None],      # <nom de l'étape>__<paramètre>
    "classifier__learning_rate": [0.05, 0.1],
}
recherche = GridSearchCV(pipeline, grille, cv=5, scoring="f1", n_jobs=-1, return_train_score=True)
recherche.fit(X_train, y_train)

recherche.best_params_
recherche.best_score_
resultats = pd.DataFrame(recherche.cv_results_)   # la trace de TOUTES les combinaisons
```

- Dans un pipeline, un hyperparamètre se désigne par **le nom de l'étape, deux tirets bas, le nom du paramètre**. `pipeline.get_params().keys()` les liste tous.
- Une grille de 3 × 2 valeurs avec 5 folds, c'est **30 entraînements**. Commence petit.
- `RandomizedSearchCV` tire un nombre fixé de combinaisons au hasard : plus efficace quand il y a beaucoup d'hyperparamètres.
- `recherche.best_estimator_` est le pipeline ré-entraîné sur tout le jeu d'entraînement avec les meilleurs réglages.

## 4. Le surapprentissage

Un modèle **surapprend** quand il apprend les particularités de ses données d'entraînement au lieu des tendances générales. Le symptôme : un **score d'entraînement bien meilleur que le score de validation**.

| Score d'entraînement | Score de validation | Diagnostic |
|---|---|---|
| 0,99 | 0,75 | surapprentissage : le modèle « récite » |
| 0,78 | 0,76 | bon équilibre |
| 0,62 | 0,61 | sous-apprentissage : le modèle est trop simple |

Remèdes au surapprentissage : un modèle plus simple (arbres moins profonds, moins de variables), plus de régularisation, plus de données.

La **courbe d'apprentissage** (`sklearn.model_selection.learning_curve`) trace les deux scores en fonction de la quantité de données d'entraînement. Si l'écart se réduit encore quand on ajoute des données, en collecter davantage aiderait.

## 5. La fuite de données



Une **fuite de données** (*data leakage*), c'est quand le modèle a accès, pendant l'entraînement, à une information qu'il **n'aura pas** au moment de prédire. Le score de validation est excellent, et le modèle est inutilisable.

Deux formes fréquentes :

- **une variable connue trop tard** : elle n'existe qu'après l'événement à prédire, ou elle en est une conséquence ;
- **une préparation apprise sur toutes les données** : imputer, mettre à l'échelle ou sélectionner des variables **avant** de découper. Le pipeline évite ce piège, puisque chaque `fit` ne voit que les données d'entraînement du fold.

Le réflexe : **un score trop beau pour être vrai est suspect.** Regarde quelles variables pèsent le plus, et demande-toi, pour chacune, *à quel moment* elle est connue.

## 6. Choisir la métrique



Pour une classification binaire, tout part de la **matrice de confusion** :

| | Prédit : non | Prédit : oui |
|---|---|---|
| **Réel : non** | vrai négatif | **faux positif** (fausse alerte) |
| **Réel : oui** | **faux négatif** (cas raté) | vrai positif |

| Métrique | Question à laquelle elle répond |
|---|---|
| *Accuracy* | Quelle part des prédictions est juste ? Trompeuse si une classe est rare |
| **Précision** | Quand le modèle dit « oui », a-t-il raison ? (pénalise les fausses alertes) |
| **Rappel** (*recall*) | Parmi les vrais « oui », combien le modèle en trouve-t-il ? (pénalise les cas ratés) |
| F1 | Compromis entre précision et rappel |
| ROC AUC | Le modèle classe-t-il bien les cas les uns par rapport aux autres, quel que soit le seuil ? |

**On choisit la métrique à partir du coût des erreurs pour le métier**, pas l'inverse. Si rater un cas coûte cher, on surveille le rappel ; si une fausse alerte coûte cher, la précision.

```python
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
```

Le **seuil** de décision (0,5 par défaut avec `predict`) se règle aussi : on peut l'abaisser pour gagner du rappel au prix de la précision, à partir de `predict_proba`.

## 7. Exercice guidé — valider la v1 rejouée

On s'entraîne sur la recette de la v1 (tâche 5), ce qui te donnera aussi ta référence.

**Étape 1 — Validation croisée.** Évalue `build_pipeline()` (importé de `model/train_baseline.py`) avec `cross_validate`, 5 folds, F1, précision et rappel, et `return_train_score=True`.

✅ *Tu obtiens 5 scores par métrique. Note la moyenne et l'écart-type.*

**Étape 2 — Surapprentissage ?** Compare `train_f1` et `test_f1` (dans `cross_validate`, « test » désigne le fold de validation, pas ton jeu de test). L'écart est-il important ?

**Étape 3 — Une mini-recherche.** Teste 2 valeurs de `classifier__max_iter` et 2 valeurs de `classifier__learning_rate` avec `GridSearchCV`. Affiche `cv_results_` sous forme de `DataFrame`, en ne gardant que les paramètres, le score moyen et l'écart-type.

✅ *Un tableau de 4 lignes, une par combinaison.*

**Étape 4 — TimeSeriesSplit.** Refais l'étape 1 avec `cv=TimeSeriesSplit(n_splits=5)`, après avoir trié tes données par date d'arrivée. Les scores changent-ils ? Pourquoi, à ton avis ?

## 8. Pièges fréquents

| Symptôme | Cause probable |
|---|---|
| Score de test bien plus bas qu'en validation | Le jeu de test a servi à faire des choix, ou le découpage de la validation ne ressemble pas à celui du test |
| Score proche de 100 % | Fuite de données : chercher la variable « trop forte » |
| `Invalid parameter 'max_depth' for estimator Pipeline` | Il manque le préfixe de l'étape : `classifier__max_depth` |
| La recherche ne se termine pas | Grille trop grande, modèle lent : réduire la grille, `n_jobs=-1`, `RandomizedSearchCV` |
| Les scores varient beaucoup d'un fold à l'autre | Peu de données, ou des folds qui ne se ressemblent pas (saisonnalité) |
| `TimeSeriesSplit` donne des résultats incohérents | Les données ne sont pas triées par date |
| `UndefinedMetricWarning` sur la précision | Le modèle ne prédit jamais la classe positive sur un fold |
| *Accuracy* élevée mais modèle inutile | Classes déséquilibrées : regarder le rappel et la matrice de confusion |

## Pour aller plus loin

- Validation croisée : https://scikit-learn.org/stable/modules/cross_validation.html
- Réglage des hyperparamètres : https://scikit-learn.org/stable/modules/grid_search.html
- Métriques de classification : https://scikit-learn.org/stable/modules/model_evaluation.html
