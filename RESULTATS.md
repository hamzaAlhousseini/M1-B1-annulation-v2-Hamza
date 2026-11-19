# Modèle d'annulation v2 — Résultats

## Démarche

Les données sont séparées chronologiquement : 58 792 réservations du 01/07/2015 au 31/08/2016 pour l’entraînement et 19 911 du 01/09/2016 au 31/12/2016 pour le test.

Les colonnes révélant l’état final des réservations sont supprimées afin d’éviter les fuites de données.

Trois modèles sont comparés avec une validation croisée stratifiée à 3 plis : régression logistique, forêt aléatoire et HistGradientBoosting.

Leurs hyperparamètres sont ensuite réglés avec GridSearchCV, en utilisant le F1-score comme métrique principale.

Pour la forêt aléatoire, les essais testent max_depth et min_samples_leaf.

La configuration retenue est n_estimators=100, max_depth=20 et min_samples_leaf=2, avec un écart entraînement-validation de 0,0316.

## Données

| | Période d'arrivée | Nombre de réservations | Taux d'annulation |
|---|---|---|---|
| Entraînement | 2015-07-01 au 2016-08-31 | 58 792 | 35,56 % |
| Test | 2016-09-01 au 2016-12-31 | 19 911 | 38,02 % |

## Fuite de données (tâche 4)

Lors de l’exploration, le modèle utilisant toutes les colonnes obtenait des scores presque parfaits, ce qui a révélé une fuite de données. reservation_status indique directement si la réservation a été annulée, et reservation_status_date peut révéler quand elle l’a été. Ces colonnes, ainsi que reservation_id et assigned_room_type, ont donc été exclues des variables utilisées pour l’entraînement : elles peuvent révéler le résultat ou contenir des informations indisponibles au moment de la prédiction. Le code retient plutôt des variables disponibles sur la réservation, comme le délai, la composition du séjour et le type de client.

## Variables nouvelles

| Variable | Calcul | Justification métier |
|---|---|---|
| `total_nights` | `stays_in_weekend_nights + stays_in_week_nights` | Représente la durée totale du séjour, qui peut influencer la probabilité d’annulation. |
| `has_children` | `1` si `children > 0`, sinon `0` | Indique si la réservation concerne un séjour avec enfants, un facteur potentiel dans le comportement d’annulation. |

## Validation et réglage

La comparaison initiale des modèles a été faite avec une validation croisée StratifiedKFold à 3 plis (shuffle=True, random_state=42), en prenant le F1 comme métrique principale. La forêt aléatoire a obtenu le meilleur F1 moyen (0,8455), mais son F1 d’entraînement élevé (0,9928) indiquait un risque de surapprentissage.

La recherche GridSearchCV a testé max_depth (None ou 20) et min_samples_leaf (2 ou 5), avec n_estimators=100. Le modèle retenu utilise max_depth=20 et min_samples_leaf=2 : son F1 moyen est de 0,8447 à l’entraînement et de 0,8131 en validation, soit un écart de 0,0316. Cet écart, inférieur au seuil fixé de 0,05, indique un surapprentissage plus limité.

## Choix de la métrique

Une annulation prédite à tort (faux positif) peut amener l’hôtel à libérer une chambre alors que le client compte venir, avec un risque de ne plus pouvoir l’accueillir. Une annulation ratée (faux négatif) empêche l’hôtel d’anticiper et de remettre la chambre en vente, ce qui peut entraîner une perte de revenu. Comme les coûts précis de ces deux erreurs ne sont pas connus, le F1-score est retenu pour équilibrer précision et rappel.

## Verdict : v1 rejouée contre v2, sur le jeu de test

| | v1 (recette rejouée) | v2 |
|---|---:|---:|
| F1 (métrique principale) | **0,5752** | 0,5475 |
| Accuracy / Précision / Rappel / ROC-AUC | 0,7655 / 0,9235 / 0,4177 / 0,8669 | 0,7598 / 0,9647 / 0,3822 / 0,8851 |
| Temps d’entraînement | 5,49 s | 7,97 s |
| Taille du modèle | 0,75 Mo | 22,74 Mo |

Matrices de confusion (`[[TN, FP], [FN, TP]]`) :

```text
v1 : [[12079,   262],
      [ 4408,  3162]]

v2 : [[12235,   106],
      [ 4677,  2893]]
```

Matrices de confusion ([[TN, FP], [FN, TP]]) :

| | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| v1 | 12 079 | 262 | 4 408 | 3 162 |
| v2 | 12 235 | 106 | 4 677 | 2 893 |

## Recommandation

La v2 n’est pas à privilégier pour une mise en service si l’objectif principal est le F1 : sur le jeu de test, son F1 (0,5475) et son rappel (0,3822) sont inférieurs à ceux de la v1. Elle détecte moins d’annulations, même si sa précision et son ROC-AUC sont meilleurs.

Ces résultats restent à confirmer sur d’autres périodes. Avant tout déploiement, il faut évaluer le coût réel des deux types d’erreurs et ajuster le seuil de décision. En attendant, aucun des modèles ne devrait déclencher seul une action de surréservation.
