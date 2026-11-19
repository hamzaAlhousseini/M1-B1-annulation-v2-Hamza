# Modèle d'annulation v2 — Résultats

> À compléter pendant le brief (tâche 9). Les passages en italique sont des consignes, à remplacer.

## Démarche

*En 5 à 10 lignes : données utilisées, découpage, modèles essayés, réglages, choix final.*

## Données

| | Période d'arrivée | Nombre de réservations | Taux d'annulation |
|---|---|---|---|
| Entraînement | | | |
| Test | | | |

## Fuite de données (tâche 4)

*Ce que tu as observé, comment tu l'as détecté, quelles colonnes tu as retirées et pourquoi.*

## Variables nouvelles

| Variable | Calcul | Justification métier |
|---|---|---|
| | | |

## Validation et réglage

*Validation croisée utilisée, hyperparamètres testés et retenus, écart entre score d'entraînement et score de validation, et ce que tu en as conclu.*

## Choix de la métrique

La métrique principale retenue est le **F1-score**, car les deux erreurs ont un coût opérationnel : une fausse alerte peut conduire à libérer une chambre alors que le client arrive, tandis qu'une annulation ratée empêche de remettre la chambre en vente. Faute de coûts financiers fiables pour pondérer ces erreurs, le F1 offre un compromis entre précision et rappel. Les deux métriques restent affichées pour rendre ce compromis visible.

## Verdict : v1 rejouée contre v2, sur le jeu de test

Comparaison effectuée sur le même découpage chronologique : entraînement du 2015-07-01 au 2016-08-31 (58 792 réservations), test du 2016-09-01 au 2016-12-31 (19 911 réservations). Le test n'a pas servi au réglage des hyperparamètres.

La v1 reprend la recette baseline (HistGradientBoosting). La v2 est la forêt aléatoire réglée (`n_estimators=100`, `max_depth=20`, `min_samples_leaf=2`), retenue parmi les essais dont l'écart F1 entraînement-validation ne dépasse pas 0,05.

| Métrique | v1 (recette rejouée) | v2 réglée |
|---|---:|---:|
| F1 (principale) | **0,5752** | 0,5475 |
| Accuracy | 0,7655 | **0,7598** |
| Précision | 0,9235 | **0,9647** |
| Rappel | **0,4177** | 0,3822 |
| ROC-AUC | 0,8669 | **0,8851** |
| Temps d'entraînement | **4,53 s** | 9,14 s |
| Taille du modèle | **0,75 Mo** | 22,74 Mo |

Matrices de confusion (lignes = classe réelle, colonnes = classe prédite ; `[[TN, FP], [FN, TP]]`) :

```text
v1 : [[12079,   262],
      [ 4408,  3162]]

v2 : [[12235,   106],
      [ 4677,  2893]]
```

La v2 améliore la précision et réduit les faux positifs (262 → 106), mais elle détecte moins d'annulations : son rappel baisse et elle produit 4 677 faux négatifs contre 4 408 pour la v1. Le F1 de la v1 est supérieur de 0,0277.

## Recommandation

Avec le F1 comme objectif, la v1 est préférable sur ce test : elle a un meilleur F1 et un meilleur rappel, tout en étant plus rapide et beaucoup plus petite. La v2 peut être préférable si éviter les fausses alertes est la priorité métier. Aucun des deux modèles ne devrait déclencher seul une action de surréservation : le rappel reste faible et le test temporel est nettement plus difficile que la validation aléatoire. Avant déploiement, il faudrait fixer les coûts relatifs des deux erreurs et choisir le seuil de décision en conséquence.
