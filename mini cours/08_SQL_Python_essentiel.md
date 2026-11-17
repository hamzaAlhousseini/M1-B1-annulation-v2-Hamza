# 08 — SQL depuis Python : l'essentiel

> **Utilisé dans** : M1-B1 (tâches 1 à 3), puis les modules 2 et 3 (pipeline de données).


## 1. Pourquoi requêter la base plutôt qu'un CSV ?

En M0, les données arrivaient en fichier. En entreprise, elles vivent dans une **base de données** : à jour, partagée, contrôlée. Un export CSV est une photo à un instant donné, qu'on oublie vite de rafraîchir.

Deux règles de bon sens :

- **Faire travailler la base** : filtrer, agréger, joindre en SQL. On ne rapatrie dans Python que ce dont on a besoin.
- **Rendre la requête rejouable** : c'est elle, et pas un fichier exporté à la main, qui documente d'où viennent les données d'entraînement.

## 2. Se connecter : SQLAlchemy et une URL de connexion

```python
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()                                   # lit le fichier .env
engine = create_engine(os.environ["DATABASE_URL"])
```

L'URL de connexion a cette forme :

```
postgresql+psycopg://utilisateur:mot_de_passe@hôte:port/base
          └─ pilote Python utilisé (psycopg 3)
```

**Jamais de mot de passe dans le code** ni dans un notebook versionné : il reste dans `.env`, qui est dans le `.gitignore`. Le `.env.example` versionné ne contient que des valeurs factices ou de développement local.

L'`engine` ne se connecte pas tout de suite : il ouvre une connexion à chaque requête, puis la rend.

## 3. Lire le résultat d'une requête dans pandas

```python
import pandas as pd
from sqlalchemy import text

df = pd.read_sql(text("SELECT * FROM hotels"), engine)
```

- `text(...)` indique à SQLAlchemy que la chaîne est du SQL brut.
- `parse_dates=["colonne"]` convertit une colonne en dates pandas.
- Pour explorer une grosse table, commence par `LIMIT 5`.

### Passer une valeur à une requête

Ne construis **jamais** une requête en collant des valeurs dans la chaîne (`f"... WHERE hotel_id = {x}"`) : c'est la porte ouverte aux erreurs et aux injections SQL. Utilise des **paramètres** :

```python
query = text("SELECT * FROM reservations WHERE hotel_id = :hotel_id LIMIT 10")
pd.read_sql(query, engine, params={"hotel_id": 1})
```

## 4. Les requêtes d'exploration les plus utiles

### Compter et agréger par groupe

```sql
SELECT hotel_id,
       COUNT(*)                AS nb_reservations,
       AVG(stays_in_week_nights) AS nuits_semaine_moyennes
FROM reservations
GROUP BY hotel_id
ORDER BY hotel_id;
```

**Astuce** : la moyenne d'une colonne qui vaut 0 ou 1 est un **taux**. `AVG(colonne_0_1)` donne directement une proportion.

### Arrondir

`ROUND(valeur, 2)` fonctionne sur le type `NUMERIC`, pas sur les nombres à virgule flottante. Sur une moyenne, convertis d'abord :

```sql
ROUND(AVG(adr)::numeric, 2)
```

### Créer des tranches avec `CASE WHEN`

```sql
SELECT CASE
           WHEN adults = 1 THEN '1 adulte'
           WHEN adults = 2 THEN '2 adultes'
           ELSE '3 adultes ou plus'
       END AS groupe,
       COUNT(*) AS nb
FROM reservations
GROUP BY groupe
ORDER BY groupe;
```

### Joindre deux tables

Une table de faits (`reservations`) renvoie vers une table de référence (`hotels`) par une clé (`hotel_id`) :

```sql
SELECT h.nom, r.arrival_date, r.adr
FROM reservations AS r
JOIN hotels AS h ON h.hotel_id = r.hotel_id
LIMIT 5;
```

Les alias (`r`, `h`) évitent de répéter le nom des tables et lèvent les ambiguïtés quand deux tables ont une colonne de même nom.

### Filtrer sur des dates

```sql
SELECT COUNT(*)
FROM reservations
WHERE arrival_date BETWEEN '2016-01-01' AND '2016-03-31';
```



## Pour aller plus loin

- `pandas.read_sql` : https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html
- Tutoriel SQL de PostgreSQL (en anglais) : https://www.postgresql.org/docs/current/tutorial-sql.html
