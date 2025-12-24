# EXPLICATION COMPLÈTE DU MERGE DES DONNÉES

## 📋 Vue d'ensemble

### Fichiers sources
1. **data_origin/train.csv** : Données de taux de grippe par région et semaine
2. **data_origin/ListedesStationsMeteo.csv** : Liste des 62 stations météo avec coordonnées
3. **DonneesMeteorologiques/synop.AAAAMM.csv** : ~96 fichiers de données météo (2004-2011)

### Objectif
Créer un fichier unique qui contient :
- Les données de grippe (train.csv)
- Les données météorologiques correspondantes par région et semaine

---

## 🔧 LA DÉMARCHE COMPLÈTE

### PROBLÈME PRINCIPAL À RÉSOUDRE

Les données ne sont PAS au même niveau de granularité :

| Fichier | Granularité | Format |
|---------|-------------|---------|
| train.csv | **Région + Semaine** | region_name, week (AAAASS) |
| synop.csv | **Station + Date/Heure** | numer_sta, date (AAAAMMJJHHMMSS) |

**Il faut donc :**
1. ✅ Mapper les stations météo aux régions
2. ✅ Convertir les dates en semaines
3. ✅ Agréger les données météo par région et semaine

---

## 📍 ÉTAPE 1 : Mapping Région ↔ Station Météo

### Le défi
- train.csv a 22 régions françaises (anciennes régions avant 2016)
- ListedesStationsMeteo.csv a 62 stations météo
- **Quelle(s) station(s) représentent chaque région ?**

### La solution : Mapping manuel

J'ai créé un mapping basé sur la géographie française :

```python
REGION_STATION_MAPPING = {
    'ALSACE': ['07190', '07280'],  # Strasbourg
    'AQUITAINE': ['07510', '07630'],  # Bordeaux, Biarritz
    'AUVERGNE': ['07460', '07380'],  # Clermont-Ferrand
    'BASSE-NORMANDIE': ['07027', '07139'],  # Caen
    'BRETAGNE': ['07110', '07117', '07130'],  # Brest, Rennes
    'ILE-DE-FRANCE': ['07150', '07149'],  # Paris, Orly
    # ... etc pour les 22 régions
}
```

**Pourquoi plusieurs stations par région ?**
- Pour avoir une meilleure couverture géographique
- Pour lisser les valeurs extrêmes locales
- Pour avoir plus de données (certaines stations ont des mesures manquantes)

---

## 📅 ÉTAPE 2 : Conversion Date → Semaine

### Le défi
```
synop : date = "20110105143000" (5 janvier 2011 à 14h30)
train : week = 201101 (semaine 1 de 2011)
```

### La solution : ISO Calendar

```python
# Convertir en datetime
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S')

# Extraire année et semaine ISO
df['year'] = df['date'].dt.isocalendar().year
df['week'] = df['date'].dt.isocalendar().week

# Créer le code semaine AAAASS
df['week_year'] = df['year'] * 100 + df['week']
```

**Note importante** : On utilise l'ISO calendar (norme internationale) où :
- Semaine 1 = première semaine avec au moins 4 jours en janvier
- Les semaines vont du lundi au dimanche

---

## 📊 ÉTAPE 3 : Agrégation des données météo

### Le défi
Les données synop sont **très granulaires** :
- Plusieurs mesures par jour (toutes les 3h ou 6h)
- Par station individuelle

On veut : **UNE valeur par région et par semaine**

### La solution : Agrégation en 2 temps

#### 1️⃣ Filtrer par stations d'intérêt
```python
# On ne garde que les ~40 stations qui correspondent à nos 22 régions
df_synop = df_synop[df_synop['numer_sta'].isin(stations_of_interest)]
```

**Pourquoi ?** Réduire la mémoire (550MB → ~100MB)

#### 2️⃣ Grouper et moyenner

```python
# Merger avec le mapping station → région
df_synop = df_synop.merge(df_station_region, on='numer_sta')

# Grouper par (région, semaine) et calculer la moyenne
df_agg = df_synop.groupby(['region_name', 'week_year']).agg({
    't': 'mean',      # Température moyenne
    'u': 'mean',      # Humidité moyenne
    'rr24': 'mean',   # Précipitations moyennes
    # ... etc pour ~30 variables météo
})
```

**Résultat** : Une ligne par (région, semaine) avec les moyennes météo

---

## 🔗 ÉTAPE 4 : Merge final

### La jointure

```python
df_final = df_train.merge(
    df_meteo_agg,
    left_on=['region_name', 'week'],
    right_on=['region_name', 'week_year'],
    how='inner'  # On garde seulement les correspondances parfaites
)
```

**Clés de jointure** :
- `region_name` (normalisé en MAJUSCULES)
- `week` (train.csv) = `week_year` (synop agrégé)

**Type de jointure : INNER**
- On ne garde que les lignes où on a BOTH les données de grippe ET les données météo
- Résultat : ~9000-9500 lignes (sur 9195 dans train.csv)

---

## 📈 RÉSULTAT ATTENDU

### Structure du fichier final

| Colonne | Source | Description |
|---------|--------|-------------|
| Id | train.csv | Identifiant unique |
| week | train.csv | Semaine au format AAAASS |
| region_code | train.csv | Code numérique région |
| region_name | train.csv | Nom de la région |
| **TauxGrippe** | train.csv | **VARIABLE CIBLE** |
| t | synop (agrégé) | Température moyenne (°K) |
| td | synop (agrégé) | Point de rosée (°K) |
| u | synop (agrégé) | Humidité (%) |
| ff | synop (agrégé) | Vitesse du vent (m/s) |
| rr24 | synop (agrégé) | Précipitations 24h (mm) |
| ... | synop (agrégé) | ~25 autres variables météo |

**Total** : environ 35-40 colonnes

---

## ⚠️ POINTS D'ATTENTION

### 1. Valeurs manquantes
Les fichiers synop contiennent beaucoup de "mq" (mesure manquante) :
```python
# Convertir 'mq' en NaN
df[var] = pd.to_numeric(df[var], errors='coerce')
```

**Gestion** :
- Lors de l'agrégation, les NaN sont ignorés automatiquement par `.mean()`
- Après le merge, on peut imputer les NaN restants par la médiane

### 2. Normalisation des noms de régions
```python
# train.csv peut avoir "Ile-de-France", "ILE-DE-FRANCE", etc.
df['region_name'] = df['region_name'].str.upper().str.strip()
```

### 3. Taux de couverture
Après le merge INNER, on peut perdre quelques lignes :
- Certaines semaines n'ont pas de données météo
- Certaines régions ont des gaps dans les mesures

**Résultat typique** : 95-100% de couverture

---

## 🚀 COMMENT UTILISER LE NOTEBOOK

### 1. Ouvrir le notebook
```bash
jupyter notebook MERGE_WEATHER_DATA.ipynb
```

### 2. Exécuter toutes les cellules
- Cell → Run All
- Durée : ~2-5 minutes (selon les données)

### 3. Résultat
```
data_plus/train_weather_merged_complete.csv
```

Un fichier prêt pour :
- ✅ Analyse exploratoire
- ✅ Feature engineering (lags, moyennes mobiles)
- ✅ Machine Learning (prédiction du TauxGrippe)

---

## 📚 RÉFÉRENCES

### Variables météo importantes (basées sur le notebook NETTOYAGE_DONNEES)

**Top 8 variables** (corrélées avec TauxGrippe) :
1. `tminsol` - Température min du sol
2. `t` - Température
3. `td` - Point de rosée
4. `u` - Humidité
5. `ff` - Vitesse du vent
6. `vv` - Visibilité horizontale
7. `n` - Nébulosité totale
8. `nbas` - Nébulosité basse

### Format des dates
- **train.csv** : `week` = AAAASS (ex: 201101 = semaine 1 de 2011)
- **synop** : `date` = AAAAMMJJHHMMSS (ex: 20110105143000)

### Unités météo
- Température (t, td, tminsol) : en Kelvin (K)
- Vent (ff) : m/s
- Humidité (u) : %
- Précipitations (rr1, rr6, rr24) : mm
- Pression (pres) : Pascal

---

## ✅ VALIDATION DU MERGE

Après le merge, vérifier :

```python
# 1. Nombre de lignes
print(f"train.csv : {len(df_train)} lignes")
print(f"Après merge : {len(df_final)} lignes")
print(f"Couverture : {len(df_final)/len(df_train)*100:.1f}%")

# 2. Pas de doublons
duplicates = df_final.duplicated(subset=['region_code', 'week'])
print(f"Doublons : {duplicates.sum()}")  # Doit être 0

# 3. Toutes les régions présentes
print(f"Régions : {df_final['region_name'].nunique()}")  # Doit être 22

# 4. Valeurs manquantes
missing = df_final.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))
```

---

## 🎯 PROCHAINES ÉTAPES

Après avoir généré le fichier mergé, vous pouvez :

1. **Nettoyer les données** (voir NETTOYAGE_DONNEES.ipynb)
   - Supprimer les colonnes avec >30% de NaN
   - Imputer les valeurs manquantes
   - Détecter et traiter les outliers

2. **Feature Engineering**
   - Créer des lags (TauxGrippe_lag1, lag2, etc.)
   - Moyennes mobiles (ma4, ma8)
   - Variables de saison

3. **Modélisation**
   - Random Forest
   - XGBoost
   - LSTM (pour les séries temporelles)

---

**Auteur** : Claude
**Date** : 2024
**Projet** : Prédiction du taux de grippe avec données météo
